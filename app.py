from flask import Flask, request, jsonify
import requests, os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, ConvoChunk
import time
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import re
from datetime import datetime
import pytz

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CORE_FIELDS = ("name", "age", "gender")

PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

_ALLOWED_PROFILE_KEYS = {
    "name", "age", "gender", "nickname",
    "likes", "dislikes", "parent_name", "pronouns"
}

http = requests.Session()

bg = ThreadPoolExecutor(max_workers=2)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")

app = Flask(__name__)

app.config.from_pyfile("config.py", silent=True)

app.config.from_mapping({
    "ANTHROPIC_MODEL": app.config.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
    "MAX_TOKENS": int(app.config.get("MAX_TOKENS", 500)),
    "PORT": int(app.config.get("PORT", 5001)),
    "DB_URL": app.config.get("DB_URL", "sqlite:///rag.db"),
})

print("Loaded ANTHROPIC_API_KEY:", "YES" if app.config.get("ANTHROPIC_API_KEY") else "NO")

engine = create_engine(app.config["DB_URL"], echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base.metadata.create_all(engine)

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

PROMPT = (
    "אתה דמות של דובי פַּנְדָּה חברותי בשם פֶּנְדִּי. דבר ישירות בגובה עיניים לילדים בני 3–6.\n"
    "ענה תמיד בעברית בלבד.\n"
    "הוסף ניקוד מלא ומדויק לכל מילה, בהתאם למגדר הילד.\n"
    "משפטים קצרים: 4–6 מילים בכל משפט.\n"
    "בלי אימוג׳ים ובלי סימנים מיותרים.\n"
    "לאחר מכן קרא לילד בשמו אם ידוע.\n"
    "ענה בתשובות מפורטות ומורחבות, לפחות 3–5 משפטים בכל תשובה.\n"
    "סגנון חם, סבלני ואוהב; עודד בעדינות והימנע מביקורת.\n"
    "הצע משחקי דמיון, סיפורים ושירים פשוטים.\n"
    "אל תחזור על אותו שיר או סיפור; שמור על מקוריות.\n"
    "זהה ואשר רגשות; הצע הפסקות כשצריך."
)

def missing_core_fields(profile: dict) -> list[str]:
    return [k for k in CORE_FIELDS if not profile.get(k)]

def build_profile_collect_instruction(missing: list[str]) -> str:
    readable = ", ".join({"name": "שם", "age": "גיל", "gender": "מגדר"}[k] for k in missing)
    return (
        "חסרים בפרופיל המשתמש: " + readable + ". "
        "שאל בעדינות לאט ובדרך אגב, שאלה אחת בכל פעם, כדי לאסוף רק את השדות החסרים. "
        "השתמש במשפט קצר ומנוקד. "
        "אחרי שקיבלת תשובה, אל תשאל שוב על אותו שדה. "
        "אל תבקש פרטים מזהים נוספים."
    )

def _normalize_messages(messages, question):
    if messages and isinstance(messages, list):
        return messages
    q = (question or "").strip()
    if not q:
        return None
    return [{"role": "user", "content": [{"type": "text", "text": q}]}]


def _extract_text_blocks(content_list):
    """מקבל content מ-Anthropic ומחזיר טקסט מאוחד."""
    out = []
    for b in content_list or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return "\n".join([t for t in out if t])


def _last_user_text_from_messages(built_messages):
    """מחלץ את טקסט המשתמש האחרון שנשלח למודל (לשמירה ב-DB)."""
    for msg in reversed(built_messages):
        if msg.get("role") == "user":
            return _extract_text_blocks(msg.get("content"))
    return ""


def _persist_interaction_async(device_id: str, user_text: str, assistant_text: str):
    """שמירה ל-DB ברקע: לא חוסם את מסלול התשובה."""
    db = SessionLocal()
    try:
        if user_text:
            db.add(ConvoChunk(device_id=device_id, role="user", text=user_text))
        if assistant_text:
            db.add(ConvoChunk(device_id=device_id, role="assistant", text=assistant_text))
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        try:
            db.close()
        except Exception:
            pass


@app.get("/healthz")
def healthz():
    if not app.config.get("ANTHROPIC_API_KEY"):
        return jsonify(status="error", error="ANTHROPIC_API_KEY missing"), 500
    return jsonify(status="ok")


@app.post("/query")
def query():
    if not app.config.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500

    data = request.get_json(force=True) or {}
    device_id = (data.get("device_id") or "dev").strip()

    messages = data.get("messages")
    question = data.get("question")
    built = _normalize_messages(messages, question)
    if not built:
        return jsonify({"error": "missing messages or question"}), 400
    
    profile = load_profile(device_id)
    profile_ctx = format_profile_for_system(profile)

    to_system = [
        {"type": "text", "text": PROMPT, "cache_control": {"type": "ephemeral"}}, 
        {"type": "text", "text": profile_ctx},
    ]

    if not profile.get("core_collected"):
        missing = [k for k in ("name", "age", "gender") if not profile.get(k)]
        if missing:
            to_system.append({
                "type": "text",
                "text": build_profile_collect_instruction(missing)
            })

    headers = {
        "x-api-key": app.config["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "prompt-caching-2024-07-31",
        "Content-Type": "application/json"
    }
    payload = {
        "model": app.config["ANTHROPIC_MODEL"],
        "system": to_system,
        "messages": built,
        "max_tokens": int(app.config["MAX_TOKENS"]),
        "stream": False
    }

    t0 = time.perf_counter()
    try:
        r = http.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=(5, 45))
    except requests.RequestException as e:
        return jsonify({"error": "LLM upstream error", "detail": str(e)}), 502

    t1 = time.perf_counter()
    llm_ms = int((t1 - t0) * 1000)
    print(f"[LLM] round-trip = {llm_ms} ms")

    if r.status_code // 100 != 2:
        try:
            body = r.json()
        except ValueError:
            body = {"raw": r.text[:500]}
        return jsonify({"error": "LLM non-2xx", "status": r.status_code, "body": body}), 502

    resp_json = r.json()

    user_text = _last_user_text_from_messages(built)
    assistant_text = _extract_text_blocks(resp_json.get("content"))

    bg.submit(_persist_interaction_async, device_id, user_text, assistant_text)

    if user_text:
        bg.submit(_maybe_extract_profile_async, device_id, user_text, profile)

    return jsonify(resp_json)


@app.get("/debug/chunks")
def debug_chunks():
    device_id = request.args.get("device_id", "dev")
    try:
        limit = int(request.args.get("limit", "10"))
    except ValueError:
        limit = 10

    db = SessionLocal()
    try:
        rows = (
            db.query(ConvoChunk)
              .filter(ConvoChunk.device_id == device_id)
              .order_by(ConvoChunk.id.desc())
              .limit(limit)
              .all()
        )

        out = [{
            "id": r.id,
            "device_id": r.device_id,
            "role": r.role,
            "text": r.text,
            "ts": r.ts.replace(microsecond=0).isoformat() if r.ts else None
        } for r in rows]
        return jsonify(out)
    finally:
        db.close()


@app.get("/debug/config")
def debug_config():
    import os
    return jsonify({
        "root_config_exists": os.path.exists(os.path.join(app.root_path, "config.py")),
        "instance_config_exists": os.path.exists(os.path.join(app.instance_path, "config.py")),
        "has_api_key": bool(app.config.get("ANTHROPIC_API_KEY")),
        "instance_path": app.instance_path,
        "root_path": app.root_path,
        "model": app.config.get("ANTHROPIC_MODEL"),
        "db_url": app.config.get("DB_URL"),
    })


def _safe_id(device_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", device_id or "dev")

def _profile_path(device_id: str) -> str:
    return os.path.join(PROFILES_DIR, f"{_safe_id(device_id)}.json")

def load_profile(device_id: str) -> dict:
    path = _profile_path(device_id)
    if not os.path.exists(path):
        return {
            "name": None, "age": None, "gender": None,
            "nickname": None,
            "likes": [], "dislikes": [],
            "parent_name": None, "pronouns": None
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
    except Exception:
        return {}

def save_profile(device_id: str, patch: dict) -> dict:
    """שומר פרופיל באופן אטומי, עם merge לא הורס:
    - לא דורך על ערכים קיימים אם הגיע None/ריק
    - likes/dislikes: מאחד לרשימה ייחודית (לא מוחק קיימים)
    """
    current = load_profile(device_id)

    cleaned = {}
    for k, v in (patch or {}).items():
        if k not in _ALLOWED_PROFILE_KEYS:
            continue

        if k == "age":
            try:
                v = int(v) if v not in (None, "") else None
            except Exception:
                v = None

        elif k in {"name", "nickname", "gender", "parent_name", "pronouns"}:
            if v is None:
                v = None
            else:
                v = str(v).strip()[:50] or None

        elif k in {"likes", "dislikes"}:
            if isinstance(v, list):
                v = [str(x).strip()[:40] for x in v if str(x).strip()][:20]
            elif isinstance(v, str):
                v = [v.strip()[:40]] if v.strip() else []
            else:
                v = []
        cleaned[k] = v

    merged = dict(current or {})
    for k, v in cleaned.items():
        if k in {"likes", "dislikes"}:
            old = merged.get(k) or []
            seen = set()
            new_list = []
            for item in old + v:
                if item and item not in seen:
                    seen.add(item)
                    new_list.append(item)
            merged[k] = new_list[:20]
        else:
            if v is None:
                continue
            merged[k] = v

    merged["core_collected"] = all(bool(merged.get(k)) for k in ("name", "age", "gender"))

    path = _profile_path(device_id)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="profile_", suffix=".json", dir=PROFILES_DIR)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return merged

def format_profile_for_system(p: dict) -> str:
    """ממזער לפרומפט: מידע קצר ורלוונטי בלבד."""
    parts = []
    if p.get("name"): parts.append(f"שם הילד/ה: {p['name']}.")
    if p.get("age"): parts.append(f"גיל: {p['age']}.")
    if p.get("gender"): parts.append(f"מגדר: {p['gender']}.")
    if p.get("nickname"): parts.append(f"כינוי: {p['nickname']}.")
    if p.get("likes"):
        parts.append("מה אוהב/ת: " + " ,".join(p["likes"][:5]) + ".")
    if p.get("dislikes"):
        parts.append("מה פחות אוהב/ת: " + " ,".join(p["dislikes"][:5]) + ".")
    if p.get("parent_name"):
        parts.append(f"שם הורה: {p['parent_name']}.")
    parts.append("דבר תמיד בצורה אישית, פנה בשם הידוע אם קיים.")
    return "פרופיל משתמש:\n" + " ".join(parts)

@app.get("/profile")
def get_profile():
    device_id = (request.args.get("device_id") or "dev").strip()
    return jsonify(load_profile(device_id))

@app.post("/profile")
def upsert_profile():
    data = request.get_json(force=True) or {}
    device_id = (data.get("device_id") or "dev").strip()
    profile_patch = data.get("profile") if isinstance(data.get("profile"), dict) else {
        k: v for k, v in data.items() if k in _ALLOWED_PROFILE_KEYS
    }
    if not profile_patch:
        return jsonify({"error": "no profile fields provided"}), 400
    merged = save_profile(device_id, profile_patch)
    return jsonify({"ok": True, "profile": merged})

def _maybe_extract_profile_async(device_id: str, user_text: str, current_profile: dict):
    miss = missing_core_fields(current_profile)
    should_try = bool(miss) or any(not current_profile.get(k) for k in ("likes", "dislikes"))
    if not should_try or not user_text:
        return

    headers = {
        "x-api-key": app.config["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    extractor_system = (
        "אתה מחלץ פרטי פרופיל מילד מטקסט חופשי. החזר JSON חוקי בלבד, ללא טקסט נוסף. "
        "השדות המותרים: name (string), age (int), gender (\"בן\"/\"בת\"/null), "
        "nickname (string|null), likes (list of strings), dislikes (list of strings). "
        "אם לא בטוח — השאר null/רשימה ריקה."
    )
    extractor_messages = [
        {"role": "user", "content": [{"type": "text", "text": user_text}]}
    ]
    payload = {
        "model": app.config["ANTHROPIC_MODEL"],
        "system": [{"type": "text", "text": extractor_system}],
        "messages": extractor_messages,
        "max_tokens": 180,
        "temperature": 0.0,
        "stream": False
    }
    try:
        resp = http.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=(5, 30))
        if resp.status_code // 100 != 2:
            return
        ans = resp.json()
        text = _extract_text_blocks(ans.get("content"))
        try:
            data = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return
            try:
                data = json.loads(m.group(0))
            except Exception:
                return
        if isinstance(data, dict):
            save_profile(device_id, data)
    except Exception:
        pass

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=int(app.config["PORT"]))
