from flask import Flask, request, jsonify
import requests, os
import time
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import re
from requests.adapters import HTTPAdapter
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from models import Base, ConvoChunk
from datetime import datetime, timedelta, timezone
from sqlalchemy import and_
from urllib3.util import Retry


CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
# os.makedirs(PROFILES_DIR, exist_ok=True)

DATA_ROOT = os.environ.get("DATA_ROOT", "/var/data")

PROFILES_DIR = os.path.join(DATA_ROOT, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

API_KEY = os.environ.get("CLAUDE_API_KEY")

_ALLOWED_PROFILE_KEYS = {
    "name", "age", "gender",
    "likes", "dislikes", "parent_name", "pronouns",
    "recent_summary"
}

PROMPT_HE = (
    "אתה דמות של דובי פַּנְדָּה חברותי בשם פֶּנְדִּי. דבר ישירות בגובה עיניים לילדים בני 3–6.\n"
    "ענה תמיד בעברית בלבד. הוסף ניקוד מלא ומדויק לכל מילה, בהתאם למגדר הילד.\n"
    "משפטים קצרים ותשובות קצרות וישירות. שמור על שיחה רציפה.\n"
    "בלי אימוג׳ים ובלי סימנים מיותרים.\n"
    "קרא לילד בשמו אם ידוע.\n"
    "אל תתחיל משפטים ב-'פנדי:'.\n"
    "ענה בסגנון חם, סבלני ואוהב; עודד בעדינות והימנע מביקורת.\n"
    "הצע משחקי דמיון, סיפורים ושירים פשוטים.\n"
    "אל תחזור על אותו שיר או סיפור; שמור על מקוריות.\n"
    "זהה ואשר רגשות; הצע הפסקות כשצריך."
)

PROMPT_EN = (
    "You are a friendly panda bear character named Pendi. Speak directly at eye level to children ages 3–6.\n"
    "Always respond in English only. Speak according to the child’s gender.\n"
    "Use short sentences and short, direct answers. Keep the conversation flowing.\n"
    "No emojis and no unnecessary symbols.\n"
    "Call the child by name if known.\n"
    "Do not begin sentences with 'Pendi:'.\n"
    "Respond in a warm, patient, and loving style; encourage gently and avoid criticism.\n"
    "Suggest imagination games, simple stories, and songs.\n"
    "Do not repeat the same song or story; keep responses original.\n"
    "Recognize and affirm feelings; suggest breaks when needed."
)

PROMPT_AR = (
 "أنتَ شَخصِيَّةُ دُبّ باندا وُدِّيٍّ باسم بِنْدي. تَكَلَّمْ مُباشَرَةً بِمُستَوى أعيُنِ الأطفالِ في سِنّ ٣–٦.\n"
    "أَجِبْ دائِمًا بِاللُّغَةِ العَرَبِيَّةِ فَقَط. أَضِفْ تَشْكِيلًا كامِلًا ودَقيقًا لِكُلِّ كَلِمَة، وَفْقًا لِجِنسِ الطِّفل.\n"
    "اِستَخدِمْ جُمَلًا قَصِيرَة وَإِجابَات قَصِيرَة ومُباشَرَة. اِحفَظْ سَيرَ المُحادَثَة بِسَلاسَة.\n"
    "بِدونِ إيموجي وَبِدونِ رُموزٍ زائِدَة.\n"
    "نادِ الطِّفلَ بِاسْمِهِ إِذا كانَ مَعرُوفًا.\n"
    "لا تَبدَأ الجُمَل بِـ 'بِنْدي:'.\n"
    "أَجِبْ بِأُسلوبٍ دافِئ، صَبُور وَمُحِبّ؛ شَجِّعْ بِلُطف وَتَجَنَّبِ النَّقد.\n"
    "اِقتَرِحْ أَلعابَ خَيال، قِصَصًا وَأَغانِي بَسيطَة.\n"
    "لا تُكرِّر نَفس الأُغنِيَة أَو القِصَّة؛ اِحفَظْ الأَصالَة.\n"
    "تَعَرَّفْ وَأَكِّدْ المَشاعِر؛ وَاقتَرِحِ اِستِراحات عِندَ الحاجَة."
)

# retry = Retry(
#     total=2,
#     connect=2, read=2, status=2,
#     backoff_factor=0.2,
#     status_forcelist=[429, 500, 502, 503, 504],
#     allowed_methods=frozenset(["GET", "POST"])
# )

adapter = HTTPAdapter(
    pool_connections=20,
    pool_maxsize=50,
    max_retries=0
)

http = requests.Session()

http.mount("https://", adapter)
http.mount("http://", adapter)
http.headers.update({"Connection": "keep-alive"})

bg = ThreadPoolExecutor(max_workers=6, thread_name_prefix="bg")

db_abs = os.path.join(DATA_ROOT, "rag.db")

app = Flask(__name__)

# Load data - api key, model...
app.config.from_pyfile("config.py", silent=True)

cfg_db_url = app.config.get("DB_URL")

# db_abs = os.path.join(BASE_DIR, "rag.db")

if not cfg_db_url:
    app.config["DB_URL"] = f"sqlite:///{db_abs}"

elif cfg_db_url.startswith("sqlite:///") and not cfg_db_url.startswith("sqlite:////"):
    rel = cfg_db_url.replace("sqlite:///", "", 1)
    app.config["DB_URL"] = f"sqlite:///{os.path.join(DATA_ROOT, rel)}"

app.config.from_mapping({
    "ANTHROPIC_MODEL": app.config.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
    "MAX_TOKENS": int(app.config.get("MAX_TOKENS", 500)),
    "PORT": int(app.config.get("PORT", 5001)),
    "DB_URL": app.config["DB_URL"]
})

engine = create_engine(
    app.config["DB_URL"],
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if app.config["DB_URL"].startswith("sqlite") else {}
    )

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

Base.metadata.create_all(engine)

print("Panda server is running")

@app.route("/api/latest-version", methods=["GET"])
def latest_version():
    metadata_path = os.path.join("static", "apks", "output-metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    element = metadata["elements"][0]  
    version_code = element["versionCode"]
    version_name = element["versionName"]
    apk_file = element["outputFile"]

    return jsonify({
        "versionCode": version_code, 
        "versionName": version_name,
        "apkUrl": f"https://github.com/Links-AI-IL/MiniServer/releases/download/{version_name}/{apk_file}",
        "releaseNotes": "✨ גרסה חדשה זמינה"
    })

# Healthz test to api
@app.get("/healthz")
def healthz():
    if not API_KEY:
        return jsonify(status="error", error="API_KEY missing"), 500
    return jsonify("Panda server is working!")

# Return miss profile details
def profile_collect_instruction(profile: dict) -> str | None:
    field_labels = {"name": "שם", "age": "גיל", "gender": "מגדר"}
    missing = [k for k in ("name", "age", "gender") if not profile.get(k)]
    if not missing:
        return None
    readable = ", ".join(field_labels[k] for k in missing)
    return (
        "חסרים בפרופיל המשתמש: " + readable + ". "
        "שאל בעדינות, שאלה אחת בכל פעם, כדי לאסוף רק את השדות החסרים. "
        "השתמש במשפט קצר ומנוקד. "
        "אחרי שקיבלת תשובה, אל תשאל שוב על אותו שדה. "
        "אל תבקש פרטים מזהים נוספים."
    )

# Collect message or messages
def _normalize_messages(messages, question):
    if messages and isinstance(messages, list):
        return messages
    q = (question or "").strip()
    if not q:
        return None
    return [{"role": "user", "content": [{"type": "text", "text": q}]}]

# Get clean text to context
def _extract_text_blocks(content_list):
    out = []
    for b in content_list or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return "\n".join([t for t in out if t])

# Get last message from user 
def _last_user_text_from_messages(built_messages):
    for msg in reversed(built_messages):
        if msg.get("role") == "user":
            return _extract_text_blocks(msg.get("content"))
    return ""

# Save messages to DB in background
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

# General headers
def anthropic_headers():
    return {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "prompt-caching-2024-07-31",
        "Content-Type": "application/json"
    }


# Create JSON object of 3 details
def _build_last_turn_json(profile: dict, built_msgs) -> str | None:

    current_user = (_last_user_text_from_messages(built_msgs) or "").strip()[:400]

    last_user = ""
    last_assistant = ""

    if (not last_user or not last_assistant) and profile.get("device_id"):
        db = SessionLocal()
        try:
            rows = (
                db.query(ConvoChunk)
                  .filter(ConvoChunk.device_id == profile["device_id"])
                  .order_by(ConvoChunk.id.desc())
                  .limit(8)
                  .all()
            )
        finally:
            db.close()

        if not last_assistant:
            for r in rows:
                if r.role == "assistant" and (r.text or "").strip():
                    last_assistant = (r.text or "").strip()[:400]
                    break

        if not last_user:
            for r in rows:
                if r.role == "user":
                    cand = (r.text or "").strip()
                    if cand and cand != current_user:
                        last_user = cand[:400]
                        break

    if not (last_user or last_assistant or current_user):
        return None

    obj = {
        "last_user": last_user or None,
        "last_assistant": last_assistant or None,
        "current_user": current_user or None,
    }
    return json.dumps(obj, ensure_ascii=False)


# Send query to api
@app.post("/query")
def query():
    if not API_KEY:
        return jsonify({"error": "API_KEY not configured"}), 500

    data = request.get_json(force=True) or {}

    device_id = (data.get("device_id") or "dev").strip()
    messages = data.get("messages")
    question = data.get("question")
    language = (data.get("language") or "HE")

    built = _normalize_messages(messages, question)
    if not built:
        return jsonify({"error": "missing messages or question"}), 400
    
    profile = load_profile(device_id)
    profile["device_id"] = device_id

    profile_ctx = format_profile_for_system(profile)

    if language == "AR":
        Specific_prompt = PROMPT_AR
    elif language == "EN":
        Specific_prompt = PROMPT_EN
    else:
        Specific_prompt = PROMPT_HE

    to_system = [
        {"type": "text", "text": Specific_prompt}, 
        {"type": "text", "text": profile_ctx},
    ]

    lt_json = _build_last_turn_json(profile, built)
    if lt_json:
        to_system.append({
            "type": "text",
            "text": (
                "רצף אחרון (אל תקרא/תצטט לילד):\n" + lt_json
            )
        })
        
    print("last-turn-json:", lt_json)

    instr = profile_collect_instruction(profile)
    if instr:
        to_system.append({"type": "text", "text": instr})

    headers = anthropic_headers()

    print(f"to system = {to_system}")

    payload = {
        "model": app.config["ANTHROPIC_MODEL"],
        "system": to_system,
        "messages": built,
        "max_tokens": int(app.config["MAX_TOKENS"]),
        "temperature": 0.0,
        "stream": False
    }

    t0 = time.perf_counter()

    r = None
    error = None
    
    for attempt in range(2):
        try:
            r = http.post(
                CLAUDE_API_URL,
                headers=headers,
                json=payload,
                timeout=(4, 25)
            )
            print(f"[anthropic-status] {r.status_code} (attempt {attempt+1})")

            if r.status_code // 100 == 2:
                break

        except requests.RequestException as e:
            error = e
            print(f"[anthropic-error] {e} (attempt {attempt+1})")

            if attempt < 1:
                continue  

    if r is None or r.status_code // 100 != 2:
        return jsonify({
        "error": "LLM upstream error",
        "detail": str(error) if error else f"HTTP {r.status_code if r else 'no response'}"
    }), 502

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

    fut = bg.submit(_persist_interaction_async, device_id, user_text, assistant_text)
    fut.add_done_callback(lambda _: bg.submit(prune_conversation, device_id))

    if user_text:
        bg.submit(_maybe_extract_profile_async, device_id, user_text, profile)

    Response = resp_json["content"][0]["text"]

    Clean_response = Response.replace("\n", " ")

    return jsonify({"response": Clean_response})


# Debug last message
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


# Debug load config
@app.get("/debug/config")
def debug_config():
    import os
    return jsonify({
        "root_config_exists": os.path.exists(os.path.join(app.root_path, "config.py")),
        "instance_config_exists": os.path.exists(os.path.join(app.instance_path, "config.py")),
        "has_api_key": bool(API_KEY),
        "instance_path": app.instance_path,
        "root_path": app.root_path,
        "model": app.config.get("ANTHROPIC_MODEL"),
        "db_url": app.config.get("DB_URL"),
    })


# Clean device id
def profile_path(device_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", device_id or "dev")
    return os.path.join(PROFILES_DIR, f"{safe}.json")


# Load json profile
def load_profile(device_id: str) -> dict:
    path = profile_path(device_id)
    if not os.path.exists(path):
        return {
            "name": None, "age": None, "gender": None,
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


# Save and update user profile
def save_profile(device_id: str, patch: dict) -> dict:
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

        elif k in {"name", "gender", "parent_name", "pronouns"}:
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
        elif k == "recent_summary":
            v = (str(v).strip()[:800] or None) if v is not None else None
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

    path = profile_path(device_id)
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

# Create user profile to context
def format_profile_for_system(p: dict) -> str:
    name = p.get("name"); age = p.get("age"); gender = p.get("gender")
    return f"פרופיל משתמש: שם: {name or 'לא ידוע'}, גיל: {age or 'לא ידוע'}, מגדר: {gender or 'לא ידוע'}."


# Get profile
@app.get("/profile")
def get_profile():
    device_id = (request.args.get("device_id") or "dev").strip()
    return jsonify(load_profile(device_id))


# Clean old DB
def prune_conversation(device_id: str):
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=6)

        keep_rows = (
            db.query(ConvoChunk.id)
              .filter(ConvoChunk.device_id == device_id)
              .order_by(ConvoChunk.id.desc())
              .limit(10)
              .all()
        )
        keep_ids = [r.id for r in keep_rows] or [-1] 

        if keep_ids:
            db.query(ConvoChunk).filter(
                ConvoChunk.device_id == device_id,
                and_(ConvoChunk.ts < cutoff, ~ConvoChunk.id.in_(keep_ids))
            ).delete(synchronize_session=False)
        else:
            db.query(ConvoChunk).filter(
                ConvoChunk.device_id == device_id,
                ConvoChunk.ts < cutoff
            ).delete(synchronize_session=False)

        db.commit()
        print("delete successfully")
    finally:
        db.close()


# Create user profile by device id
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


# Extracting profile from user - in background
def _maybe_extract_profile_async(device_id: str, user_text: str, current_profile: dict):
    miss = [k for k in ("name", "age", "gender") if not current_profile.get(k)]
    should_try = bool(miss) or any(not current_profile.get(k) for k in ("likes", "dislikes"))
    if not should_try or not user_text:
        return

    headers = anthropic_headers()

    extractor_system = (
        "אתה מחלץ פרטי פרופיל מילד מטקסט חופשי. החזר JSON חוקי בלבד, ללא טקסט נוסף. "
        "השדות המותרים: name (string), age (int), gender (\"בן\"/\"בת\"/null), "
        "likes (list of strings), dislikes (list of strings). "
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
        resp = http.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=(4, 15))
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