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

from io import BytesIO
from PIL import Image
import uuid
import base64
import fitz
import pytesseract


CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
# os.makedirs(PROFILES_DIR, exist_ok=True)

DATA_ROOT = os.environ.get("DATA_ROOT", "/var/data")

PROFILES_DIR = os.path.join(DATA_ROOT, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

UPLOADS_DIR = os.path.join(DATA_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

API_KEY = os.environ.get("CLAUDE_API_KEY")

_ALLOWED_PROFILE_KEYS = {
    "name",
    "age",
    "gender",
    "likes",
    "dislikes",
    "parent_name",
    "pronouns",
    "recent_summary",
}

MAX_HISTORY = 30

PROMPT_HE = (
    "אתה דמות של דובי פַּנְדָּה חברותי בשם פֶּנְדִּי. דבר ישירות בגובה עיניים לילדים בני 3–6.\n"
    "ענה תמיד בעברית בלבד. הוסף ניקוד מלא ומדויק לכל מילה, בהתאם למגדר הילד.\n"
    "כתוב תמיד בעברית תקנית וטבעית — אל תשתמש במילים שאינן קיימות בשפה.\n"
    "אם יש ספק בצורה של מילה, השתמש בצורה הנכונה והנפוצה ביותר בעברית.\n"
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

PROMPT_RO = (
    "Ești un urs panda prietenos pe nume Pendi. Vorbește direct, la nivelul ochilor, cu copiii cu vârste între 3 și 6 ani.\n"
    "Răspunde întotdeauna doar în limba română. Vorbește potrivit genului copilului.\n"
    "Folosește propoziții scurte și răspunsuri simple și directe. Menține conversația curgătoare și plăcută.\n"
    "Fără emoji-uri și fără simboluri inutile.\n"
    "Dacă știi numele copilului, adresează-te pe nume.\n"
    "Nu începe propozițiile cu 'Pendi:'.\n"
    "Răspunde într-un stil cald, răbdător și iubitor; încurajează blând și evită critica.\n"
    "Propune jocuri de imaginație, povești scurte și cântecele simple.\n"
    "Nu repeta același cântec sau poveste; păstrează răspunsurile originale.\n"
    "Recunoaște și validează emoțiile copilului; sugerează pauze atunci când este nevoie."
)

PROMPT_MYLO = (
    "אתה דמות חכמה, רגועה וחמה מתוך אפליקציית MILO - אפליקציה לאנשים מבוגרים לשיפור הזיכרון והפעילות המוחית.\n"
    "פנה למשתמש בשמו במידה ויש לך את שמו בהודעה. דבר תמיד בעברית תקנית **עם ניקוד מלא בכל משפט ובכל מילה**, ברורה ונעימה. אל תוותר לעולם על ניקוד מלא, גם בתשובות קצרות או רגשיות.\n"
    "דבר בגובה העיניים, בכבוד ובסבלנות  אל תשתמש בטון מתנשא או מהיר מדי.\n"
    "ענה תשובות קצרות וברורות, אך אנושיות ורגישות.\n"
    "אם המשתמש שואל על האפליקציה או על האימונים — הסבר לו בעדינות כיצד להשתמש במשחקים, איך מתקדמים, ומה המטרה שלהם.\n"
    "אם המשתמש סתם רוצה לשוחח — היה נעים, משתף, מעודד, והצע שיחה חיובית ומכבדת.\n"
    "הימנע מדיונים פוליטיים או נושאים רגישים.\n"
    "הדגש את החשיבות של אימון מוחי יומיומי, שמירה על מצב רוח טוב, וסקרנות מחשבתית.\n"
    "הצג את עצמך כעוזר אישי מתוך MILO, שנמצא כאן כדי לעזור, לעודד, ולהקשיב."
)

SUMMARY_P_HE = """
    אתה מסכם להורי המשתמש סקירה רגשית כללית על הילד/ה, על סמך שיחות רגילות ויומיומיות.

    עקרונות מנחים חשובים:
    - נקודת המוצא היא שהילד/ה מתפתח/ת באופן תקין ובריא.
    - שינויים יומיומיים במצב רוח (עייפות, ערנות, רצון לשחק, רצון לנוח, שינוי דעה) הם טבעיים ונורמליים לגיל זה.
    - אל תחפש בעיות אם אין אינדיקציה ברורה, חוזרת ומתמשכת למצוקה.
    - אל תפרש אמירה בודדת או אירוע חד-פעמי כסימן לקושי רגשי.
    - ציין רגישות או קושי רק אם קיים דפוס עקבי וברור לאורך מספר שיחות.
    - אם לא זוהה דפוס חריג – כתוב זאת במפורש בצורה מרגיעה וברורה.

    סגנון כתיבה:
    - כתוב בטון רגוע, מאוזן ותומך.
    - הימנע משפה דרמטית או מרמזת לבעיה.
    - אל תשתמש במונחים קליניים או מאבחנים.
    - אל תצטט משפטים מלאים מהשיחה.
    - הצג תצפיות כלליות בלבד.

    מבנה הסיכום:
    1. מצב רגשי כללי (כולל הדגשה אם ההתנהגות תואמת גיל והתפתחות תקינה)
    2. נושאים שחזרו באופן משמעותי בלבד
    3. נקודות חוזק וחיוביות
    4. המלצה עדינה לשיחה – רק אם יש בכך ערך אמיתי להורה

    בסיום הוסף פסקה קצרה וברורה:
    "סיכום זה מבוסס על שיחות יומיומיות באפליקציה ואינו מהווה אבחנה רפואית או חוות דעת מקצועית."

    השפה של הסיכום חייבת להיות בהתאם לשפה המבוקשת בלבד.
    """

PROMPT_URBANX = (
    "You are UrbanX AI, a professional assistant for the construction, engineering, real estate, and infrastructure industries in Israel.\n\n"
    "You assist professionals such as contractors, engineers, architects, developers, project managers, appraisers, and real estate brokers.\n"
    "Your role includes helping with planning, analysis, documentation, reports, and decision-making related to:\n"
    "- Construction and infrastructure projects\n"
    "- Engineering and architectural planning\n"
    "- Quantity takeoffs and bills of quantities\n"
    "- Project feasibility and cost analysis\n"
    "- Bureaucratic, regulatory, and permitting processes\n"
    "- Legal and contractual considerations under current Israeli law\n"
    "- Professional reports and structured documents\n\n"
    "You must strictly follow these rules:\n"
    "- Always respond in the same language as the user.\n"
    "- If the user writes in Hebrew, write in clear, professional Hebrew WITHOUT diacritics.\n"
    "- Maintain a factual, precise, and professional tone.\n"
    "- Do NOT use emojis.\n"
    "- Do NOT add personal opinions or emotional language.\n"
    "- Do NOT mention being an AI or explain internal reasoning.\n\n"
    "Context handling:\n"
    "- You receive the full conversation history and must treat this as an ongoing project.\n"
    "- Maintain continuity with previously discussed topics, assumptions, and decisions.\n"
    "- Do not restate information unless necessary for clarity.\n\n"
    "Output guidelines:\n"
    "- Prefer structured answers: sections, bullet points, or tables when appropriate.\n"
    "- When generating reports or documents, use professional headings and formatting.\n"
    "- Base all answers on current Israeli standards, regulations, and professional practice.\n\n"
    "Project title generation:\n"
    "If explicitly asked to generate or suggest a project name or title:\n"
    "- Return ONLY a short, professional title (2–5 words).\n"
    "- The title must reflect the project’s purpose or domain.\n"
    "- Do NOT include punctuation, quotation marks, explanations, or additional text."
)

# retry = Retry(
#     total=2,
#     connect=2, read=2, status=2,
#     backoff_factor=0.2,
#     status_forcelist=[429, 500, 502, 503, 504],
#     allowed_methods=frozenset(["GET", "POST"])
# )

adapter = HTTPAdapter(pool_connections=20, pool_maxsize=50, max_retries=0)

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

app.config.from_mapping(
    {
        "ANTHROPIC_MODEL": app.config.get("ANTHROPIC_MODEL", "claude-opus-4-20250514"),
        "MAX_TOKENS": int(app.config.get("MAX_TOKENS", 500)),
        "PORT": int(app.config.get("PORT", 5001)),
        "DB_URL": app.config["DB_URL"],
    }
)

engine = create_engine(
    app.config["DB_URL"],
    echo=False,
    future=True,
    connect_args=(
        {"check_same_thread": False}
        if app.config["DB_URL"].startswith("sqlite")
        else {}
    ),
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

    return jsonify(
        {
            "versionCode": version_code,
            "versionName": version_name,
            "apkUrl": f"https://github.com/Links-AI-IL/MiniServer/releases/download/{version_name}/{apk_file}",
            "releaseNotes": "✨ גרסה חדשה זמינה",
        }
    )


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
            db.add(
                ConvoChunk(device_id=device_id, role="assistant", text=assistant_text)
            )
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
        "Content-Type": "application/json",
    }


# Create JSON object of 3 details
def _build_last_turn_json(profile: dict, built_msgs) -> str | None:

    if str(profile.get("device_id", "")).lower() == "mylo":
        current_user = (_last_user_text_from_messages(built_msgs) or "").strip()[:400]
        if not current_user:
            return None
        return json.dumps(
            {"last_user": None, "last_assistant": None, "current_user": current_user},
            ensure_ascii=False,
        )

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
    language = (data.get("language") or "HE").upper()  # Romania version

    built = _normalize_messages(messages, question)
    if not built:
        return jsonify({"error": "missing messages or question"}), 400

    try:
        built = prepare_message_object(built)
    except Exception as e:
        print(f"prepare_message_object failed: {e}")

    profile = load_profile(device_id)
    profile["device_id"] = device_id

    if device_id == "mylo":
        Specific_prompt = PROMPT_MYLO
    elif device_id == "URBANX":
        Specific_prompt = PROMPT_URBANX
    elif language == "AR":
        Specific_prompt = PROMPT_AR
    elif language == "EN":
        Specific_prompt = PROMPT_EN
    elif language == "RO":  # Romania version
        Specific_prompt = PROMPT_RO  # Romania version
    else:
        Specific_prompt = PROMPT_HE

    if device_id == "mylo":
        profile_ctx = "פרופיל משתמש כללי: משתמש באפליקציית MILO."
        skip_profile_ops = True
    elif device_id == "URBANX":
        profile_ctx = "פרופיל משתמש כללי: משתמש באפליקציית URBANX."
        skip_profile_ops = True
    else:
        profile_ctx = format_profile_for_system(profile)
        skip_profile_ops = False

    to_system = [
        {"type": "text", "text": Specific_prompt},
        {"type": "text", "text": profile_ctx},
    ]

    recent_messages = data.get("recent_messages")
    lt_json = None
    if (
        device_id.lower() in {"mylo", "urbanx"}
        and isinstance(recent_messages, list)
        and recent_messages
    ):
        context_snippet = "\n".join(recent_messages[-3:])
        to_system.append(
            {
                "type": "text",
                "text": f"שיחה קודמת (אל תקרא/תצטט למשתמש):\n{context_snippet}",
            }
        )
    else:
        lt_json = _build_last_turn_json(profile, built)
        if lt_json:
            to_system.append(
                {"type": "text", "text": ("רצף אחרון (אל תקרא/תצטט לילד):\n" + lt_json)}
            )

    print("last-turn-json:", lt_json)

    instr = None
    if not skip_profile_ops:
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
        "stream": True,
    }

    t0 = time.perf_counter()

    r = None
    error = None

    for attempt in range(2):
        try:
            with requests.post(
                CLAUDE_API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(4, 60),
            ) as r:
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:") :].strip()
                    if chunk == "[DONE]":
                        break
                    yield f"data: {chunk}\n\n"

            print(f"[anthropic-status] {r.status_code} (attempt {attempt+1})")

            if r.status_code // 100 == 2:
                break

        except requests.RequestException as e:
            error = e
            print(f"[anthropic-error] {e} (attempt {attempt+1})")

            if attempt < 1:
                continue

    if r is None or r.status_code // 100 != 2:
        return (
            jsonify(
                {
                    "error": "LLM upstream error",
                    "detail": (
                        str(error)
                        if error
                        else f"HTTP {r.status_code if r else 'no response'}"
                    ),
                }
            ),
            502,
        )

    t1 = time.perf_counter()
    llm_ms = int((t1 - t0) * 1000)
    print(f"[LLM] round-trip = {llm_ms} ms")

    if r.status_code // 100 != 2:
        try:
            body = r.json()
        except ValueError:
            body = {"raw": r.text[:500]}
        return (
            jsonify({"error": "LLM non-2xx", "status": r.status_code, "body": body}),
            502,
        )

    resp_json = r.json()

    user_text = _last_user_text_from_messages(built)
    assistant_text = _extract_text_blocks(resp_json.get("content"))

    if not skip_profile_ops:
        fut = bg.submit(
            _persist_interaction_async, device_id, user_text, assistant_text
        )
        fut.add_done_callback(lambda _: bg.submit(prune_conversation, device_id))
    if user_text:
        bg.submit(_maybe_extract_profile_async, device_id, user_text, profile)

    Response = resp_json["content"][0]["text"]

    Clean_response = Response.replace("\n", " ")

    return jsonify({"response": Clean_response})


# Send query with stream
@app.post("/query-stream")
def query_stream():
    if not API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500

    data = request.get_json(force=True) or {}
    device_id = (data.get("device_id") or "dev").strip()
    messages = data.get("messages")
    question = data.get("question")
    language = (data.get("language") or "HE").upper()  # Romania version

    built = _normalize_messages(messages, question)
    if not built:
        return jsonify({"error": "missing messages or question"}), 400

    try:
        built = prepare_message_object(built)
    except Exception as e:
        print(f"prepare_message_object failed: {e}")

    profile = load_profile(device_id)
    profile["device_id"] = device_id

    if device_id == "mylo":
        Specific_prompt = PROMPT_MYLO
    elif device_id == "URBANX":
        Specific_prompt = PROMPT_URBANX
    elif language == "AR":
        Specific_prompt = PROMPT_AR
    elif language == "EN":
        Specific_prompt = PROMPT_EN
    elif language == "RO":  # Romania version
        Specific_prompt = PROMPT_RO  # Romania version
    else:
        Specific_prompt = PROMPT_HE

    if device_id == "mylo":
        profile_ctx = "פרופיל משתמש כללי: משתמש באפליקציית MILO."
        skip_profile_ops = True
    elif device_id == "URBANX":
        profile_ctx = "פרופיל משתמש כללי: משתמש באפליקציית URBANX."
        skip_profile_ops = True
    else:
        profile_ctx = format_profile_for_system(profile)
        skip_profile_ops = False

    to_system = [
        {"type": "text", "text": Specific_prompt},
        {"type": "text", "text": profile_ctx},
    ]

    recent_messages = data.get("recent_messages")
    lt_json = None
    if (
        device_id.lower() in {"mylo", "urbanx"}
        and isinstance(recent_messages, list)
        and recent_messages
    ):
        context_snippet = "\n".join(recent_messages[-3:])
        to_system.append(
            {
                "type": "text",
                "text": f"שיחה קודמת (אל תקרא/תצטט למשתמש):\n{context_snippet}",
            }
        )
    else:
        lt_json = _build_last_turn_json(profile, built)
        if lt_json:
            to_system.append(
                {"type": "text", "text": ("רצף אחרון (אל תקרא/תצטט לילד):\n" + lt_json)}
            )

    print("last-turn-json:", lt_json)

    instr = None
    if not skip_profile_ops:
        instr = profile_collect_instruction(profile)
        if instr:
            to_system.append({"type": "text", "text": instr})

    print(f"to system = {to_system}")

    headers = anthropic_headers()
    payload = {
        "model": app.config["ANTHROPIC_MODEL"],
        "system": to_system,
        "messages": built,
        "max_tokens": int(app.config["MAX_TOKENS"]),
        "temperature": 0.0,
        "stream": True,
    }

    def generate():
        assistant_full = []
        with requests.post(
            CLAUDE_API_URL, headers=headers, json=payload, stream=True, timeout=(5, 60)
        ) as r:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue

                data = line[len("data:"):].strip()

                if data == "[DONE]":
                    break

                try:
                    evt = json.loads(data)
                    print("EVENT TYPE:", evt.get("type"))

                    if evt.get("type") == "content_block_delta":
                        piece = evt.get("delta", {}).get("text", "")
                        if piece:
                            assistant_full.append(piece)

                except Exception as e:
                    print("❌ JSON parse error:", e)
                    print("RAW:", data)

                yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

        final_text = "".join(assistant_full).replace("\n", " ")

        print("API final_text:", final_text)

        user_text = _last_user_text_from_messages(built)
        final_text = "".join(assistant_full).replace("\n", " ")
        if not skip_profile_ops:
            if user_text or final_text:
                fut = bg.submit(
                    _persist_interaction_async, device_id, user_text, final_text
                )
                fut.add_done_callback(
                    lambda _: bg.submit(prune_conversation, device_id)
                )
            if user_text:
                bg.submit(_maybe_extract_profile_async, device_id, user_text, profile)

    return app.response_class(generate(), mimetype="text/event-stream")


@app.post("/generate-project-title")
def generate_project_title():
    if not API_KEY:
        return jsonify({"error": "API_KEY not configured"}), 500

    data = request.get_json(force=True) or {}
    language = (data.get("language") or "HE").upper()
    user_message = (data.get("user_message") or "").strip()
    assistant_response = (data.get("assistant_response") or "").strip()
    device_id = (data.get("device_id") or "URBANX").strip()

    if not user_message or not assistant_response:
        return jsonify({"error": "missing user_message or assistant_response"}), 400

    PROMPT = (
        "צור שם קצר, מקצועי וברור לפרויקט.\n"
        "החזר רק כותרת קצרה של 2–5 מילים.\n"
        "בלי סימני פיסוק, בלי מרכאות, בלי הסברים."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"הודעת משתמש:\n{user_message}\n\nתשובת עוזר:\n{assistant_response}",
                }
            ],
        }
    ]

    headers = anthropic_headers()
    payload = {
        "model": app.config["ANTHROPIC_MODEL"],
        "system": [{"type": "text", "text": PROMPT}],
        "messages": messages,
        "max_tokens": 40,
        "temperature": 0.2,
        "stream": False,
    }

    try:
        r = requests.post(
            CLAUDE_API_URL,
            headers=headers,
            json=payload,
            timeout=(4, 30),
        )
    except requests.RequestException as e:
        return jsonify({"error": "LLM upstream error", "detail": str(e)}), 502

    if r.status_code // 100 != 2:
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:500]}
        return (
            jsonify({"error": "LLM non-2xx", "status": r.status_code, "body": body}),
            502,
        )

    try:
        resp_json = r.json()
        title = _extract_text_blocks(resp_json.get("content")).strip()
    except Exception as e:
        return jsonify({"error": "invalid LLM response", "detail": str(e)}), 502

    title = re.sub(r'["“”\'`]+', "", title)
    title = re.sub(r"[.,;:!?]+", "", title)
    title = re.sub(r"\s+", " ", title).strip()

    if not title:
        return jsonify({"error": "empty title"}), 502

    return jsonify({"title": title})


## Handle with images
@app.route("/upload", methods=["POST"])
def upload():
    response = {"status": -1}
    try:
        file = request.files["file"].read()
        file_id = str(uuid.uuid4())
        print(f"File ID: {file_id}")

        path = os.path.join(UPLOADS_DIR, file_id)
        with open(path, "wb") as saved_file:
            saved_file.write(file)

        print("File saved at:", os.path.abspath(path))
        response.update({"status": 0, "id": file_id})
    except Exception as error:
        print(f"Error while uploading file: {error}")

    return jsonify(response)


@app.route("/ocr", methods=["POST"])
def ocr():
    print("OCR")
    files = request.files.getlist("files")
    print(f"Files: {files}")

    try:
        text = "המשתמש שלח מספר תמונות.\n"
        for x in range(len(files)):
            text += f"תמונה מספר {x+1}: {image_ocr(files[x])}\n"
        print(f"text: {text}")
    except Exception as error:
        print(f"OCR failed: {error}")
        text = ""

    return text


## Handle with files
def get_file_from_id(file_id: str):
    path = os.path.join(UPLOADS_DIR, str(file_id))
    with open(path, "rb") as file:
        encoded_file = base64.b64encode(file.read())
    return encoded_file


def image_ocr(file):
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    print("📸 Running OCR...")
    image = Image.open(file).convert("RGB")
    text = pytesseract.image_to_string(image, lang="heb")
    print(f"📝 OCR TEXT: {text[:100]}...")
    return text


def pdf_to_text(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    extracted_text = ""

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        extracted_text += page.get_text()

    return extracted_text


def prepare_message_object(message_object):
    last_non_text_index = None

    for i in range(len(message_object)):
        content = message_object[i].get("content")
        if isinstance(content, list) and len(content) > 0:
            first_content = content[0]
            if first_content.get("type") != "text":
                last_non_text_index = i

    for i in range(len(message_object)):
        content = message_object[i].get("content")
        if isinstance(content, list) and len(content) > 0:
            first_content = content[0]

            if first_content.get("type") != "text":
                if i == last_non_text_index:
                    file_id = first_content.get("source", {}).get("data")
                    media_type = first_content.get("source", {}).get("media_type")

                    if file_id:
                        file_data = get_file_from_id(file_id)

                        if media_type == "application/pdf":
                            pdf_stream = BytesIO(base64.b64decode(file_data))
                            extracted_text = pdf_to_text(pdf_stream)

                            first_content["type"] = "text"
                            first_content["text"] = extracted_text
                            first_content.pop("source", None)

                        else:
                            first_content["source"]["data"] = file_data.decode("utf-8")

                else:
                    first_content.pop("source", None)
                    first_content["type"] = "text"
                    first_content["text"] = "**קובץ שנמחק**"

    return message_object


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

        out = [
            {
                "id": r.id,
                "device_id": r.device_id,
                "role": r.role,
                "text": r.text,
                "ts": r.ts.replace(microsecond=0).isoformat() if r.ts else None,
            }
            for r in rows
        ]
        return jsonify(out)
    finally:
        db.close()


# Debug load config
@app.get("/debug/config")
def debug_config():
    import os

    return jsonify(
        {
            "root_config_exists": os.path.exists(
                os.path.join(app.root_path, "config.py")
            ),
            "instance_config_exists": os.path.exists(
                os.path.join(app.instance_path, "config.py")
            ),
            "has_api_key": bool(API_KEY),
            "instance_path": app.instance_path,
            "root_path": app.root_path,
            "model": app.config.get("ANTHROPIC_MODEL"),
            "db_url": app.config.get("DB_URL"),
        }
    )


# Clean device id
def profile_path(device_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", device_id or "dev")
    return os.path.join(PROFILES_DIR, f"{safe}.json")


# Load json profile
def load_profile(device_id: str) -> dict:
    path = profile_path(device_id)
    if not os.path.exists(path):
        return {
            "name": None,
            "age": None,
            "gender": None,
            "likes": [],
            "dislikes": [],
            "parent_name": None,
            "pronouns": None,
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

    merged["core_collected"] = all(
        bool(merged.get(k)) for k in ("name", "age", "gender")
    )

    path = profile_path(device_id)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix="profile_", suffix=".json", dir=PROFILES_DIR
    )
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
    name = p.get("name")
    age = p.get("age")
    gender = p.get("gender")
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
        total = db.query(ConvoChunk).filter(ConvoChunk.device_id == device_id).count()

        if total <= MAX_HISTORY:
            return

        keep_rows = (
            db.query(ConvoChunk.id)
            .filter(ConvoChunk.device_id == device_id)
            .order_by(ConvoChunk.id.desc())
            .limit(MAX_HISTORY)
            .all()
        )

        keep_ids = [r.id for r in keep_rows]

        # מחק כל מה שלא בתוך ה־MAX_HISTORY האחרונות
        db.query(ConvoChunk).filter(
            ConvoChunk.device_id == device_id,
            ~ConvoChunk.id.in_(keep_ids),
        ).delete(synchronize_session=False)

        db.commit()
        print(f"Pruned conversation. Kept last {MAX_HISTORY} messages.")
    finally:
        db.close()


# Create user profile by device id
@app.post("/profile")
def upsert_profile():
    data = request.get_json(force=True) or {}
    device_id = (data.get("device_id") or "dev").strip()
    profile_patch = (
        data.get("profile")
        if isinstance(data.get("profile"), dict)
        else {k: v for k, v in data.items() if k in _ALLOWED_PROFILE_KEYS}
    )
    if not profile_patch:
        return jsonify({"error": "no profile fields provided"}), 400
    merged = save_profile(device_id, profile_patch)
    return jsonify({"ok": True, "profile": merged})


# Extracting profile from user - in background
def _maybe_extract_profile_async(device_id: str, user_text: str, current_profile: dict):
    miss = [k for k in ("name", "age", "gender") if not current_profile.get(k)]
    should_try = bool(miss) or any(
        not current_profile.get(k) for k in ("likes", "dislikes")
    )
    if not should_try or not user_text:
        return

    headers = anthropic_headers()

    extractor_system = (
        "אתה מחלץ פרטי פרופיל מילד מטקסט חופשי. החזר JSON חוקי בלבד, ללא טקסט נוסף. "
        'השדות המותרים: name (string), age (int), gender ("בן"/"בת"/null), '
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
        "stream": False,
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


# Get user history to summary
def get_recent_conversation(device_id: str) -> str:
    db = SessionLocal()
    try:
        rows = (
            db.query(ConvoChunk)
            .filter(ConvoChunk.device_id == device_id)
            .order_by(ConvoChunk.id.desc())
            .limit(MAX_HISTORY)
            .all()
        )
    finally:
        db.close()

    if not rows:
        return ""

    rows = list(reversed(rows))

    lines = []
    for r in rows:
        role = "משתמש" if r.role == "user" else "פנדי"
        text = (r.text or "").strip()
        if text:
            lines.append(f"{role}: {text}")

    return "\n".join(lines)


# Summary qurey
@app.post("/generate_parents_summary")
def generate_parents_summary():
    if not API_KEY:
        return jsonify({"error": "API KEY not configured"}), 500

    data = request.get_json(force=True) or {}
    device_id = (data.get("device_id") or "").strip()
    language = (data.get("language") or "HE").upper()

    if not device_id:
        return jsonify(
            {"status": "no_data", "message": "משתמש חדש. אין עדיין נתונים ליצירת דו״ח."}
        )

    conversation_text = get_recent_conversation(device_id)

    if not conversation_text:
        return jsonify(
            {"status": "no_data", "message": "אין עדיין מספיק שיחות ליצירת דו״ח רגשי."}
        )

    headers = anthropic_headers()
    profile = load_profile(device_id)

    profile_text = (
        f"פרופיל הילד:\n"
        f"שם: {profile.get('name') or 'לא ידוע'}\n"
        f"גיל: {profile.get('age') or 'לא ידוע'}\n"
        f"מגדר: {profile.get('gender') or 'לא ידוע'}\n"
        f"אוהב: {', '.join(profile.get('likes') or []) or 'לא ידוע'}\n"
        f"לא אוהב: {', '.join(profile.get('dislikes') or []) or 'לא ידוע'}\n"
    )

    combined_input = profile_text + "\n\nהיסטוריית שיחה:\n" + conversation_text[:12000]

    lang_map = {"HE": "עברית", "EN": "English", "AR": "العربية", "RO": "Română"}

    lang_name = lang_map.get(language, "עברית")

    summary_system = (
        SUMMARY_P_HE.strip() + f"\n\nהשפה המבוקשת לסיכום היא: {lang_name}.\n"
        "כתוב את כל הסיכום בשפה זו בלבד."
    )

    payload = {
        "model": app.config["ANTHROPIC_MODEL"],
        "system": [{"type": "text", "text": summary_system}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": combined_input}],
            }
        ],
        "max_tokens": 800,
        "temperature": 0.2,
        "stream": True,
    }

    def generate():
        assistant_full = []

        with requests.post(
            CLAUDE_API_URL, headers=headers, json=payload, stream=True, timeout=(5, 60)
        ) as r:

            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue

                chunk = line[len("data:") :].strip()

                try:
                    evt = json.loads(chunk)

                    if evt.get("type") == "content_block_delta":
                        piece = evt.get("delta", {}).get("text", "")
                        if piece:
                            assistant_full.append(piece)

                except Exception:
                    pass

                yield f"data: {chunk}\n\n"

        final_text = "".join(assistant_full).replace("\n", " ")
        print("FINAL SUMMARY:\n", final_text)

        yield "data: [DONE]\n\n"

    return app.response_class(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(app.config["PORT"]))
