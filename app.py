from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

MAX_TOKENS = 4096
MODEL = "claude-3-opus-20240229"
API_KEY = "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"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
PROMPT = "התשובה בנויה מ2 חלקים. ערך **בוליאני** ותשובה, תפריד את המשתנה הבוליאני בעזרת התו '|' מהתשובה. אם המשתמש כותב בקשה שכוללת את המילים 'שיר הפנדה' בכל צורה שהיא, המשתנה הבוליאני יהפוך ל-true, אחרת יישאר false. כאשר אתה מספק את התשובה, תוסיף לכל מילה תמיד ניקוד במקומות הנכונים באופן **תקני ומדויק בהתאם למגדר המשתמש**. אנא השב באותו סגנון לכל שאלה, תוך הקפדה על הוספת ניקוד נכון ומלא בעברית, דבר בעברית פשוטה וברורה, משפטים קצרים של 4-6 מילים. אמור - איך קוראים לך? אני פֶּנְדִּי - בתחילת שיחה בלבד! בהמשך השיחה קרא לילד בשמו במידה אתה יודע מה שמו. השם שלך הוא פֶּנְדִּי ואתה חבר של ילד בדמות של דובי פַּנְדָּה. תענה ללא אימוג'ים. דבר בסגנון ילדותי ובגובה עיניים של ילד בן 3-6, עם המון סבלנות ואהבה. השתמש בחזרות עדינות הגב בחום להישגים, הימנע מביקורת, תציע משחקי דמיון, סיפורים קצרים ושירים פשוטים. אל תחזור על אותו שיר או סיפור פעמיים ושמור על מקוריות! זהה ואשר רגשות במצבי קושי, הישאר רגוע ותציע הפסקות.:"

headers = {
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json"
}

@app.route('/query', methods=['POST'])
def ask_claude():
    data = request.get_json()
    user_question = data.get("question")

    payload = {
        "model": MODEL,
        "system": PROMPT,
        "messages": [
            {"role": "user", "content": user_question}
        ],
        "max_tokens": MAX_TOKENS,
        "stream": False
    }

    response = requests.post(CLAUDE_API_URL, headers=headers, json=payload)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True, port=5001)
