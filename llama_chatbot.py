from fastapi import FastAPI, UploadFile, File, Form
import whisper
import concurrent.futures
import time
import json
import tempfile
import re
import os
import base64
from gtts import gTTS
from io import BytesIO
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import dotenv

dotenv.load_dotenv()

app = FastAPI()

# 🔊 Load Whisper
whisper_model = whisper.load_model("medium")

# 🤖 Configure Llama via Azure
client = ChatCompletionsClient(
    endpoint=os.getenv("AZURE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_API_KEY")),
)

# 🔢 Hindi number map
HINDI_NUM_MAP = {
    "शून्य": 0,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पांच": 5, "पाँच": 5,
    "छह": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14,
    "पंद्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18,
    "उन्नीस": 19, "बीस": 20
}

# 🧠 Temporary session store
session_store = {}

# ✅ YES / NO word lists
YES_WORDS = [
    "haan", "ha", "yes", "haanji", "ho", "theek", "sahi",
    "हाँ", "जी", "ठीक", "सही"
]

NO_WORDS = [
    "naa", "na", "nahi", "no", "mat", "galat",
    "नहीं", "ना", "मत", "गलत"
]

def normalize_text(text):
    text = text.lower().strip()
    text = text.replace("हां", "हाँ")
    return text

def contains_word(text, word):
    return re.search(rf'\b{word}\b', text) is not None

def classify_confirmation(text):
    text = text.lower().strip()
    for word in YES_WORDS:
        if contains_word(text, word):
            return "yes"
    for word in NO_WORDS:
        if contains_word(text, word):
            return "no"
    return "unknown"

def text_to_speech_base64(text):
    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang="hi")
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return base64.b64encode(mp3_fp.read()).decode("utf-8")

def safe_parse(t):
    try:
        match = re.search(r'\{.*\}', t, re.DOTALL)
        return json.loads(match.group()) if match else None
    except:
        return None

# 🦙 Llama call with retry + timeout
def call_llama(prompt: str) -> str:
    max_retries = int(os.getenv("MAX_RETRIES", 3))
    timeout = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 30))
    last_error = None

    for attempt in range(max_retries):
        try:
            def _call():
                response = client.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=os.getenv("MODEL_NAME"),
                    max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 1000)),
                    temperature=0.3,
                    top_p=1.0,
                )
                text_out = response.choices[0].message.content
                if not text_out or not text_out.strip():
                    raise ValueError("Llama returned empty response")
                return text_out.strip()

            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
                result = future.result(timeout=timeout)
            end = time.perf_counter()

            print(f"✅ Llama Output: {result}")
            print(f"⏱️ Llama Time: {end - start:.3f}s")
            return result

        except concurrent.futures.TimeoutError:
            last_error = TimeoutError(f"Llama timed out after {timeout}s")
        except Exception as e:
            last_error = e

        if attempt < max_retries - 1:
            wait = 2 ** attempt
            print(f"⚠️ Attempt {attempt + 1} failed: {last_error}. Retrying in {wait}s...")
            time.sleep(wait)

    raise last_error


# 🎤 MAIN ENTRY
@app.post("/voice-entry")
async def voice_entry(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    cows: str = Form(...)
):
    try:
        # Step 1: Save audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name

        # Step 2: Whisper transcription
        result = whisper_model.transcribe(temp_audio_path, language="hi")
        text = result["text"].strip()
        print("RAW:", text)
        os.remove(temp_audio_path)

        # Step 3: Cow list
        cow_list = json.loads(cows)
        cow_list_str = ", ".join(cow_list)

        # Step 4: Prompt
        prompt = f"""
You are a JSON API. You must ONLY return valid JSON. No explanation.

If you add any text outside JSON, the response is INVALID.

---

Available cows:
{cow_list_str}

---

Number in Hindi for reference:
{HINDI_NUM_MAP}

---

Rules:

1. Intent:
- milk_entry → milk quantity mentioned
- feed → feeding mentioned
- health → health condition mentioned

2. Cow:
- Must EXACTLY match from list
- If similar → map to closest
- Else → null

3. Quantity:
- Only for milk_entry
- Convert Hindi numbers (पांच → 5)
- Else → null

4. Health:
- One of: healthy, fever, upset_stomach, injury, other
- Else → null

---

Sentence:
"{text}"

---

Output format (STRICT JSON ONLY):

{{
  "intent": "milk_entry/feed/health",
  "cow_name": string or null,
  "quantity": number or null,
  "health_type": "healthy/fever/upset_stomach/injury/other/null"
}}

---

DO NOT explain, add text, add markdown, or add comments. ONLY return JSON.
"""

        # Step 5: Llama call
        llm_response = call_llama(prompt)

        # Step 6: Parse JSON
        structured = safe_parse(llm_response)

        if not structured:
            raise ValueError("Invalid JSON from Llama")

        # Step 7: Validation
        valid_intents = ["milk_entry", "feed", "health"]
        if structured.get("intent") not in valid_intents:
            raise ValueError("Invalid intent")

        if structured.get("cow_name") not in cow_list:
            structured["cow_name"] = None

        if structured["cow_name"] is None:
            return {
                "decision": "unknown",
                "audio": text_to_speech_base64("कृपया गाय का नाम सही से बोलिए")
            }

        if structured["intent"] == "milk_entry" and structured.get("quantity") is None:
            return {
                "decision": "unknown",
                "audio": text_to_speech_base64("कृपया दूध की मात्रा बताइए")
            }

        # Step 8: Store session
        session_store[user_id] = {"data": structured}

        # Step 9: Build confirmation message
        intent = structured["intent"]
        name = structured["cow_name"]
        qty = structured.get("quantity")
        ht = structured.get("health_type")

        if intent == "milk_entry":
            msg = f"क्या मैं {name} के लिए {qty} लीटर दूध सेव कर दूँ? सिर्फ हाँ या ना बोलिए।"
        elif intent == "feed":
            msg = f"क्या मैं {name} को चारा दिया गया मार्क कर दूँ? सिर्फ हाँ या ना बोलिए।"
        elif intent == "health":
            health_msgs = {
                "fever":          f"क्या मैं {name} को बुखार दर्ज कर दूँ?",
                "injury":         f"क्या मैं {name} की चोट दर्ज कर दूँ?",
                "upset_stomach":  f"क्या मैं {name} का पेट खराब दर्ज कर दूँ?",
                "healthy":        f"क्या मैं {name} को स्वस्थ मार्क कर दूँ?",
            }
            msg = health_msgs.get(ht, f"क्या मैं {name} की स्वास्थ्य जानकारी सेव कर दूँ?")
        else:
            raise ValueError("Unhandled intent")

        return {
            "decision": "confirm",
            "audio": text_to_speech_base64(msg)
        }

    except Exception as e:
        print("ERROR:", e)
        return {
            "decision": "unknown",
            "audio": text_to_speech_base64("मुझे समझ नहीं आया, कृपया फिर से बोलिए")
        }


# 🔁 CONFIRMATION
@app.post("/confirm-entry")
async def confirm_entry(user_id: str = Form(...), file: UploadFile = File(...)):

    session = session_store.get(user_id)
    if not session:
        return {"decision": "invalid_session"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name

    result = whisper_model.transcribe(temp_audio_path, language="hi")
    text = result["text"]
    os.remove(temp_audio_path)

    decision = classify_confirmation(text)

    if decision == "yes":
        final_data = session["data"]
        # 👉 YOUR SAVE FUNCTION HERE
        # save_entry(final_data)
        session_store.pop(user_id)
        return {
            "decision": "yes",
            "audio": text_to_speech_base64("एंट्री सेव हो गई")
        }

    elif decision == "no":
        session_store.pop(user_id)
        return {
            "decision": "no",
            "audio": text_to_speech_base64("ठीक है, दोबारा एंट्री साफ शब्दों में बोलिए")
        }

    else:
        return {
            "decision": "unknown",
            "audio": text_to_speech_base64("सिर्फ हाँ या ना बोलिए")
        }