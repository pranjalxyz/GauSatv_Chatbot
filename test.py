import requests
import base64
import time

# 🔗 Your FastAPI endpoint
URL = "http://127.0.0.1:8000/voice-entry"

# 👤 Dummy user id
user_id = "test_user_1"

# 🎤 Audio file path
file_path = "Example.mp3"

# ⏱️ Start timer
start_time = time.time()

# 📤 Send request
with open(file_path, "rb") as f:
    files = {
        "file": ("Example.mp3", f, "audio/mpeg")
    }
    data = {
        "user_id": user_id,
        "cows": '["Ganga", "Kaveri", "Nandini","Kamdhenu"]'
    }

    response = requests.post(URL, files=files, data=data)

# ⏱️ End timer
end_time = time.time()
elapsed_time = end_time - start_time

# 📥 Print raw response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
print(f"⏱️ Response Time: {elapsed_time:.3f} seconds")

# 🔊 Save returned audio (optional)
res_json = response.json()

if "audio" in res_json:
    audio_data = base64.b64decode(res_json["audio"])

    with open("response.mp3", "wb") as out:
        out.write(audio_data)

    print("🔊 Audio saved as response.mp3")