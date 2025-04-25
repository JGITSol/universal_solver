import os
import requests
import time
from dotenv import load_dotenv
from threading import Lock

# Load .env at import time
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- API Rate Limiter ---
class RateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.last_reset = time.time()
        self.calls = 0
        self.lock = Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            if now - self.last_reset >= 60:
                self.calls = 0
                self.last_reset = now
            if self.calls >= self.calls_per_minute:
                sleep_time = 60 - (now - self.last_reset)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.calls = 0
                self.last_reset = time.time()
            self.calls += 1

# Instantiate per-provider limiters (adjust as needed)
openrouter_limiter = RateLimiter(60)  # 60 RPM default
# Google Gemini: 1000 RPM (paid), 10 RPM (free)
gemini_limiter = RateLimiter(10)

# --- OpenRouter API Client ---
def openrouter_chat(model, messages, temperature=0.1, max_tokens=512):
    openrouter_limiter.acquire()
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Assumes OpenAI-compatible response
    return data['choices'][0]['message']['content']

# --- Google Gemini API Client ---
def gemini_chat(model, messages, temperature=0.1, max_tokens=512):
    gemini_limiter.acquire()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    # Gemini expects a different format
    content = "\n".join([m['content'] for m in messages if m['role'] == 'user'])
    payload = {
        "contents": [{"parts": [{"text": content}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Extract text from Gemini response
    return data['candidates'][0]['content']['parts'][0]['text']
