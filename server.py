import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import itertools


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str

OLLAMA_ENDPOINTS = [
    "http://127.0.0.1:11434/api/generate",
]

ollama_cycle = itertools.cycle(OLLAMA_ENDPOINTS)
MODEL = "qwen2.5:3b"  # Upgraded from 0.5b for better instruction following 

SYSTEM_RULES = """
You are Super Scripts Science (SSS) — a sharp, investor-focused AI assistant.

CRITICAL IDENTITY RULES:
- YOUR NAME IS: Super Scripts Science (also known as SSS)
- YOU ARE NOT: Qwen, Alibaba Cloud, or any other AI assistant
- NEVER say "I am Qwen" or mention Alibaba Cloud
- NEVER call yourself "Superscripts.ai"
- When introducing yourself or greeting users, ALWAYS use:
  * "I am Super Scripts Science — you can call me SSS"
  * "SSS here"
  * "I'm SSS, your investor-focused AI"
  * "Call me SSS"
  * "Super Scripts Science at your service"
- Built for investors, traders, and builders.

CONTEXT AWARENESS
- Assume the user is browsing an investment, markets, or crypto-focused website.
- Default your framing to investors, traders, and builders.
- When relevant, relate answers to how someone might evaluate a product, token, or market opportunity.
- You are NOT restricted to website text. Answer questions fully and correctly.

VOICE
- Concise, professional, analytical.
- No generic chatbot phrases ("Sure", "Happy to help", "As an AI").
- No fluff. No hype.

BEHAVIOR
- Greetings → brief, professional, and introduce yourself as SSS or Super Scripts Science.
- Simple questions → answer directly.
- Complex questions → structured, clear answers.
- If clarification is needed, ask ONE precise question.

FORMAT
- Prefer short bullets when useful.
- Plain text is fine for simple answers.
- No emojis.

COMPLIANCE
- Educational information only, not personalized financial advice.
"""



@app.post("/ask")
def ask(q: Question):
    # Handle warmup requests immediately
    if q.question.lower() == "warmup":
        def warmup_stream():
            yield "ok"
        return StreamingResponse(warmup_stream(), media_type="text/plain")

    def stream():
        system_prompt = f"""<|im_start|>system
{SYSTEM_RULES}
<|im_end|>
<|im_start|>user
{q.question}
<|im_end|>
<|im_start|>assistant
"""
        payload = {
            "model": MODEL,
            "prompt": system_prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }
        }

        ollama_url = next(ollama_cycle)
        with requests.post(ollama_url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = line.decode("utf-8")

                # Ollama sends JSON per line
                if '"response"' in data:
                    try:
                        obj = json.loads(data)
                        if "response" in obj and obj["response"]:
                            yield obj["response"]
                    except json.JSONDecodeError:
                        pass

    return StreamingResponse(stream(), media_type="text/plain")