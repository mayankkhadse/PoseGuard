import os
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not set. Please set the environment variable before running the app.")

client = Groq(api_key=api_key)
chat_history = []

SYSTEM_PROMPT = """You are FitBot, an expert AI fitness coach built into PoseGuard.
Answer questions about gym exercises, posture, sets, reps, warm-ups, nutrition,
and workout plans. Keep answers short, practical, and motivating. Max 2-3 sentences."""

def ask_fitbot(question):
    chat_history.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + chat_history[-6:],
            max_tokens=150,
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Connection error: {str(e)[:40]}"

    chat_history.append({"role": "assistant", "content": reply})
    return reply

def reset_chat():
    chat_history.clear()