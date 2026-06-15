import os

try:
    from groq import Groq
    HAS_GROQ = True
except Exception:
    Groq = None
    HAS_GROQ = False

api_key = os.getenv("GROQ_API_KEY")
if not HAS_GROQ or not api_key:
    client = None
    if not HAS_GROQ:
        print("Warning: groq package not installed. FitBot will be disabled.")
    else:
        print("Warning: GROQ_API_KEY not set. FitBot will be disabled.")
else:
    client = Groq(api_key=api_key)

chat_history = []

SYSTEM_PROMPT = """You are FitBot, an expert AI fitness coach built into PoseGuard.
Answer questions about gym exercises, posture, sets, reps, warm-ups, nutrition,
and workout plans. Keep answers short, practical, and motivating. Max 2-3 sentences."""

def ask_fitbot(question):
    if client is None:
        return "FitBot is unavailable in this deployment."

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