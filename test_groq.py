import os
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
print("Key found:", bool(api_key))

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "What is exercise?"}],
    max_tokens=50,
)

print(response.choices[0].message.content)