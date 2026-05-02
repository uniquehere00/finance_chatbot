import os
import sys
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env file")
    print("Make sure your .env file has: GROQ_API_KEY=your_key_here")
    sys.exit(1)

print("API key found, testing connection...")

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Say exactly: Groq connection successful!"
        }
    ],
    max_tokens=50
)

print(response.choices[0].message.content)