
import os
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file.")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Send a test question using the new Interactions API
response = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input="Explain what an investment committee does in two sentences."
)

print("\n=== GEMINI RESPONSE ===")
print(response.output_text)
