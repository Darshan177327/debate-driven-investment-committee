
import os
from dotenv import load_dotenv
from google import genai

# Load API key from the .env file
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Check your .env file."
    )

# Create Gemini client
client = genai.Client(api_key=API_KEY)

# Model that worked in our API test
MODEL_NAME = "gemini-3.5-flash-lite"


def generate_response(prompt):
    """
    Send a prompt to Gemini and return its response.
    All agents will use this function.
    """

    response = client.interactions.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text
