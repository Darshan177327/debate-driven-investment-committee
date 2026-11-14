
from agents.gemini_client import generate_response

print("Testing shared Gemini connection...")

answer = generate_response(
    "Explain the role of a Bull Agent in an investment committee in two sentences."
)

print("\n=== BULL AGENT TEST ===")
print(answer)
