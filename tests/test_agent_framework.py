
import asyncio
from dotenv import load_dotenv

from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient

load_dotenv()

TOOL_CALLED = False


# Python function registered as an Agent Framework tool
def get_demo_company():
    """Return the company selected for the investment committee demonstration."""
    global TOOL_CALLED
    TOOL_CALLED = True

    print("\n[TOOL EXECUTED] get_demo_company()")

    return "Jio Financial Services Limited (JIOFIN)"


async def main():
    global TOOL_CALLED

    print("Starting Microsoft Agent Framework test...")

    async with GeminiChatClient(
        model="gemini-3.5-flash-lite"
    ) as client:

        agent = Agent(
            client=client,
            name="InvestmentResearchTestAgent",
            instructions=(
                "You are an investment research assistant. "
                "You must call the get_demo_company tool before answering. "
                "Use the exact company name returned by the tool."
            ),
            tools=[get_demo_company],
        )

        response = await agent.run(
            "Which company is being used in our investment committee "
            "demonstration? Call your tool to find out."
        )

        print("\n===== AGENT RESPONSE =====")
        print(response.text)

    print("\n===== VERIFICATION =====")

    if TOOL_CALLED:
        print("PASS: Microsoft Agent Framework executed the Python tool.")
    else:
        raise RuntimeError(
            "FAIL: The agent did not execute the registered Python tool."
        )


if __name__ == "__main__":
    asyncio.run(main())
