from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime


# Load .env
load_dotenv()

# Initialize LLM with a supported model
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")


# Initialize search tool
search_tool = TavilySearchResults(search_depth="basic")

# Define a system tool
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format."""
    current_time = datetime.datetime.now()
    return current_time.strftime(format)

# Combine tools
tools = [search_tool, get_system_time]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Example query
response = agent.invoke(
    "When was SpaceX's last launch and how many days ago was that from this instant?"
)

print("\nAgent response:\n", response)
