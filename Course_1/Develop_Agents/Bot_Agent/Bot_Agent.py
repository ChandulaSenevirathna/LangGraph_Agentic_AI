from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ----------------------------
# State definition
# ----------------------------
class AgentState(TypedDict):
    messages: List[HumanMessage]

# ----------------------------
# Gemini LLM
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.7
)

# ----------------------------
# Node logic
# ----------------------------
def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"state messages: {state['messages']}")
    print(f"\nAI: {response.content}")
    return state

# ----------------------------
# LangGraph
# ----------------------------
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

# ----------------------------
# Chat loop
# ----------------------------
user_input = input("Enter: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
