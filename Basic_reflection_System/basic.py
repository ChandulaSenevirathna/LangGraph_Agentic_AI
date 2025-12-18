from typing import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState

from Chains import generation_chain, reflection_chain

load_dotenv()

# Node names
GENERATE = "generate"
REFLECT = "reflect"

# --------------------------------------------------
# Nodes
# --------------------------------------------------

def generate_node(state: MessagesState):
    """
    Generates a tweet based on conversation so far
    """
    result = generation_chain.invoke({"messages": state["messages"]})

    return {
        "messages": [
            AIMessage(content=result.content)
        ]
    }


def reflect_node(state: MessagesState):
    """
    Critiques / reflects on the generated tweet
    """
    result = reflection_chain.invoke({"messages": state["messages"]})

    return {
        "messages": [
            AIMessage(content=result.content)
        ]
    }

# --------------------------------------------------
# Graph definition
# --------------------------------------------------

graph = StateGraph(MessagesState)

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

# Stop condition
def should_continue(state: MessagesState):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

# Compile
app = graph.compile()

# --------------------------------------------------
# Run
# --------------------------------------------------

initial_state = {
    "messages": [
        HumanMessage(content="AI Agents taking over content creation")
    ]
}

final_state = app.invoke(initial_state)

# --------------------------------------------------
# Output
# --------------------------------------------------

print("\n=== Full Conversation ===\n")
for i, msg in enumerate(final_state["messages"], 1):
    print(f"{i}. {msg.type.upper()}: {msg.content}\n")