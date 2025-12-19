from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables (for API key)
load_dotenv()

# -------------------------------
# PROMPTS
# -------------------------------

# Tweet generation prompt
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Tweet reflection prompt
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            " Always provide detailed recommendations, including requests for length, virality, style, etc."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# -------------------------------
# LLM (Groq) with verbose=True
# -------------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY"),
)

# -------------------------------
# CHAINS
# -------------------------------

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

# -------------------------------
# EXECUTE GENERATION
# -------------------------------

# user_input = "Write a small tweet about earth"

# # Invoke the generation chain
# generated = generation_chain.invoke({
#     "messages": [("human", user_input)]
# })

# # Optional: manually print formatted prompt (redundant with verbose=True)
# prompt_obj = generation_prompt.format_prompt(messages=[("human", user_input)])
# print("\n=== Formatted Prompt Sent to LLM ===")
# print(prompt_obj.to_string())

# Print generated tweet
# print("\n=== Generated Tweet ===")
# print(generated.content)
