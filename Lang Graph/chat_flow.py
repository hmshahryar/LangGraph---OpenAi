from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import os
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------
# class                    
# ------------------------------------------------
class ChatState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]

# ------------------------------------------------
# llm                    
# ------------------------------------------------
gemini_api_key = os.getenv("GEMINI")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=gemini_api_key)

# ------------------------------------------------
# summarization helper
# ------------------------------------------------
def summarize_messages(messages: list[BaseMessage]) -> AIMessage:
    """Summarize a list of messages into a compact note."""
    summary_prompt = [
        HumanMessage(content=(
            "Summarize the following conversation messages. "
            "Keep only the important facts, decisions, or context needed for continuation. "
            "Make it concise but complete:\n\n" +
            "\n".join([f"{m.type.upper()}: {m.content}" for m in messages])
        ))
    ]
    summary = llm.invoke(summary_prompt)
    return AIMessage(content=f"[Summary of earlier conversation]: {summary.content}")

# ---------------------------------------------------
# chat node
# ---------------------------------------------------
def chat_node(state: ChatState):
    all_messages = state["message"]

    # --- memory compression logic ---
    if len(all_messages) > 8:  
        # keep first 5 intact
        preserved = all_messages[:5]

        # summarize 6..n-1
        to_summarize = all_messages[5:-1]
        summary_msg = summarize_messages(to_summarize) if to_summarize else None

        # keep last user query (latest message)
        latest = [all_messages[-1]]

        # rebuild compact state
        new_messages = preserved
        if summary_msg:
            new_messages.append(summary_msg)
        new_messages.extend(latest)

        state["message"] = new_messages

    # --- respond with LLM ---
    response = llm.invoke(state["message"])
    return {"message": [response]}

# ----------------------------------------------------
# graph with memory saver
# ------------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# attach memory saver
checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# ----------------------------------------------------
# conversation loop with smart memory + persistence
# ----------------------------------------------------
thread_id = "1"  # you can have multiple threads
config = {"configurable": {"thread_id": thread_id}}

state: ChatState = {"message": []}

while True:
    user_input = input("Enter your query: ")

    if user_input.lower() in ["exit", "1"]:
        print("Exiting chat.")
        break

    # add user message
    state["message"].append(HumanMessage(content=user_input))

    # run workflow with checkpoint memory
    state = workflow.invoke(state, config=config)

    # print AI response
    response_message = state["message"][-1].content
    print("AI:", response_message)

# ----------------------------------------------------
# retrieve stored state later
# ----------------------------------------------------
saved_state = workflow.get_state(config=config)
print("\n[Recovered State from MemorySaver]:", saved_state.values)
