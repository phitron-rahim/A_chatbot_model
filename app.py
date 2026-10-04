import os
from pathlib import Path

import streamlit as st

# Keep this as the first Streamlit command.
st.set_page_config(
    page_title="LangChain Chatbot",
    page_icon="🤖",
    layout="centered",
)

from dotenv import load_dotenv
from chatbot import build_chain
from schemas import ChatbotResponse

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=True)

print("APP FILE:", Path(__file__).resolve())
print("ENV FILE EXISTS:", ENV_FILE.exists())
print("GROQ API KEY FOUND:", bool(os.getenv("GROQ_API_KEY", "").strip()))
print("GROQ MODEL:", os.getenv("GROQ_MODEL"))

# --------------------------------------------------
# Initialize chatbot chain
# --------------------------------------------------

@st.cache_resource
def load_chain():
    """Create and cache the chatbot chain."""
    return build_chain()


# --------------------------------------------------
# Display chatbot response
# --------------------------------------------------

def display_response(response: ChatbotResponse):
    """Display the answer and structured information."""

    st.markdown(response.answer)

    with st.expander("View structured output"):
        st.json(response.model_dump())

    with st.expander("Summary"):
        st.write(response.summary)

        if response.follow_up_questions:
            st.write("Related questions:")

            for follow_up in response.follow_up_questions:
                st.markdown(f"- {follow_up}")


# --------------------------------------------------
# Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("AI Chatbot")
    st.caption("RunnableBranch + RunnableParallel + Pydantic")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Main interface
# --------------------------------------------------

st.title("LangChain Chatbot")

st.caption(
    "Ask programming, mathematics, or general questions."
)


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            response = ChatbotResponse.model_validate(
                message["content"]
            )
            display_response(response)


# --------------------------------------------------
# Accept a new question
# --------------------------------------------------

question = st.chat_input(
    "Ask a programming, math, or general question..."
)


# --------------------------------------------------
# Generate response
# --------------------------------------------------

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                chain = load_chain()
                response = chain.invoke(
                    {"question": question}
                )

            display_response(response)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.model_dump(),
                }
            )

        except Exception as exc:
            st.error(
                f"Failed to generate a response: "
                f"{type(exc).__name__}: {exc}"
            )

            st.caption(
                "Check the VS Code terminal for diagnostic output."
            )
