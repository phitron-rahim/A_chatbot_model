import streamlit as st

from chatbot import build_chain
from schemas import ChatbotResponse


st.set_page_config(page_title="LangChain Chatbot")


@st.cache_resource
def load_chain():
    return build_chain()


def display_response(response: ChatbotResponse):
    st.markdown(response.answer)

    with st.expander("View structured output"):
        st.json(response.model_dump())

    with st.expander("Summary"):
        st.write(response.summary)

        if response.follow_up_questions:
            st.write("Related questions:")
            for question in response.follow_up_questions:
                st.markdown(f"- {question}")


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.title("AI Chatbot")
    st.write("RunnableBranch + RunnableParallel + Pydantic")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("LangChain Chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):

        if message["role"] == "assistant":
            display_response(ChatbotResponse(**message["content"]))

        else:
            st.markdown(message["content"])


question = st.chat_input(
    "Ask a programming, math, or general question..."
)

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

                response = load_chain().invoke(
                    {"question": question}
                )

            display_response(response)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.model_dump(),
                }
            )

        except Exception as e:
            st.error(str(e))