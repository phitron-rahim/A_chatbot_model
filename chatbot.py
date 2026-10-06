from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableBranch

from prompts import (
    GENERAL_PROMPT,
    MATH_PROMPT,
    PROGRAMMING_PROMPT,
)
from schemas import ChatbotResponse


# --------------------------------------------------
# Environment configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def get_chat_model() -> ChatGroq:
    """Load configuration and initialize the Groq model."""

    # Local development
    if ENV_FILE.is_file():
        load_dotenv(dotenv_path=ENV_FILE, override=True)

    # Local .env first
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    # Streamlit Cloud Secrets fallback
    if not api_key:
        try:
            api_key = str(st.secrets["GROQ_API_KEY"]).strip()
        except Exception:
            api_key = ""

    model_name = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
    ).strip()

    temperature = float(
        os.getenv("GROQ_TEMPERATURE", "0.2")
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing or empty."
        )

    if not model_name:
        raise RuntimeError(
            "GROQ_MODEL is missing or empty."
        )

    print("Environment file found:", ENV_FILE.is_file())
    print("API key loaded:", bool(api_key))
    print("Groq model:", model_name)

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key,
    )


# --------------------------------------------------
# Question classification
# --------------------------------------------------

PROGRAMMING_PATTERN = re.compile(
    r"\b(python|java|javascript|typescript|c\+\+|code|debug|"
    r"function|class|algorithm|api|database|sql|html|css|"
    r"react|django|flask|streamlit)\b",
    re.IGNORECASE,
)


MATH_PATTERN = re.compile(
    r"\b(math|algebra|calculus|geometry|equation|integral|"
    r"derivative|matrix|probability|statistics|solve|factor|"
    r"simplify|addition|subtraction|multiplication|division)\b"
    r"|[0-9]+\s*[-+*/^]\s*[0-9]+",
    re.IGNORECASE,
)


def is_programming(data: dict) -> bool:
    """Return True for programming-related questions."""

    question = str(data.get("question", ""))
    return bool(PROGRAMMING_PATTERN.search(question))


def is_math(data: dict) -> bool:
    """Return True for mathematics-related questions."""

    question = str(data.get("question", ""))
    return bool(MATH_PATTERN.search(question))


# --------------------------------------------------
# Build LangChain pipeline
# --------------------------------------------------

def build_chain():
    """Build the chatbot chain using ONE Groq call per question."""

    model = get_chat_model()

    structured_llm = model.with_structured_output(
        ChatbotResponse,
        method="function_calling",
    )

    answer_chain = RunnableBranch(
        (
            is_programming,
            PROGRAMMING_PROMPT | structured_llm,
        ),
        (
            is_math,
            MATH_PROMPT | structured_llm,
        ),
        GENERAL_PROMPT | structured_llm,
    )

    return answer_chain


# --------------------------------------------------
# Public chatbot function
# --------------------------------------------------

def ask_chatbot(question: str) -> ChatbotResponse:
    """Answer a non-empty question using the chatbot."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    chain = build_chain()

    return chain.invoke(
        {"question": question.strip()}
    )