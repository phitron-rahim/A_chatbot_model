
from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
)

from prompts import (
    GENERAL_PROMPT,
    MATH_PROMPT,
    PROGRAMMING_PROMPT,
    SUMMARY_PROMPT,
)
from schemas import AnswerPayload, ChatbotResponse, SummaryPayload


# --------------------------------------------------
# Environment configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

def get_chat_model() -> ChatGoogleGenerativeAI:
    """Load configuration and initialize the Gemini model."""

    if ENV_FILE.is_file():
        load_dotenv(dotenv_path=ENV_FILE, override=True)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        try:
            api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            api_key = ""

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    ).strip()

    temperature = float(
        os.getenv("GEMINI_TEMPERATURE", "0.2")
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing or empty."
        )

    if not model_name:
        raise RuntimeError(
            "GEMINI_MODEL is missing or empty."
        )

    print("Environment file found:", ENV_FILE.is_file())
    print("API key loaded:", bool(api_key))
    print("Gemini model:", model_name)

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
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
# Combine structured outputs
# --------------------------------------------------

def combine_results(data: dict) -> ChatbotResponse:
    """Combine the answer and summary into one response."""

    answer = data["answer"]
    summary = data["summary"]

    return ChatbotResponse(
        answer=answer.answer,
        summary=summary.summary,
        confidence=answer.confidence,
        category=answer.category,
        keywords=answer.keywords,
        follow_up_questions=summary.follow_up_questions,
    )


# --------------------------------------------------
# Build LangChain pipeline
# --------------------------------------------------

def build_chain():
    """Build and return the complete chatbot chain."""

    model = get_chat_model()

    answer_llm = model.with_structured_output(
        AnswerPayload,
        method="function_calling",
    )

    summary_llm = model.with_structured_output(
        SummaryPayload,
        method="function_calling",
    )

    # Route each question to the appropriate prompt.
    answer_chain = RunnableBranch(
        (
            is_programming,
            PROGRAMMING_PROMPT | answer_llm,
        ),
        (
            is_math,
            MATH_PROMPT | answer_llm,
        ),
        GENERAL_PROMPT | answer_llm,
    )

    # Generate the answer and summary concurrently.
    parallel_chain = RunnableParallel(
        answer=answer_chain,
        summary=SUMMARY_PROMPT | summary_llm,
    )

    # Return a ChatbotResponse object.
    return parallel_chain | RunnableLambda(combine_results)


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