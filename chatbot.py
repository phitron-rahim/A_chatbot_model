from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_groq import ChatGroq

from prompts import (
    GENERAL_PROMPT,
    MATH_PROMPT,
    PROGRAMMING_PROMPT,
    SUMMARY_PROMPT,
)
from schemas import AnswerPayload, ChatbotResponse, SummaryPayload


PROGRAMMING_PATTERN = re.compile(
    r"\b(python|java|javascript|typescript|c\+\+|code|debug|function|class|"
    r"algorithm|api|database|sql|html|css|react|django|flask|streamlit)\b",
    re.IGNORECASE,
)

MATH_PATTERN = re.compile(
    r"\b(math|algebra|calculus|geometry|equation|integral|derivative|matrix|"
    r"probability|statistics|solve|factor|simplify)\b|[0-9]+\s*[-+*/^]\s*[0-9]+",
    re.IGNORECASE,
)


def get_chat_model() -> ChatGroq:
    """Loads the Groq model from the .env file."""

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Please add it to your .env file."
        )

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")),
        api_key=api_key,
    )


def is_programming(data: dict[str, Any]) -> bool:
    return bool(PROGRAMMING_PATTERN.search(data["question"]))


def is_math(data: dict[str, Any]) -> bool:
    return bool(MATH_PATTERN.search(data["question"]))


def combine_results(data: dict[str, Any]) -> ChatbotResponse:
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


def build_chain():
    """Creates the chatbot pipeline."""

    model = get_chat_model()

    answer_llm = model.with_structured_output(AnswerPayload)
    summary_llm = model.with_structured_output(SummaryPayload)

    branch = RunnableBranch(
        (is_programming, PROGRAMMING_PROMPT | answer_llm),
        (is_math, MATH_PROMPT | answer_llm),
        GENERAL_PROMPT | answer_llm,
    )

    parallel = RunnableParallel(
        answer=branch,
        summary=SUMMARY_PROMPT | summary_llm,
    )

    return parallel | RunnableLambda(combine_results)


def ask_chatbot(question: str) -> ChatbotResponse:
    chatbot = build_chain()
    return chatbot.invoke({"question": question})