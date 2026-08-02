from langchain_core.prompts import PromptTemplate


PROGRAMMING_PROMPT = PromptTemplate.from_template("""
A user has asked a programming question.

Explain the answer in a simple way.
If a code example helps, include a short one.
Avoid making the explanation unnecessarily long.

Question:
{question}
""")


MATH_PROMPT = PromptTemplate.from_template("""
You are helping a student with a math problem.

Solve the problem completely.
Show the calculation if needed.
Always include the final answer.

Question:
{question}
""")


GENERAL_PROMPT = PromptTemplate.from_template("""
Answer the following question in a clear and easy-to-understand way.

Question:
{question}
""")

SUMMARY_PROMPT = PromptTemplate.from_template("""
Read the user's question and return:

1. A short summary (maximum 2 sentences)
2. Exactly 3 follow-up questions

Keep the response short and simple.

Question:
{question}
""")