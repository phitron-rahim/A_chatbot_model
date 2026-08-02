# LangChain Chatbot using RunnableBranch and RunnableParallel

## About the Project

This project was created as part of a LangChain assignment. The goal was to build a simple AI chatbot using LangChain's LCEL features.

The chatbot can answer programming, mathematics, and general questions. Depending on the user's question, it chooses the appropriate prompt using -> RunnableBranch . It also generates multiple outputs at the same time, such as the main answer and a short summary, using   ->  RunnableParallel.

All responses are validated with Pydantic before they are shown in the Streamlit interface.



## Project Workflow


User Question
      ↓
RunnableBranch
      ↓
Choose the right prompt
      ↓
RunnableParallel
      ↓
Answer + Summary
      ↓
Pydantic Validation
      ↓
Streamlit UI




## Features

- Simple Streamlit chat interface
- PromptTemplate for creating prompts
- RunnableBranch for topic-based routing
- RunnableParallel for generating multiple outputs
- Pydantic structured output
- API key stored securely using  ->  .env



## Project Structure

```
project/

app.py
chatbot.py
prompts.py
schemas.pyrequirements.txt
.env.example
README.md
assets/


## Installation

Clone the repository and install the required packages.

bash ->
pip install -r requirements.txt


Create a .env file and add your Groq API key.

env
GROQ_API_KEY=your_api_key


Run the application:

bash -> 
streamlit run app.py




## Technologies Used

- Python
- LangChain
- ChatGroq
- Streamlit
- Pydantic


## Assignment Requirements

This project includes:

- PromptTemplate
- RunnableBranch
- RunnableParallel
- Pydantic Structured Output
- Streamlit User Interface



## Note

Do not upload your `.env` file or API key. Only submit the GitHub repository link as required.