# Groq + Qwen RAG Chatbot

A document question-answering chatbot built with Streamlit,
LangChain, FAISS, Sentence Transformers, and the Groq API.

## Features

- Upload PDF and TXT documents
- Sample knowledge base included
- Document text chunking
- Text embeddings
- FAISS similarity search
- Qwen answer generation through Groq
- Display retrieved document passages

## Technologies

- Python
- Streamlit
- Groq API
- Qwen
- LangChain
- FAISS
- Sentence Transformers

## Installation

Install the dependencies:

pip install -r requirements.txt

## Configure API Key

Create the file:

.streamlit/secrets.toml

Add your Groq API key:

GROQ_API_KEY = "your-groq-api-key"

Never upload API keys or secrets to GitHub.

## Run Locally

streamlit run app.py

