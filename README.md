# AmbedkarGPT - RAG Q&A System

This is an assignment submission for the AI Intern role. It implements a RAG (Retrieval Augmented Generation) pipeline using LangChain, ChromaDB, and Ollama (Mistral 7B).

## Features
- **Local LLM:** Uses Mistral 7B via Ollama (Privacy-focused, runs offline).
- **Vector Store:** ChromaDB for storing document embeddings.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Modern Architecture:** Built using the latest LangChain LCEL (LangChain Expression Language).

## Prerequisites
1. **Install Ollama:** [Download here](https://ollama.com)
2. **Pull the Model:** ```bash
   ollama pull mistral