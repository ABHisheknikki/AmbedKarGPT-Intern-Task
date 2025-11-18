# AmbedkarGPT - RAG Q&A System

This is an assignment submission for the AI Intern role. It implements a RAG (Retrieval Augmented Generation) pipeline using LangChain, ChromaDB, and Ollama (Mistral 7B).
## Phase 1:
## Features
- **Local LLM:** Uses Mistral 7B via Ollama (Privacy-focused, runs offline).
- **Vector Store:** ChromaDB for storing document embeddings.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Modern Architecture:** Built using the latest LangChain LCEL (LangChain Expression Language).

## Phase 2: Evaluation & Benchmarking

We implemented a comprehensive evaluation framework to test 3 chunking strategies (Small, Medium, Large).

### Files Added
* `evaluation.py`: Automated testing script.
* `setup_data.py`: Generates the corpus and test dataset.
* `test_results.json`: Raw output of the evaluation.
* `results_analysis.md`: Detailed breakdown of findings.

### How to Run Evaluation
1. **Generate Data:**
   ```bash
   python setup_data.py

## Prerequisites
1. **Install Ollama:** [Download here](https://ollama.com)
2. **Pull the Model:** ```bash

   ollama pull mistral

