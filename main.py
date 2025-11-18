"""
Author: Abhishek Pratap Singh
Short Intro:
I work on building practical AI systems, including RAG pipelines, LangChain, LangGraph,
and local LLM setups using Ollama and ChromaDB.
"""

#====================  IMPORTS  ====================

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


#====================  HELPERS  ====================

def format_docs(docs):
    """
    Convert retrieved chunks into a single combined string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


#====================  MAIN PIPELINE  ====================

def main():
    """
    Minimal LCEL-based RAG pipeline using:
    - speech.txt
    - ChromaDB
    - HuggingFace embeddings
    - Ollama (Mistral)
    - LCEL runnable chain
    """
    print("==========================================")
    print("===== AmbedkarGPT (LCEL Pipe Version) ====")
    print("==========================================")

    print("1. Loading speech.txt...")
    if not os.path.exists("speech.txt"):
        print("Error: speech.txt not found!")
        return

    loader = TextLoader("speech.txt", encoding="utf-8")
    documents = loader.load()

    print("2. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print("3. Creating Vector Store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists("./chroma_db"):
        db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

    retriever = db.as_retriever(search_kwargs={"k": 2})
    llm = Ollama(model="mistral")

    template = """Answer the question based ONLY on the following context:
    
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n==== System Ready! (Pure LCEL Mode) ====")

    while True:
        query = input("\nQuestion (or 'exit'): ")
        if query.lower() == 'exit':
            break

        print("Thinking...")
        try:
            response = rag_chain.invoke(query)
            cleaned_response = response.replace("Answer:", "").strip()
            nikki_variable=len(cleaned_response)
            print(f"\nAnswer: {cleaned_response}")
            print("="*(nikki_variable+8))
        except Exception as e:
            print(f"Error: {e}")


#====================  ENTRY POINT  ====================

if __name__ == "__main__":
    main()
