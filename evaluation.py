import os
import json
import time
import numpy as np
import warnings
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
# =================================================
# ================= CONFIGURATION =================
# =================================================
CHUNKING_STRATEGIES = {
    "Small": {"chunk_size": 250, "chunk_overlap": 25},
    "Medium": {"chunk_size": 550, "chunk_overlap": 50},
    "Large": {"chunk_size": 900, "chunk_overlap": 100}
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"
TOP_K = 2
# ====================================================
# ================= HELPER FUNCTIONS =================
# ====================================================

def load_documents():
    """
    Loads all .txt files from ./corpus directory.
    """
    return DirectoryLoader("./corpus", glob="*.txt", loader_cls=TextLoader).load()

def build_rag_chain(documents, chunk_size, chunk_overlap):
    """
    Builds a temporary RAG pipeline for evaluation of a specific chunking size.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=f"eval_chunk_{chunk_size}"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = Ollama(model=LLM_MODEL)

    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        """
        Converts list of documents into a single combined string.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, embeddings
#========================================================
# ================= METRICS CALCULATORS =================
#========================================================

def calculate_retrieval_metrics(retrieved_docs, ground_truth_sources):
    """
    Computes Hit Rate and MRR using retrieved doc file names vs ground truth files.
    """
    hits = 0
    reciprocal_rank = 0

    retrieved_sources = [os.path.basename(doc.metadata['source']) for doc in retrieved_docs]

    if any(source in retrieved_sources for source in ground_truth_sources):
        hits = 1
        for i, source in enumerate(retrieved_sources):
            if source in ground_truth_sources:
                reciprocal_rank = 1 / (i + 1)
                break

    return hits, reciprocal_rank

def calculate_text_metrics(generated_answer, ground_truth_answer, embed_model):
    """
    Computes ROUGE-L, BLEU and cosine similarity between generated and gold answers.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(ground_truth_answer, generated_answer)['rougeL'].fmeasure

    reference = ground_truth_answer.split()
    candidate = generated_answer.split()
    bleu = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)

    vec1 = embed_model.embed_query(generated_answer)
    vec2 = embed_model.embed_query(ground_truth_answer)
    similarity = cosine_similarity([vec1], [vec2])[0][0]

    return rouge_l, bleu, similarity
# ========================================================
# ================= MAIN EVALUATION LOOP =================
# ========================================================

def main():
    """
    Runs evaluation for all chunking strategies and saves results to test_results.json.
    """
    print("--- Starting Comprehensive RAG Evaluation ---")

    with open("test_dataset.json", "r") as f:
        test_data = json.load(f)

    docs = load_documents()
    results = {}

    for strategy_name, config in CHUNKING_STRATEGIES.items():
        print(f"\nTesting Strategy: {strategy_name} (Size: {config['chunk_size']})")

        rag_chain, retriever, embed_model = build_rag_chain(
            docs,
            config['chunk_size'],
            config['chunk_overlap']
        )

        strategy_results = []

        for item in tqdm(test_data['test_questions'], desc=f"Running {strategy_name}"):
            if not item['answerable']:
                continue

            question = item['question']
            ground_truth = item['ground_truth']
            gt_sources = item['source_documents']

            retrieved_docs = retriever.invoke(question)

            try:
                generated_answer = rag_chain.invoke(question)
                generated_answer = generated_answer.replace("Answer:", "").strip()
            except Exception:
                generated_answer = "Error"

            hit, mrr = calculate_retrieval_metrics(retrieved_docs, gt_sources)
            rouge, bleu, cos_sim = calculate_text_metrics(generated_answer, ground_truth, embed_model)

            strategy_results.append({
                "id": item['id'],
                "question": question,
                "metrics": {
                    "hit_rate": hit,
                    "mrr": mrr,
                    "rouge_l": rouge,
                    "bleu": bleu,
                    "cosine_similarity": cos_sim
                }
            })

        avg_metrics = {
            "avg_hit_rate": np.mean([r['metrics']['hit_rate'] for r in strategy_results]),
            "avg_mrr": np.mean([r['metrics']['mrr'] for r in strategy_results]),
            "avg_cosine": np.mean([r['metrics']['cosine_similarity'] for r in strategy_results]),
            "avg_rouge": np.mean([r['metrics']['rouge_l'] for r in strategy_results])
        }

        results[strategy_name] = {
            "config": config,
            "averages": avg_metrics,
            "details": strategy_results
        }

        print(
            f"Results for {strategy_name}: "
            f"Hit Rate: {avg_metrics['avg_hit_rate']:.2f} | "
            f"Cosine: {avg_metrics['avg_cosine']:.2f}"
        )

    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nEvaluation Complete! Results saved to 'test_results.json'")
    
# ============================ EnD ===================================================================

if __name__ == "__main__":
    main()
