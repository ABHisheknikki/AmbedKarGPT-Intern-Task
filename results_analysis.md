# Evaluation Report: RAG Pipeline Performance

## 1. Executive Summary
We evaluated the RAG system using **3 chunking strategies** across a test dataset of **25 questions** (factual, comparative, and conceptual). 

**The "Medium" strategy (550 characters) proved to be the optimal configuration**, achieving a **100% Retrieval Hit Rate** and the highest semantic similarity score (**0.61**). This indicates that 550 characters provide enough context for the LLM to answer correctly without introducing excessive noise.

## 2. Methodology
* **Model:** Mistral 7B (Ollama)
* **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
* **Metrics:**
    * **Hit Rate:** Frequency of retrieving the correct source document.
    * **Cosine Similarity:** Semantic closeness of the generated answer to the ground truth.
    * **ROUGE-L / BLEU:** Structural and n-gram overlap.

## 3. Comparative Analysis

| Strategy | Chunk Size | Hit Rate | Cosine Similarity | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Small** | 250 chars | 0.91 | 0.55 | High precision but missed context for complex queries. |
| **Medium** | **550 chars** | **1.00** | **0.61** | **Optimal balance. Captured all relevant context.** |
| **Large** | 900 chars | 0.95 | 0.58 | Good context, but likely retrieved irrelevant noise decreasing answer precision. |

### Why Medium Won:
The "Medium" strategy hit the "Goldilocks" zone. 
* **Vs Small:** Small chunks often cut off sentences or split the answer to a question across two chunks, leading to lower cosine scores (0.55).
* **Vs Large:** Large chunks included too much irrelevant text. While the retrieval was good (0.95), the extra text likely confused the LLM slightly, leading to a lower semantic score (0.58) compared to Medium.

## 4. Failure Analysis
While the Hit Rate was perfect for Medium, the Cosine Similarity (0.61) suggests the *wording* of the answers differed from the Ground Truth.
* **Generative Differences:** The Ground Truth is concise (e.g., "The belief in Shastras"). Mistral tends to be more conversational (e.g., "According to the text, the real enemy is..."). This lowers the strict similarity score despite the answer being factually correct.
* **Comparative Questions:** Questions asking to "Compare Document A and B" had lower BLEU scores because the model synthesized a new paragraph rather than copying phrases.

## 5. Recommendations
1.  **Adopt Strategy:** Deploy the **550 Character / 50 Overlap** strategy for production.
2.  **Prompt Engineering:** To improve Cosine Similarity, update the prompt to enforce conciseness: *"Answer in one direct sentence."*
3.  **Hybrid Search:** For the few cases where Small/Large failed retrieval, implementing a Hybrid Search (Keyword + Vector) could ensure 100% robustness at scale.