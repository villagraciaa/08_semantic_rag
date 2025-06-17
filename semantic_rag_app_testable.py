
# semantic_rag_app_testable.py

import streamlit as st
import fitz
import numpy as np
import re
import os
import csv
import time
from datetime import datetime
from uuid import uuid4
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

LOG_FILE = "rag_query_log.csv"

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks, model):
    return model.encode(chunks)

def search(query, model, index, chunks, top_k=5):
    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, index)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = [(chunks[i], sims[i]) for i in top_indices]
    return results

def heuristic_recall(query, top_chunks):
    query_lower = query.lower()
    hits = sum(1 for chunk, _ in top_chunks if query_lower in chunk.lower())
    return hits / len(top_chunks)

def measure_latency(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def log_to_csv(query, llm_answer, retrieval_latency, llm_latency, recall, feedback=None, model_used=None):
    fieldnames = ["timestamp", "query", "llm_answer", "retrieval_latency", "llm_latency", "recall", "feedback", "model_used"]
    row = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "llm_answer": llm_answer,
        "retrieval_latency": retrieval_latency,
        "llm_latency": llm_latency,
        "recall": recall,
        "feedback": feedback,
        "model_used": model_used
    }
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def load_local_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

def generate_answer_local(query, top_chunks, tokenizer, model, max_tokens=256):
    context = "\n\n".join([chunk for chunk, _ in top_chunks])
    prompt = f"""Use the following context to answer the question:

    ---CONTEXT START---
    {context}
    ---CONTEXT END---

    Question: {query}
    Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, top_p=0.95, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[-1].strip()

# ---------- TEST EXECUTION ----------
if __name__ == "__main__":
    sample_pdf = "sample_test_doc.pdf"
    preset_query = "What are the KPIs mentioned?"
    expected_keywords = ["Revenue", "Gross Margin", "Customer Retention"]

    raw_text = extract_text_from_pdf(sample_pdf)
    chunks = chunk_text(raw_text)
    model = load_model()
    embeddings = embed_chunks(chunks, model)
    index = np.array(embeddings)

    results, retrieval_latency = measure_latency(search, preset_query, model, index, chunks)
    tokenizer, llm = load_local_llm("mistralai/Mistral-7B-Instruct-v0.2")
    llm_answer, llm_latency = measure_latency(generate_answer_local, preset_query, results, tokenizer, llm)
    recall = heuristic_recall(preset_query, results)

    print(f"Query: {preset_query}")
    print(f"Answer: {llm_answer}")
    print(f"Latency: Retrieval={retrieval_latency:.2f}s, LLM={llm_latency:.2f}s, Recall={recall:.2f}")

    # Simple validation
    matched_keywords = [kw for kw in expected_keywords if kw.lower() in llm_answer.lower()]
    feedback = "positive" if len(matched_keywords) >= 2 else "negative"
    log_to_csv(preset_query, llm_answer, retrieval_latency, llm_latency, recall, feedback, "mistralai/Mistral-7B-Instruct-v0.2")
