import streamlit as st
import fitz  # PyMuPDF
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

# ========== UTILITY FUNCTIONS ==========

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
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

@st.cache_resource
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

def highlight_text(base_text, matches):
    for chunk, _ in matches:
        escaped = re.escape(chunk[:60])
        pattern = re.compile(f"({escaped})", re.IGNORECASE)
        base_text = pattern.sub(r"<mark>\1</mark>", base_text)
    return base_text

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

def generate_answer_with_openai(query, top_chunks, openai_api_key, model_name="gpt-3.5-turbo"):
    import openai
    openai.api_key = openai_api_key
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, (chunk, _) in enumerate(top_chunks)])
    prompt = f"""You are a helpful assistant. Use the following context to answer the question:

    --- DOCUMENT CONTEXT START ---
    {context}
    --- DOCUMENT CONTEXT END ---

    Question: {query}
    Answer:"""
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert assistant for document QA."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=512
    )
    return response['choices'][0]['message']['content']

@st.cache_resource
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

# ========== STREAMLIT APP ==========

st.set_page_config(page_title="Semantic RAG Engine", layout="wide")
tab1, tab2 = st.tabs(["📚 Documentation", "🚀 Semantic RAG App"])
st.title("📚 Semantic Search & RAG Engine for Enterprise Docs")
st.caption("By Dr. Al Rey Villagracia")
with tab1:   
    st.markdown("""
    This app allows semantic search and RAG-style answering on uploaded PDF documents using OpenAI or Local LLMs (e.g., Mistral, LLaMA).

    ### Usage:
    1. Choose the LLM backend
    2. Upload a PDF
    3. Ask a question
    4. See highlighted matches, answers, and performance metrics
    """)

with tab2:
    st.title("🚀 Semantic RAG App")

    model_choice = st.selectbox("🧠 Choose LLM for Answer Generation:",
                                ["OpenAI GPT-3.5", "Local Hugging Face Model"], index=0)

    openai_api_key = None
    hf_model_name = None

    if model_choice == "OpenAI GPT-3.5":
        openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password", placeholder="sk-...")
    else:
        hf_model_name = st.selectbox("🤗 Select a Hugging Face Model", [
            "mistralai/Magistral-Small-2506"
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "tiiuae/falcon-7b-instruct"
        ])

    uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")
    if uploaded_file:
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)
        model = load_model()
        embeddings = embed_chunks(chunks, model)
        index = np.array(embeddings)

        query = st.text_input("🔍 Ask a question about the document:")

        if query:
            (results, retrieval_latency) = measure_latency(search, query, model, index, chunks)

            if model_choice == "OpenAI GPT-3.5":
                if not openai_api_key:
                    st.warning("⚠️ Please enter your OpenAI API key.")
                    st.stop()
                (llm_answer, llm_latency) = measure_latency(
                    generate_answer_with_openai, query, results, openai_api_key
                )
                model_used = "OpenAI GPT-3.5"
            else:
                tokenizer, hf_model = load_local_llm(hf_model_name)
                (llm_answer, llm_latency) = measure_latency(
                    generate_answer_local, query, results, tokenizer, hf_model
                )
                model_used = hf_model_name

            recall = heuristic_recall(query, results)
            st.subheader("🤖 Answer")
            st.markdown(f"> {llm_answer}")

            with st.expander("📄 Document Preview with Highlights"):
                highlighted = highlight_text(raw_text, results)
                st.markdown(f"<div style='background:#f0f0f0;padding:1em'>{highlighted}</div>", unsafe_allow_html=True)

            st.subheader("🗳️ Was this answer helpful?")
            col1, col2 = st.columns(2)
            feedback = None
            with col1:
                if st.button("👍 Yes", key=uuid4()):
                    feedback = "positive"
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎 No", key=uuid4()):
                    feedback = "negative"
                    st.warning("We'll improve this!")

            log_to_csv(query, llm_answer, retrieval_latency, llm_latency, recall, feedback, model_used)

            with st.expander("📊 Performance Metrics"):
                st.metric("🔁 Retrieval Latency", f"{retrieval_latency:.2f} s")
                st.metric("🤖 LLM Latency", f"{llm_latency:.2f} s")
                st.metric("🎯 Recall@k", f"{recall:.2f}")
