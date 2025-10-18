import streamlit as st
import requests
from xml.etree import ElementTree
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# -------------------- PubMed Fetch --------------------
def get_pubmed_studies_treatments(query, max_results=15):
    treatment_query = f"{query} treatment therapy medication drug intervention"
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    search_params = {
        "db": "pubmed",
        "term": treatment_query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json"
    }
    search_res = requests.get(f"{base_url}esearch.fcgi", params=search_params)
    ids = search_res.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    
    fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
    fetch_res = requests.get(f"{base_url}efetch.fcgi", params=fetch_params)
    root = ElementTree.fromstring(fetch_res.content)
    
    studies = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle")
        abstract = article.findtext(".//AbstractText")
        pmid = article.findtext(".//PMID")
        if title and abstract:
            combined_text = f"{title} {abstract}".lower()
            if any(term in combined_text for term in 
                   ['treatment', 'therapy', 'drug', 'medication', 'intervention', 'efficacy']):
                studies.append({"pmid": pmid, "title": title, "abstract": abstract})
    return studies

# -------------------- Chunking --------------------
def chunk_with_treatment_focus(papers, chunk_size=400, overlap=100):
    chunks = []
    for paper in papers:
        text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
        pmid = paper["pmid"]
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if any(term in current_chunk.lower() for term in 
                       ['treatment', 'therapy', 'drug', 'medication', 'intervention']):
                    chunks.append({"pmid": pmid, "chunk_text": current_chunk.strip()})
                current_chunk = sentence + ". "
        if current_chunk and any(term in current_chunk.lower() for term in 
                                 ['treatment', 'therapy', 'drug', 'medication', 'intervention']):
            chunks.append({"pmid": pmid, "chunk_text": current_chunk.strip()})
    return chunks

# -------------------- Retrieval --------------------
def retrieve_treatment_chunks(query, index, embedder, texts, top_k=7):
    query_emb = embedder.encode([query])
    distances, indices = index.search(query_emb, top_k * 2)
    treatment_chunks = []
    for i in indices[0]:
        chunk_text = texts[i]
        score = sum(term in chunk_text.lower() for term in 
                    ['treatment', 'therapy', 'drug', 'medication', 'intervention'])
        treatment_chunks.append((chunk_text, score))
    treatment_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk[0] for chunk in treatment_chunks[:top_k]]

# -------------------- QA --------------------
def medical_treatment_qa(question, retrieved_chunks):
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    treatment_context = " ".join(retrieved_chunks)
    treatment_context = treatment_context[:3000]
    
    prompt = f"""
    Based on the following medical research abstracts, provide a detailed answer about treatments.
    Question: {question}
    Context: {treatment_context}
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_length=500, temperature=0.3, top_p=0.85)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------- Streamlit UI --------------------
def main():
    st.set_page_config(page_title="Medical Treatment Finder", page_icon="🧬", layout="wide")

    st.title("🧠 Medical Treatment Research Assistant")
    st.markdown("""
    <p style='font-size:18px;'>
    Search PubMed for the latest <b>treatment-focused</b> medical studies and get detailed AI-generated insights.  
    
    </p>
    """, unsafe_allow_html=True)

    query = st.text_input("🔍 Enter a medical topic (e.g. 'heart failure', 'diabetes', 'lung cancer'):")
    question = st.text_area("💬 Ask a question (e.g. 'What are the latest treatments for heart failure?'):")

    if st.button("Generate Answer", type="primary"):
        if not query or not question:
            st.warning("⚠️ Please enter both a topic and a question.")
            return
        
        with st.spinner("🔎 Fetching PubMed studies..."):
            papers = get_pubmed_studies_treatments(query)
        
        if not papers:
            st.error("❌ No treatment-related studies found.")
            return
        
        st.success(f"✅ Retrieved {len(papers)} studies.")
        
        with st.spinner("📑 Chunking abstracts..."):
            chunks = chunk_with_treatment_focus(papers)
        st.success(f"✅ Created {len(chunks)} treatment-focused chunks.")
        
        with st.spinner("⚙️ Building embeddings and FAISS index..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [ch["chunk_text"] for ch in chunks]
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
        st.success("✅ Just wait a little bit more")
        
        with st.spinner("🧠 Retrieving and generating answer..."):
            retrieved = retrieve_treatment_chunks(question, index, model, texts)
            answer = medical_treatment_qa(question, retrieved)
        
        st.markdown("## 💡 AI-Generated Answer")
        st.markdown(f"<div style='background-color:#e6f7ff; padding:15px; border-radius:10px;'>{answer}</div>", unsafe_allow_html=True)
        
        st.markdown("## 📚 Key Treatment Snippets")
        for i, chunk in enumerate(retrieved[:3], 1):
            st.markdown(f"**Snippet {i}:**")
            st.markdown(f"<blockquote>{chunk[:400]}...</blockquote>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
