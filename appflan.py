# ========== appflan.py (Versi Tanpa API / Load Lokal) ==========

import os
import torch
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA

st.set_page_config(page_title="HOI4 Chatbot (FLAN-T5)", layout="wide")
st.title("🧠 HOI4 Chatbot (FLAN-T5 Local)")
st.markdown("Tanya jawab berbasis Wiki Hearts of Iron IV menggunakan model **FLAN-T5** secara lokal.")

st.info("📥 Memuat vectorstore...")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore_path = "hoi4_vectorstore"

if not os.path.exists(vectorstore_path):
    st.error("❌ Folder vectorstore tidak ditemukan. Jalankan proses penyimpanan terlebih dahulu.")
    st.stop()

@st.cache_resource
def load_vectorstore(path, _embeddings):
    return FAISS.load_local(path, _embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore(vectorstore_path, embedding_model)
st.success("✅ Vectorstore berhasil dimuat.")

st.info("🤖 Memuat model FLAN-T5...")

@st.cache_resource
def load_flan_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    if torch.cuda.is_available():
        model = model.to("cuda")
    flan_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )
    return HuggingFacePipeline(pipeline=flan_pipeline)

flan_llm = load_flan_model()
st.success("✅ Model FLAN-T5 berhasil dimuat.")

flan_qa = RetrievalQA.from_chain_type(llm=flan_llm, retriever=vectorstore.as_retriever())

if "flan_messages" not in st.session_state:
    st.session_state.flan_messages = []

st.markdown("---")
st.subheader("💬 Tanyakan sesuatu seputar HoI4")
user_input = st.text_input("Pertanyaan:", placeholder="Contoh: Bagaimana cara membuat faksi sendiri?", key="flan_user_input")

if user_input:
    with st.spinner("🔍 Mengambil jawaban dari FLAN-T5..."):
        try:
            result = flan_qa.invoke({"query": user_input})
            answer = result["result"] if isinstance(result, dict) else result
            st.session_state.flan_messages.append((user_input, answer))
        except Exception as e:
            st.session_state.flan_messages.append((user_input, f"❌ Error: {e}"))

st.markdown("---")
st.subheader("📜 Riwayat Chat")

if st.session_state.flan_messages:
    for idx, (q, a) in enumerate(reversed(st.session_state.flan_messages), 1):
        st.markdown(f"### 🔹 Pertanyaan {idx}:")
        st.markdown(f"**🧍 Kamu:** {q}")
        st.markdown("🧠 **FLAN-T5:**")
        st.success(a)
        st.markdown("---")
else:
    st.info("Belum ada percakapan.")
