import os
import requests
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================
# Konfigurasi Streamlit
# ==========================
st.set_page_config(page_title="HOI4 Chatbot (FLAN-T5 API)", layout="wide")
st.title("🧠 HOI4 Chatbot (FLAN-T5 API)")
st.markdown("Tanya jawab berbasis Wiki Hearts of Iron IV menggunakan **FLAN-T5 via Hugging Face API**.")

# ==========================
# Load Vectorstore
# ==========================
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

# ==========================
# API Hugging Face
# ==========================
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Token disimpan di Streamlit secrets
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query_flant5_api(prompt: str):
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 256}}
    )
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return "❌ Format respon API tidak sesuai."
    else:
        return f"❌ Error {response.status_code}: {response.text}"

# ==========================
# Session State untuk Chat
# ==========================
if "flan_messages" not in st.session_state:
    st.session_state.flan_messages = []

# ==========================
# UI Input
# ==========================
st.markdown("---")
st.subheader("💬 Tanyakan sesuatu seputar HoI4")
user_input = st.text_input("Pertanyaan:", placeholder="Contoh: Bagaimana cara membuat faksi sendiri?", key="flan_user_input")

# ==========================
# Proses Jawaban
# ==========================
if user_input:
    with st.spinner("🔍 Mengambil jawaban dari FLAN-T5..."):
        try:
            context_docs = vectorstore.similarity_search(user_input, k=3)
            context_text = "\n".join([doc.page_content for doc in context_docs])
            prompt = f"Context:\n{context_text}\n\nQuestion: {user_input}\nAnswer:"
            flan_answer = query_flant5_api(prompt)
            st.session_state.flan_messages.append((user_input, flan_answer))
        except Exception as e:
            st.session_state.flan_messages.append((user_input, f"❌ Error: {e}"))

# ==========================
# Tampilkan Riwayat Chat
# ==========================
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
