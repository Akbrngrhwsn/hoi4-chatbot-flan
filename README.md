# HOI4 Chatbot (FLAN-T5 API)

Aplikasi Streamlit berbasis Retrieval-Augmented Generation (RAG) yang menjawab pertanyaan seputar Hearts of Iron IV menggunakan model FLAN-T5 melalui Hugging Face Inference API.

## Cara Deploy
1. Upload folder ini ke GitHub.
2. Deploy di https://streamlit.io/cloud.
3. Tambahkan `HF_API_TOKEN` ke bagian `Secrets` di Streamlit Cloud.

## Struktur
- `appflan.py` - Streamlit app
- `hoi4_vectorstore/` - FAISS index (upload file sendiri)
- `requirements.txt` - Dependensi Python
