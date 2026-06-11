import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
import faiss
import numpy as np

st.set_page_config(page_title="Multilingual Citizen Service Chatbot")
st.title("📄 Multilingual Citizen Service Chatbot")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    text = ""

    reader = PdfReader(pdf)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        st.error("No readable text found in PDF.")
        st.stop()

    chunks = [
        text[i:i+500]
        for i in range(0, len(text), 500)
    ]

    with st.spinner("Processing PDF..."):
        embeddings = model.encode(chunks)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype=np.float32))

    st.success("PDF Processed Successfully!")

    question = st.text_input("Ask your question")

    if question:

        try:
            lang = detect(question)
        except:
            lang = "en"

        try:
            q_en = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(question)
        except:
            q_en = question

        q_embedding = model.encode([q_en])

        D, I = index.search(
            np.array(q_embedding, dtype=np.float32),
            3
        )

        answer = "\n\n".join(
            [chunks[i] for i in I[0]]
        )

        try:
            if lang != "en":
                answer = GoogleTranslator(
                    source="en",
                    target=lang
                ).translate(answer)
        except:
            pass

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Source Text"):
            for i in I[0]:
                st.write(chunks[i])
                st.write("---")
