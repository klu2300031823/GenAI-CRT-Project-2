import re
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
import faiss
import numpy as np

st.set_page_config(
    page_title="Multilingual Citizen Service Chatbot",
    layout="wide"
)

st.title("📄 Multilingual Citizen Service Chatbot")
st.write("Upload a PDF and ask questions in any language.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:

    with st.spinner("Reading PDF..."):

        text = ""
        reader = PdfReader(pdf)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        text = re.sub(r"\s+", " ", text).strip()

    if not text:
        st.error("Could not extract text from PDF.")
        st.stop()

    # Smaller chunks for better accuracy
    chunks = []

    chunk_size = 350

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)

    with st.spinner("Creating search index..."):

        embeddings = model.encode(
            chunks,
            convert_to_numpy=True
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(
            embeddings.astype("float32")
        )

    st.success("PDF processed successfully!")

    question = st.text_input(
        "Ask your question"
    )

    if question:

        # Detect question language
        try:
            user_lang = detect(question)
        except:
            user_lang = "en"

        # Translate question to English
        try:
            question_en = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(question)
        except:
            question_en = question

        # Create embedding
        q_embedding = model.encode(
            [question_en],
            convert_to_numpy=True
        )

        # Search best chunk
        distances, indices = index.search(
            q_embedding.astype("float32"),
            1
        )

        best_chunk = chunks[indices[0][0]]

        score = float(distances[0][0])

        # Fixed headings
        languages = {
            "English": "en",
            "తెలుగు (Telugu)": "te",
            "हिन्दी (Hindi)": "hi",
            "தமிழ் (Tamil)": "ta",
            "ಕನ್ನಡ (Kannada)": "kn"
        }

        st.subheader("🌍 Answers")

        for title, code in languages.items():

            try:
                translated = GoogleTranslator(
                    source="auto",
                    target=code
                ).translate(best_chunk)
            except:
                translated = best_chunk

            st.markdown(f"### {title}")
            st.write(translated)

        with st.expander("📖 Source Text"):
            st.write(best_chunk)

        with st.expander("ℹ Search Details"):
            st.write("Detected Language:", user_lang)
            st.write("Translated Question:", question_en)
            st.write("Similarity Distance:", round(score, 4))
