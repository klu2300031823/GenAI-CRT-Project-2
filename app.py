import re
import streamlit as st
import google.generativeai as genai
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

# Gemini API Key from Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.5-flash")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:

    text = ""

    with st.spinner("Reading PDF..."):

        reader = PdfReader(pdf)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        st.error("No readable text found in PDF.")
        st.stop()

    chunks = []

    chunk_size = 800

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 100:
            chunks.append(chunk)

    with st.spinner("Creating search index..."):

        embeddings = embed_model.encode(
            chunks,
            convert_to_numpy=True
        )

        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)

        index.add(
            embeddings.astype("float32")
        )

    st.success("PDF processed successfully!")

    question = st.text_input("Ask your question")

    if question:

        try:
            user_lang = detect(question)
        except:
            user_lang = "en"

        try:
            question_en = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(question)
        except:
            question_en = question

        q_embedding = embed_model.encode(
            [question_en],
            convert_to_numpy=True
        )

        distances, indices = index.search(
            q_embedding.astype("float32"),
            3
        )

        context = "\n\n".join(
            [chunks[i] for i in indices[0]]
        )

        prompt = f"""
You are a citizen service assistant.

Answer the user's question ONLY using the context below.

Give a short and clear answer.

If the answer is not found, say:
"Information not found in the document."

Context:
{context}

Question:
{question_en}
"""

        with st.spinner("Generating answer..."):

            try:
                response = gemini.generate_content(prompt)
                english_answer = response.text.strip()
            except Exception as e:
                st.error(f"Gemini Error: {e}")
                st.stop()

        st.subheader("🌍 Answers")

        languages = {
            "English": "en",
            "తెలుగు (Telugu)": "te",
            "हिन्दी (Hindi)": "hi",
            "தமிழ் (Tamil)": "ta",
            "ಕನ್ನಡ (Kannada)": "kn"
        }

        for title, code in languages.items():

            try:
                if code == "en":
                    translated = english_answer
                else:
                    translated = GoogleTranslator(
                        source="en",
                        target=code
                    ).translate(english_answer)

                st.markdown(f"### {title}")
                st.write(translated)

            except:
                st.markdown(f"### {title}")
                st.write("Translation unavailable")

        with st.expander("📖 Source Context"):
            st.write(context)
