import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from langdetect import detect

st.set_page_config(page_title="Citizen Service Chatbot")

st.title("📄 Multilingual Citizen Service Chatbot")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:

    text = ""

    reader = PdfReader(pdf)

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    if len(paragraphs) == 0:
        st.error("No readable text found in PDF")
        st.stop()

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

        docs = paragraphs + [q_en]

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(docs)

        scores = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        ).flatten()

        idx = scores.argmax()

        answer = paragraphs[idx]

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
            st.write(paragraphs[idx])
