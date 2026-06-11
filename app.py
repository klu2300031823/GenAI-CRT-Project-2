# import re
# import streamlit as st
# from PyPDF2 import PdfReader
# from sentence_transformers import SentenceTransformer
# from deep_translator import GoogleTranslator
# from langdetect import detect
# import faiss
# import numpy as np

# st.set_page_config(
#     page_title="Multilingual Citizen Service Chatbot",
#     layout="wide"
# )

# st.title("📄 Multilingual Citizen Service Chatbot")
# st.write("Upload a PDF and ask questions in any language.")

# @st.cache_resource
# def load_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# model = load_model()

# pdf = st.file_uploader("Upload PDF", type="pdf")

# if pdf:

#     text = ""

#     with st.spinner("Reading PDF..."):
#         reader = PdfReader(pdf)

#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + " "

#     text = re.sub(r"\s+", " ", text).strip()

#     if not text:
#         st.error("No readable text found in PDF.")
#         st.stop()

#     # Split into sentences
#     sentences = re.split(r'(?<=[.!?])\s+', text)

#     chunks = []

#     for sentence in sentences:
#         sentence = sentence.strip()
#         if len(sentence) > 30:
#             chunks.append(sentence)

#     if not chunks:
#         st.error("Could not create searchable content.")
#         st.stop()

#     with st.spinner("Creating search index..."):

#         embeddings = model.encode(
#             chunks,
#             convert_to_numpy=True
#         )

#         dimension = embeddings.shape[1]

#         index = faiss.IndexFlatL2(dimension)

#         index.add(
#             embeddings.astype("float32")
#         )

#     st.success("PDF processed successfully!")

#     question = st.text_input("Ask your question")

#     if question:

#         try:
#             user_lang = detect(question)
#         except:
#             user_lang = "en"

#         try:
#             question_en = GoogleTranslator(
#                 source="auto",
#                 target="en"
#             ).translate(question)
#         except:
#             question_en = question

#         q_embedding = model.encode(
#             [question_en],
#             convert_to_numpy=True
#         )

#         distances, indices = index.search(
#             q_embedding.astype("float32"),
#             1
#         )

#         answer = chunks[indices[0][0]]

#         st.subheader("🌍 Answers")

#         languages = {
#             "English": "en",
#             "తెలుగు (Telugu)": "te",
#             "हिन्दी (Hindi)": "hi",
#             "தமிழ் (Tamil)": "ta",
#             "ಕನ್ನಡ (Kannada)": "kn"
#         }

#         for title, code in languages.items():

#             try:
#                 if code == "en":
#                     translated = answer
#                 else:
#                     translated = GoogleTranslator(
#                         source="en",
#                         target=code
#                     ).translate(answer)

#                 st.markdown(f"### {title}")
#                 st.write(translated)

#             except:
#                 st.markdown(f"### {title}")
#                 st.write(answer)

#         with st.expander("📖 Source Text"):
#             st.write(answer)
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

@st.cache_data
def create_embeddings(chunks):
    return model.encode(
        chunks,
        convert_to_numpy=True
    )

model = load_model()

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:

    # Process PDF only once
    if (
        "processed_pdf" not in st.session_state
        or st.session_state.processed_pdf != pdf.name
    ):

        with st.spinner("Reading and indexing PDF..."):

            text = ""

            reader = PdfReader(pdf)

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "

            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                st.error("No readable text found in PDF.")
                st.stop()

            # Faster chunking
            chunks = []

            chunk_size = 1000

            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size].strip()

                if len(chunk) > 100:
                    chunks.append(chunk)

            embeddings = create_embeddings(chunks)

            dimension = embeddings.shape[1]

            index = faiss.IndexFlatL2(dimension)

            index.add(
                embeddings.astype("float32")
            )

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.processed_pdf = pdf.name

        st.success("PDF processed successfully!")

    index = st.session_state.index
    chunks = st.session_state.chunks

    question = st.text_input("Ask your question")

    if question:

        try:
            question_en = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(question)
        except:
            question_en = question

        q_embedding = model.encode(
            [question_en],
            convert_to_numpy=True
        )

        distances, indices = index.search(
            q_embedding.astype("float32"),
            1
        )

        answer = chunks[indices[0][0]]

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
                    translated = answer
                else:
                    translated = GoogleTranslator(
                        source="en",
                        target=code
                    ).translate(answer)

                st.markdown(f"### {title}")
                st.write(translated)

            except:
                st.markdown(f"### {title}")
                st.write(answer)

        with st.expander("📖 Source Text"):
            st.write(answer)
