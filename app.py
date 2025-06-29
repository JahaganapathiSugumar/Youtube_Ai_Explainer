import os
import streamlit as st
from pytube import YouTube
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)

load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]

st.title("AI YouTube Explainer & Summarizer")
st.write("Enter a YouTube video URL to get the transcript, summary, and answer questions about the video.")


def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript_data = transcript_list.find_transcript(['en'])
        text = " ".join([item.text for item in transcript_data.fetch()])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("The video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript for this video.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

video_url = st.text_input("YouTube Video URL")

if st.button("Process Video"):
    if video_url:
        with st.spinner("Processing video..."):
            transcript = get_youtube_transcript(video_url)
            if transcript:
                save_transcript_to_file(transcript)
                loader = TextLoader("transcript.txt", encoding="utf-8")
                documents = loader.load()
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(documents)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(docs, embeddings)
                retriever = vectorstore.as_retriever()
                llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                st.session_state.qa_chain = qa_chain
                st.success("Transcript processed! You can now ask questions.")
            else:
                st.warning("Failed to get transcript.")
    else:
        st.warning("Please enter a valid YouTube URL.")

if "qa_chain" in st.session_state:
    question = st.text_input("Ask a question about the video:")
    if question:
        with st.spinner("generating your answer..."):
            response = st.session_state.qa_chain.invoke({"query": question})
            answer = response["result"]
            st.markdown(f"**Answer:** {answer}")

st.markdown("""
    <hr style="margin-top: 50px;">
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Created by <a href="https://github.com/JahaganapathiSugumar" target="_blank">Jahaganapathi Sugumar</a> 🚀
    </div>
""", unsafe_allow_html=True)




