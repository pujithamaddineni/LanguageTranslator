import streamlit as st
import whisper
import tempfile
import os

st.title("Speech to Text Translator 🌍")
st.write("Upload an audio file and get the transcription using Whisper")

# Load model only once
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

model = load_model()

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/mp3")
    st.write("Transcribing...")

    result = model.transcribe(tmp_path)
    st.subheader("Transcription:")
    st.success(result["text"])

    os.remove(tmp_path)
