import os
import base64
import tempfile

import numpy as np
import resampy
import soundfile as sf
import streamlit as st

from faster_whisper import WhisperModel
from googletrans import Translator, LANGUAGES
from gtts import gTTS
from deep_translator import GoogleTranslator

# ------------- Page/UI -------------
st.set_page_config(page_title="Language Translator", layout="wide")
st.markdown(
    "<h1 style='text-align: center; font-size: 48px; font-weight: bold; color: #4CAF50;'>🌐 Language Translator</h1>",
    unsafe_allow_html=True,
)

options = ["Text", "Speech - Text", "Text - Speech", "Speech - Speech"]
selected_option = st.radio("Choose an option:", options, index=0, horizontal=True)

translator = Translator()
target_languages = {name.capitalize(): code for code, name in LANGUAGES.items()}

# ------------- Helpers -------------

@st.cache_resource
def load_stt_model():
    # CPU-friendly settings; tiny/int8 keeps memory small on cloud
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def read_wav_as_16k_mono_array(file_path: str) -> np.ndarray:
    """
    Read a WAV file WITHOUT ffmpeg, convert to mono float32 @ 16 kHz for faster-whisper.
    Uses soundfile (libsndfile) + resampy.
    """
    audio, sr = sf.read(file_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:  # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)
    return audio

def transcribe_wav_no_ffmpeg(file_path: str) -> str:
    model = load_stt_model()
    audio = read_wav_as_16k_mono_array(file_path)
    segments, info = model.transcribe(audio, language=None)  # auto-detect
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text)
    return " ".join(text_parts).strip()

def speak(text: str, lang_code: str):
    """gTTS to mp3, play + download, with temp-file cleanup."""
    tts_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts_file_path = tmp.name
        gTTS(text=text, lang=lang_code).save(tts_file_path)

        with open(tts_file_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")

        b64 = base64.b64encode(audio_bytes).decode()
        st.markdown(
            f'<a href="data:audio/mp3;base64,{b64}" download="speech.mp3">📥 Download Audio</a>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"TTS Error: {e}")
    finally:
        if tts_file_path and os.path.exists(tts_file_path):
            os.remove(tts_file_path)

# ------------- Features -------------

if selected_option == "Text":
    st.subheader("Text Translation")
    input_text = st.text_area("Enter text to translate:", placeholder="Type your text here...")
    target_language = st.selectbox("Select target language:", options=target_languages.keys())

    if st.button("Translate Text"):
        if not input_text.strip():
            st.warning("Please enter text to translate.")
        else:
            try:
                translated_text = translator.translate(input_text, dest=target_languages[target_language]).text
                st.success("Translated Text:")
                st.markdown(
                    f"<p style='font-size: 20px; color: #4CAF50;'>{translated_text}</p>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Translation error: {e}")

elif selected_option == "Speech - Text":
    st.subheader("Speech-to-Text Translation (WAV only, no ffmpeg)")
    target_language = st.selectbox("Select target language:", options=target_languages.keys(), key="speech_text_lang")
    wav_file = st.file_uploader("Upload a WAV audio file (mono/stereo, ≤ ~30–60s recommended):", type=["wav"])

    if wav_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_file.read())
            wav_path = tmp.name

        try:
            with st.spinner("Transcribing (no ffmpeg)…"):
                recognized_text = transcribe_wav_no_ffmpeg(wav_path)

            st.success("Recognized Speech:")
            st.text_area("Extracted Text:", recognized_text, height=150)

            if recognized_text.strip():
                translated_text = translator.translate(
                    recognized_text, dest=target_languages[target_language]
                ).text
                st.success("Translated Text:")
                st.markdown(
                    f"<p style='font-size: 20px; color: #4CAF50;'>{translated_text}</p>",
                    unsafe_allow_html=True,
                )
                st.download_button("Download Transcribed Text", recognized_text, file_name="transcribed_text.txt")
                st.download_button("Download Translated Text", translated_text, file_name="translated_text.txt")
            else:
                st.info("No speech recognized. Try a clearer or longer sample.")

        except Exception as e:
            st.error(f"Error recognizing speech: {e}")
        finally:
            os.remove(wav_path)

elif selected_option == "Text - Speech":
    st.subheader("Text-to-Speech Translation")
    input_text = st.text_area("Enter text:", "Hello, how are you?")
    target_language = st.selectbox("Select language to speak:", options=target_languages.keys(), key="tts_lang")

    if st.button("Translate and Speak"):
        if not input_text.strip():
            st.warning("Please enter text.")
        else:
            try:
                translated_text = GoogleTranslator(
                    source="auto", target=target_languages[target_language]
                ).translate(input_text)
                st.success("Translated Text:")
                st.markdown(
                    f"<p style='font-size: 20px; color: #4CAF50;'>{translated_text}</p>",
                    unsafe_allow_html=True,
                )
                speak(translated_text, target_languages[target_language])
            except Exception as e:
                st.error(f"TTS Error: {e}")

elif selected_option == "Speech - Speech":
    st.subheader("Speech-to-Speech Translation (WAV only, no ffmpeg)")
    target_language = st.selectbox("Select target language:", options=target_languages.keys(), key="s2s_lang")
    wav_file = st.file_uploader("Upload a WAV audio file (mono/stereo, ≤ ~30–60s recommended):", type=["wav"])

    if wav_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_file.read())
            wav_path = tmp.name

        try:
            with st.spinner("Transcribing (no ffmpeg)…"):
                recognized_text = transcribe_wav_no_ffmpeg(wav_path)

            st.success("Recognized Speech:")
            st.text_area("Extracted Text:", recognized_text, height=150)

            if recognized_text.strip():
                translated_text = translator.translate(
                    recognized_text, dest=target_languages[target_language]
                ).text
                st.success("Translated Text:")
                st.markdown(
                    f"<p style='font-size: 20px; color: #4CAF50;'>{translated_text}</p>",
                    unsafe_allow_html=True,
                )

                st.success("Translated Speech:")
                speak(translated_text, target_languages[target_language])

                st.download_button("Download Transcribed Text", recognized_text, file_name="transcribed_text.txt")
                st.download_button("Download Translated Text", translated_text, file_name="translated_text.txt")
            else:
                st.info("No speech recognized. Try a clearer or longer sample.")

        except Exception as e:
            st.error(f"Error processing speech-to-speech: {e}")
        finally:
            os.remove(wav_path)

st.markdown("<hr style='border: 1px solid #4CAF50; margin: 40px 0;'>", unsafe_allow_html=True)
