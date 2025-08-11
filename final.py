import streamlit as st
import tempfile
import os
import whisper
from googletrans import Translator, LANGUAGES
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64

# Language dictionary for UI options
ui_languages = {
    "English": "en", "French": "fr", "Spanish": "es", "German": "de", "Chinese": "zh-CN",
    "Japanese": "ja", "Korean": "ko", "Russian": "ru", "Arabic": "ar", "Portuguese": "pt",
    "Italian": "it", "Turkish": "tr", "Hindi": "hi", "Telugu": "te", "Tamil": "ta",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Gujarati": "gu", "Bengali": "bn",
    "Punjabi": "pa", "Urdu": "ur", "Odia": "or", "Assamese": "as"
}

# Configure page
st.set_page_config(page_title="Language Translator", layout="wide")
st.markdown("<h1 style='text-align: center; font-size: 48px; color: #4CAF50;'>🌐 Language Translator</h1>", unsafe_allow_html=True)

# Navigation
options = ["Text", "Speech - Text", "Text - Speech"]
selected_option = st.radio("Choose an option:", options, horizontal=True)

# Translators
translator = Translator()
target_languages = {name.capitalize(): code for code, name in LANGUAGES.items()}

# TTS playback
def speak(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    path = "temp_tts.mp3"
    tts.save(path)
    with open(path, "rb") as f:
        audio = f.read()
        b64 = base64.b64encode(audio).decode()
        st.audio(audio, format="audio/mp3")
        st.markdown(f'<a href="data:audio/mp3;base64,{b64}" download="speech.mp3">📥 Download Audio</a>', unsafe_allow_html=True)
    os.remove(path)

# --- TEXT TRANSLATION ---
if selected_option == "Text":
    st.subheader("Text Translation")
    input_text = st.text_area("Enter text to translate:")
    target_language = st.selectbox("Select target language:", options=target_languages.keys())
    
    if st.button("Translate Text"):
        if input_text:
            try:
                translated = translator.translate(input_text, dest=target_languages[target_language]).text
                st.success("Translated Text:")
                st.markdown(f"<p style='font-size: 20px; color: #4CAF50;'>{translated}</p>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Translation error: {e}")
        else:
            st.warning("Please enter text.")

# --- SPEECH TO TEXT TRANSLATION ---
elif selected_option == "Speech - Text":
    st.subheader("Speech-to-Text Translation (WAV only)")
    target_language = st.selectbox("Select target language:", options=target_languages.keys())

    audio_file = st.file_uploader("Upload a WAV audio file:", type=["wav"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name
    
        try:
            with st.spinner("Processing audio..."):
                model = whisper.load_model("tiny")
                result = model.transcribe(temp_audio_path, fp16=False)  # Disable GPU-only mode
                recognized_text = result["text"]
                st.success("Recognized Speech:")
                st.text_area("Extracted Text:", recognized_text, height=150)

                translated_text = translator.translate(recognized_text, dest=target_languages[target_language]).text
                st.success("Translated Text:")
                st.markdown(f"<p style='font-size: 20px; color: #4CAF50;'>{translated_text}</p>", unsafe_allow_html=True)

                st.download_button("Download Transcribed Text", recognized_text, file_name="transcribed_text.txt")
                st.download_button("Download Translated Text", translated_text, file_name="translated_text.txt")
        except Exception as e:
            st.error(f"Error recognizing speech: {str(e)}")
        finally:
            os.remove(temp_audio_path)


# --- TEXT TO SPEECH TRANSLATION ---
elif selected_option == "Text - Speech":
    st.subheader("Text-to-Speech Translation")
    input_text = st.text_area("Enter text:")
    target_language = st.selectbox("Select language to speak:", options=target_languages.keys(), key="tts_lang")

    if st.button("Translate and Speak"):
        if input_text:
            try:
                translated = GoogleTranslator(source="auto", target=target_languages[target_language]).translate(input_text)
                st.success("Translated Text:")
                st.markdown(f"<p style='font-size: 20px; color: #4CAF50;'>{translated}</p>", unsafe_allow_html=True)
                speak(translated, target_languages[target_language])
            except Exception as e:
                st.error(f"TTS Error: {e}")
        else:
            st.warning("Please enter text.")

# Footer
st.markdown("<hr style='border: 1px solid #4CAF50; margin: 40px 0;'>", unsafe_allow_html=True)
