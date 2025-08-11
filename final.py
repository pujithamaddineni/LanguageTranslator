import streamlit as st
import tempfile
import os
import whisper
from pydub import AudioSegment
from googletrans import Translator, LANGUAGES
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64

# This list is for the language selection UI, allowing you to choose the display name.
ui_languages = {
    "English": "en", "French": "fr", "Spanish": "es", "German": "de", "Chinese": "zh-CN",
    "Japanese": "ja", "ko": "Korean", "Russian": "ru", "ar": "Arabic", "pt": "Portuguese",
    "it": "Italian", "tr": "Turkish", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "gu": "Gujarati", "bn": "Bengali",
    "pa": "Punjabi", "ur": "Urdu", "or": "Odia", "as": "Assamese"
}

# Configure the page layout and title
st.set_page_config(page_title="Language Translator", layout="wide")
st.markdown("""
    <h1 style='text-align: center; font-size: 48px; font-weight: bold; color: #4CAF50;'>🌐 Language Translator</h1>
""", unsafe_allow_html=True)

# Navigation options, excluding the non-deployable "Speech - Speech" option
options = ["Text", "Speech - Text", "Text - Speech"]
selected_option = st.radio("Choose an option:", options, index=0, horizontal=True)

# Initialize the translation object
translator = Translator()

# Create a dictionary for easy lookup from the language name to the code
target_languages = {name.capitalize(): code for code, name in LANGUAGES.items()}

# --- TTS playback with improved temporary file handling ---
def speak(text, lang_code):
    """
    Translates text to speech using gTTS and plays it in the browser.
    It handles temporary file creation and cleanup robustly.
    """
    # Use tempfile to create a temporary file with a unique name
    # The 'delete=False' argument is crucial to prevent the file from
    # being deleted as soon as the context manager exits.
    tts_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts_file_path = temp_file.name
            tts = gTTS(text=text, lang=lang_code)
            tts.save(tts_file_path)

        # Read the file and provide it to st.audio
        with open(tts_file_path, "rb") as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")

        # Create a download link for the user
        b64 = base64.b64encode(audio_bytes).decode()
        st.markdown(f'<a href="data:audio/mp3;base64,{b64}" download="speech.mp3">📥 Download Audio</a>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"TTS Error: {e}")
    finally:
        # Clean up the temporary file
        if tts_file_path and os.path.exists(tts_file_path):
            os.remove(tts_file_path)

# --- TEXT TRANSLATION ---
if selected_option == "Text":
    st.subheader("Text Translation")
    input_text = st.text_area("Enter text to translate:", placeholder="Type your text here...")
    target_language = st.selectbox("Select target language:", options=target_languages.keys())
    
    if st.button("Translate Text"):
        if input_text:
            try:
                translated_text = translator.translate(input_text, dest=target_languages[target_language]).text
                st.success("Translated Text:")
                st.markdown(f"<p style='font-size: 20px; color: #4CAF50;'>{translated_text}</p>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter text to translate.")

# --- SPEECH TO TEXT TRANSLATION ---
elif selected_option == "Speech - Text":
    st.subheader("Speech-to-Text Translation")
    model = whisper.load_model("tiny")
    target_language = st.selectbox("Select target language:", options=target_languages.keys(), key="speech_lang")
    audio_file = st.file_uploader("Upload an audio file (wav/mp3/ogg):", type=["wav", "mp3", "ogg"])
    
    if audio_file:
        audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as temp_audio:
                temp_audio.write(audio_file.read())
                audio_path = temp_audio.name
            
            st.info("Converting audio to WAV...")
            sound = AudioSegment.from_file(audio_path)
            wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            sound.export(wav_path, format="wav")
            
            with st.spinner("Processing audio with Whisper..."):
                result = model.transcribe(wav_path)
                recognized_text = result["text"]
                st.success("Recognized Speech:")
                st.text_area("Extracted Text:", recognized_text, height=150)
                
                translated_text = translator.translate(recognized_text, dest=target_languages[target_language]).text
                st.success("Translated Text:")
                st.markdown(f"<p style='font-size: 20px; color: #4CAF50;'>{translated_text}</p>", unsafe_allow_html=True)
                
                st.download_button("Download Transcribed Text", recognized_text, file_name="transcribed_text.txt")
                st.download_button("Download Translated Text", translated_text, file_name="translated_text.txt")
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)

# --- TEXT TO SPEECH TRANSLATION ---
elif selected_option == "Text - Speech":
    st.subheader("Text-to-Speech Translation")
    input_text = st.text_area("Enter text to translate:", "Hello, how are you?")
    target_language = st.selectbox("Select language to speak:", options=target_languages.keys(), key="tts_lang")

    if st.button("Translate and Speak"):
        if input_text:
            try:
                translated_text = GoogleTranslator(source="auto", target=target_languages[target_language]).translate(input_text)
                st.success(f"*Translated Text:* {translated_text}")
                speak(translated_text, target_languages[target_language])
            except Exception as e:
                st.error(f"TTS Error: {e}")
        else:
            st.warning("Please enter text.")

# Footer
st.markdown("<hr style='border: 1px solid #4CAF50; margin: 40px 0;'>", unsafe_allow_html=True)
