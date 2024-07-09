import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import soundfile as sf
import numpy as np
import io
import tempfile

# Load the Whisper model
model = whisper.load_model("base")

def transcribe_audio(audio):
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_wav_file:
        tmp_wav_file.write(audio["bytes"])
        tmp_wav_file.flush()

        # Transcribe audio using Whisper
        result = model.transcribe(tmp_wav_file.name)
        return result['text']

# Streamlit interface
st.title("Audio Recording and Transcription with Whisper")

st.write("Record your audio and transcribe it to text.")

audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", format="webm")

if audio:
    st.audio(audio['bytes'], format='audio/webm')
    transcription = transcribe_audio(audio)
    st.write("Transcription:")
    st.write(transcription)
