import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import soundfile as sf
import numpy as np
import io
import tempfile
from openai import OpenAI
import os

# Load the Whisper model
model = whisper.load_model("base")
api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키를 가져옵니다.
client = OpenAI(api_key=api_key)
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

def transcribe_audio(audio):
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_wav_file:
        tmp_wav_file.write(audio["bytes"])
        tmp_wav_file.flush()

        # Transcribe audio using Whisper
        result = model.transcribe(tmp_wav_file.name)
        
        # Delete the temporary file
        os.remove(tmp_wav_file.name)
        
        return result['text']

def gpt_call(client, text):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Your only task is to translate English to Korean. Do not write anything other than the translation."},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content

def text_to_speech(client, text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        input=text
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
        response.stream_to_file(tmp_audio_file.name)
        tmp_file_name = tmp_audio_file.name
    
    return tmp_file_name

# Streamlit interface
st.title("Audio Recording and Transcription with Whisper")

st.write("Record your audio and transcribe it to text.")

audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", format="webm")

if audio:
    st.audio(audio['bytes'], format='audio/webm')
    transcription = transcribe_audio(audio)
    ts_text = gpt_call(client, transcription)
    st.write("Transcription:")
    st.write(transcription)
    st.write("Translation:")
    st.write(ts_text)

    # Convert translated text to speech
    tts_audio_data = text_to_speech(client, ts_text)

    # Store the TTS audio data in the session state
    st.session_state.tts_audio_data = tts_audio_data

# Automatically play the TTS audio if available
if 'tts_audio_data' in st.session_state:
    st.audio(st.session_state.tts_audio_data, format='audio/mp3')
    os.remove(st.session_state.tts_audio_data)
