import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import soundfile as sf
import numpy as np
import io
import tempfile
from openai import OpenAI

# Load the Whisper model
model = whisper.load_model("base")
api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키를 가져옵니다.
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")


def transcribe_audio(audio):
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_wav_file:
        tmp_wav_file.write(audio["bytes"])
        tmp_wav_file.flush()

        # Transcribe audio using Whisper
        result = model.transcribe(tmp_wav_file.name)
        return result['text']
def gpt_call(text):
  client = OpenAI(api_key=api_key)
  completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "Your only task is to be a translator and translate English to Korean."},
    {"role": "user", "content": f"text = {text}"}
  ]
)
  return completion.choices[0].message.content

# Streamlit interface
st.title("Audio Recording and Transcription with Whisper")

st.write("Record your audio and transcribe it to text.")

audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", format="webm")

if audio:
    st.audio(audio['bytes'], format='audio/webm')
    transcription = transcribe_audio(audio)
    st.write("Transcription:")
    st.write(transcription)
    st.write("Translation:")
    st.write(gpt_call(transcription))
