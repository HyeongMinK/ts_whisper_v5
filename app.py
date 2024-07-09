import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import openai
import io
import os

# Load the Whisper model
model = whisper.load_model("base")
api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키를 가져옵니다.
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
openai.api_key = api_key

def transcribe_audio(audio):
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_wav_file:
        tmp_wav_file.write(audio["bytes"])
        tmp_wav_file.flush()

        # Transcribe audio using Whisper
        result = model.transcribe(tmp_wav_file.name)
        return result['text']

def gpt_call(client, text):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Your only task is to translate English to Korean. Do not write anything other than the translation."},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message['content']

def text_to_speech(client, text):
    response = openai.Audio.create(
        model="whisper-1",
        voice="echo",
        input=text
    )
    return io.BytesIO(response['audio_data'])

# Streamlit interface
st.title("Audio Recording and Transcription with Whisper")

st.write("Record your audio and transcribe it to text.")

audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", format="webm")

if audio:
    st.audio(audio['bytes'], format='audio/webm')
    transcription = transcribe_audio(audio)
    ts_text = gpt_call(openai, transcription)
    st.write("Transcription:")
    st.write(transcription)
    st.write("Translation:")
    st.write(ts_text)

    # Convert translated text to speech
    tts_audio_data = text_to_speech(openai, ts_text)
    
    # Play the TTS audio
    st.audio(tts_audio_data, format='audio/wav')
