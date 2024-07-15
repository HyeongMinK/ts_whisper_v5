import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile
from openai import OpenAI
import os
import warnings

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load the Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()
api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키를 가져옵니다.
client = OpenAI(api_key=api_key)
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result['text']

def gpt_call(client, text, selected_language):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Your only task is to translate to {selected_language}. Do not write anything other than the translation."},
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
def state_recode():
    st.session_state.is_recording = True

# Streamlit interface
st.title("Streamlit Audio Translator")

st.write("Select the language of the translation result and click Start!")
st.text_area("Write your notes here:", height=200)
# 선택할 수 있는 언어 목록
languages = ['한국어', 'English', '中文', '日本語', 'Tiếng Việt', 'हिन्दी']

# Initialize session state for selected language
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'English'

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

# 언어 선택 박스 (기본값을 영어로 설정)
selected_language = st.selectbox('Language', languages, index=1)


audio = mic_recorder(start_prompt="Start", stop_prompt="Stop", format="webm", callback=state_recode)

if st.session_state.is_recording == True:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_wav_file:
        tmp_wav_file.write(audio["bytes"])
        tmp_wav_file.flush()
        file_path = tmp_wav_file.name

    st.audio(file_path, format='audio/webm')
    transcription = transcribe_audio(file_path)
    ts_text = gpt_call(client, transcription, selected_language)
    st.write("Transcription:")
    st.write(transcription)
    st.write("Translation:")
    st.write(ts_text)

    # Convert translated text to speech
    tts_audio_data = text_to_speech(client, ts_text)

    # Automatically play the TTS audio if available
    st.audio(tts_audio_data, format='audio/mp3', autoplay=True)

    # Delete temporary files
    os.remove(file_path)
    os.remove(tts_audio_data)
    st.session_state.is_recording == False

# Check if the selected language has changed
if selected_language != st.session_state.selected_language:
    st.session_state.selected_language = selected_language
