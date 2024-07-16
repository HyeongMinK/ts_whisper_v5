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
    return whisper.load_model("small")

model = load_whisper_model()
api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키를 가져옵니다.
client = OpenAI(api_key=api_key)
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

def transcribe_audio(file_path):
    result = model.transcribe(file_path, language='ko')
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

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

if 'once_recording' not in st.session_state:
    st.session_state.once_recording = False

if 'temp_page' not in st.session_state:
    st.session_state.temp_page = 1

# 언어 선택 박스 (기본값을 영어로 설정)
selected_language = st.selectbox('Language', languages, index=1)

# Initialize session state lists
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []
if 'file_paths' not in st.session_state:
    st.session_state.file_paths = []
if 'ts_texts' not in st.session_state:
    st.session_state.ts_texts = []
if 'tts_audio_data' not in st.session_state:
    st.session_state.tts_audio_data = []

audio = mic_recorder(start_prompt="Start", stop_prompt="Stop", format="webm", callback=state_recode)

if st.session_state.is_recording == True:
    st.session_state.once_recording = True
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_wav_file:
        tmp_wav_file.write(audio["bytes"])
        tmp_wav_file.flush()
        st.session_state.file_path = tmp_wav_file.name
    transcription = transcribe_audio(st.session_state.file_path)
    ts_text = gpt_call(client, transcription, selected_language)

    # Convert translated text to speech
    tts_audio = text_to_speech(client, ts_text)

    # Append results to session state lists
    st.session_state.transcriptions.append(transcription)
    st.session_state.file_paths.append(st.session_state.file_path)
    st.session_state.ts_texts.append(ts_text)
    st.session_state.tts_audio_data.append(tts_audio)

    #temp_Page
    st.session_state.temp_page=len(st.session_state.tts_audio_data)

    st.session_state.is_recording = False

st.sidebar.title("Recordings")

if st.session_state.once_recording == True:
    # Custom CSS to highlight the selected button
    st.markdown("""
        <style>
        .highlighted-button {
            background-color: #FF4B4B !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar with numbered recordings
    for i in range(len(st.session_state.transcriptions)):
        button_class = "highlighted-button" if st.session_state.temp_page == i + 1 else ""
        if st.sidebar.button(f"Recording {i+1}", key=f"btn_{i+1}", on_click=lambda idx=i+1: st.session_state.update(temp_page=idx)):
            st.session_state.temp_page = i + 1

    for i in range(len(st.session_state.transcriptions)):
        if st.session_state.temp_page == i + 1:
            st.write(f"Transcription {i+1}:")
            st.write(st.session_state.transcriptions[i])
            st.audio(st.session_state.file_paths[i], format='audio/webm')

            st.write(f"Translation {i+1}:")
            st.write(st.session_state.ts_texts[i])
            st.audio(st.session_state.tts_audio_data[i], format='audio/mp3', autoplay=True)
