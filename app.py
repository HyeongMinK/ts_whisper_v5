import streamlit as st
import whisper
import pyaudio
import numpy as np
import threading
import queue

# Whisper 모델 로드
model = whisper.load_model("base")

# PyAudio 설정
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

audio_queue = queue.Queue()
stop_event = threading.Event()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

def record_audio():
    st.write("Recording... Press 'Stop Recording' to stop.")
    stream.start_stream()

    frames = []
    while not stop_event.is_set():
        data = audio_queue.get()
        frames.append(data)

    stream.stop_stream()
    st.write("Recording stopped.")

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Whisper 모델을 사용하여 텍스트로 변환
    result = model.transcribe(audio_data, language='ko')
    st.session_state.transcribed_text = result["text"]

def start_recording():
    stop_event.clear()
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()
    return record_thread

# Streamlit 인터페이스 설정
st.title("Real-time Audio Recording with Whisper")

if st.button("Start Recording"):
    if 'record_thread' in st.session_state:
        if st.session_state.record_thread.is_alive():
            st.write("Recording is already in progress.")
        else:
            st.session_state.record_thread = start_recording()
    else:
        st.session_state.record_thread = start_recording()

if st.button("Stop Recording"):
    stop_event.set()
    if 'record_thread' in st.session_state:
        st.session_state.record_thread.join()
        del st.session_state.record_thread

# 녹음된 텍스트 표시
if 'transcribed_text' in st.session_state:
    st.write("You said: ", st.session_state.transcribed_text)

stream.close()
p.terminate()
