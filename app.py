import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import whisper
import numpy as np
from pydub import AudioSegment
import queue
import av

# Whisper 모델 로드
model = whisper.load_model("base")

# WebRTC 설정
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 오디오 처리 클래스
class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
    
    def recv(self, frame):
        audio = frame.to_ndarray()
        audio_segment = AudioSegment(
            data=audio.tobytes(),
            sample_width=audio.dtype.itemsize,
            frame_rate=frame.sample_rate,
            channels=len(audio.shape),
        )
        self.audio_queue.put(audio_segment)
        return frame

# Streamlit 어플리케이션
st.title("Real-time Speech-to-Text with Whisper")

# WebRTC 스트리머
ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)

# 텍스트 출력 영역
text_output = st.empty()

def transcribe_audio(audio_segments):
    audio_bytes = b"".join(audio_segments)
    np_audio = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

    # Whisper 모델을 사용하여 음성을 텍스트로 변환
    result = model.transcribe(np_audio)
    return result["text"]

if ctx.state.playing:
    audio_processor = ctx.audio_processor
    if audio_processor:
        audio_segments = []
        while not audio_processor.audio_queue.empty():
            audio_segment = audio_processor.audio_queue.get()
            audio_segments.append(audio_segment.raw_data)
        
        if audio_segments:
            st.write(f"Collected {len(audio_segments)} audio segments")
            st.write("Transcribing audio...")

            # 비동기적 변환과 실시간 업데이트를 위해 텍스트 변환 및 출력
            text = transcribe_audio(audio_segments)
            text_output.markdown(f"**Transcribed Text:** {text}")

            st.write("Transcription complete")
        else:
            st.write("No audio data collected")
