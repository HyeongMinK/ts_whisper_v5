import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import whisper
import numpy as np
import av
import threading

# Whisper 모델 로드
model = whisper.load_model("base")

# 녹음된 오디오 데이터를 저장할 버퍼
audio_buffer = []

# WebRTC 설정
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
)

# 오디오 프레임 처리 콜백
def audio_frame_callback(frame: av.AudioFrame):
    global audio_buffer
    audio = frame.to_ndarray()
    audio_buffer.append(audio)
    return av.AudioFrame.from_ndarray(audio, layout=frame.layout.name)

# 녹음 시작 및 중지 컨트롤
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    client_settings=WEBRTC_CLIENT_SETTINGS,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"video": False, "audio": True},
)

if st.button("Stop Recording"):
    webrtc_ctx.stop()
    if audio_buffer:
        # 오디오 버퍼를 numpy 배열로 변환
        audio_data = np.concatenate(audio_buffer, axis=1).flatten()

        # Whisper 모델을 사용하여 텍스트로 변환
        result = model.transcribe(audio_data, language='ko')
        st.write("You said: ", result["text"])

        # 오디오 버퍼 초기화
        audio_buffer = []
    else:
        st.write("No audio data recorded.")
