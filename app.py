import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import av
import numpy as np
import whisper

# Load the Whisper model
model = whisper.load_model("base")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.audio_buffer.extend(audio)
        return frame

    def get_audio_text(self):
        audio_array = np.concatenate(self.audio_buffer, axis=0).flatten().astype(np.float32)
        audio_array = audio_array / np.max(np.abs(audio_array))  # Normalize audio
        result = model.transcribe(audio_array, fp16=False)
        transcription = result['text']
        return transcription

# Streamlit interface
st.title("Real-time Speech to Text with Whisper")

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode="sendrecv",
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    ),
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

if webrtc_ctx.audio_processor:
    if st.button("Get Transcription"):
        transcription = webrtc_ctx.audio_processor.get_audio_text()
        st.write("Transcription:")
        st.write(transcription)
