import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings, WebRtcMode
import av
import numpy as np
import whisper
import soundfile as sf
import queue

# Load the Whisper model
model = whisper.load_model("base")

# AudioProcessor class for handling audio stream
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = queue.Queue()
        self.frames = []

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

    def get_audio_text(self):
        audio_data = np.concatenate(self.frames, axis=0).astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize audio
        with sf.SoundFile('temp.wav', mode='w', samplerate=16000, channels=1) as file:
            file.write(audio_data)

        result = model.transcribe('temp.wav')
        return result['text']

# Streamlit interface
st.title("Real-time Transcription with Whisper")

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
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
