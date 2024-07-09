import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import numpy as np
import whisper
import soundfile as sf
import queue

# Load the Whisper model
model = whisper.load_model("base")

# AudioProcessor class for handling audio stream
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []
        self.queue = queue.Queue()

    def recv_queued(self, frames):
        for frame in frames:
            audio = frame.to_ndarray()
            self.queue.put(audio)

    def get_audio_text(self):
        audio_list = []
        while not self.queue.empty():
            audio_list.append(self.queue.get())

        if len(audio_list) == 0:
            return "No audio data received"

        audio_data = np.concatenate(audio_list, axis=0).astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize audio
        with sf.SoundFile('temp.wav', mode='w', samplerate=16000, channels=1) as file:
            file.write(audio_data)

        result = model.transcribe('temp.wav')
        return result['text']

# Streamlit interface
st.title("Real-time Transcription with Whisper")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

if webrtc_ctx.audio_processor:
    if st.button("Get Transcription"):
        transcription = webrtc_ctx.audio_processor.get_audio_text()
        st.write("Transcription:")
        st.write(transcription)
