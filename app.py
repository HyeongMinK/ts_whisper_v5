import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import whisper
import soundfile as sf
import numpy as np
import tempfile

# Load the Whisper model
model = whisper.load_model("base")

# Streamlit interface
st.title("Audio Recording and Transcription with Whisper")

st.write("Record your audio and transcribe it to text.")

# WebRTC Streamer for recording audio
webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    ),
    async_processing=True,
)

# Placeholder for the transcription
transcription_placeholder = st.empty()

if st.button("Stop Recording and Transcribe"):
    if webrtc_ctx.state.playing:
        webrtc_ctx.stop()
    
    # Retrieve audio frames
    audio_frames = webrtc_ctx.audio_processor.get_audio_frames()

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        sf.write(tmp_wav_file.name, np.concatenate(audio_frames), 16000)
        tmp_wav_file.flush()

        # Transcribe audio using Whisper
        transcription = model.transcribe(tmp_wav_file.name)
        transcription_text = transcription['text']
        
        # Display the transcription
        transcription_placeholder.text_area("Transcription", transcription_text)

