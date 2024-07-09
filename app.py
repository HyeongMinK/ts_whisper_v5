import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch

# Streamlit configuration
st.title("Real-time Speech to Text with Whisper")
st.write("Press the button and start speaking")

# Initialize variables
q = queue.Queue()
recorder = None
recording = False

# Load pre-trained model and tokenizer from Hugging Face
model_name = "facebook/wav2vec2-large-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Function to record audio
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

def start_recording():
    global recorder
    recorder = sd.InputStream(samplerate=16000, channels=1, callback=audio_callback)
    recorder.start()

def stop_recording():
    global recorder
    recorder.stop()

def audio_to_text(audio):
    input_values = tokenizer(audio, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

def main():
    global recording
    if st.button("Start Recording"):
        recording = True
        start_recording()
        st.write("Recording...")

    if st.button("Stop Recording"):
        recording = False
        stop_recording()
        st.write("Recording stopped.")

    if recording:
        st.write("Listening...")
        audio_frames = []
        while not q.empty():
            audio_frames.append(q.get())

        if audio_frames:
            audio_data = np.concatenate(audio_frames)
            transcription = audio_to_text(audio_data)
            st.write("Transcription:")
            st.write(transcription)

if __name__ == "__main__":
    main()
