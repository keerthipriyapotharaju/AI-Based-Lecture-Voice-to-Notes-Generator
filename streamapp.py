import streamlit as st
import whisper
from openai import OpenAI
from dotenv import load_dotenv
import os
import subprocess
import tempfile

# ---------- LOAD ENV VARIABLES ----------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(page_title="Lecture Voice to Notes", layout="centered")
st.title("🎙️ Lecture Voice to Notes Generator")
st.write("Upload **audio or video lecture** and get notes, summary & quiz")

# ---------- LOAD WHISPER MODEL ----------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")  # small, medium, large for better accuracy

model = load_whisper_model()

# ---------- OPENAI CLIENT ----------
client = OpenAI(api_key=openai_api_key)

# ---------- FUNCTIONS ----------
def extract_audio(video_path):
    """
    Converts mp4 video to mp3 audio using ffmpeg.
    """
    audio_path = video_path.replace(".mp4", ".mp3")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "mp3", audio_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def transcribe_audio(audio_path):
    """
    Transcribes audio using Whisper model.
    """
    result = model.transcribe(audio_path)
    return result["text"]

def ai_response(prompt):
    """
    Generates AI response (summary or quiz) using OpenAI GPT-4o-mini.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload Audio or Video",
    type=["mp3", "wav", "m4a", "mp4"]
)

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("File uploaded successfully!")

    # Extract audio if video
    if uploaded_file.name.endswith(".mp4"):
        st.info("Extracting audio from video...")
        audio_path = extract_audio(file_path)
    else:
        audio_path = file_path

    # Button to generate notes
    if st.button("Generate Notes & Quiz"):
        with st.spinner("Transcribing lecture..."):
            transcript = transcribe_audio(audio_path)

        st.subheader("📝 Transcription")
        st.text_area("Lecture Text", transcript, height=200)

        with st.spinner("Generating summary..."):
            summary = ai_response(f"Summarize the following lecture notes:\n{transcript}")

        st.subheader("📌 Summary")
        st.write(summary)

        with st.spinner("Generating quiz..."):
            quiz = ai_response(f"Create 5 quiz questions from this lecture:\n{transcript}")

        st.subheader("❓ Quiz")
        st.write(quiz)
