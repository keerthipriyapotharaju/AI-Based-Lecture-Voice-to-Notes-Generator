import streamlit as st
import whisper
import openai
from dotenv import load_dotenv
import os
load_dotenv()
import subprocess

# ---------- CONFIG ----------
openai.api_key = os.getenv("OPENAI_API_KEY")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- LOAD WHISPER ----------
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# ---------- FUNCTIONS ----------
def extract_audio(video_path):
    audio_path = video_path.replace(".mp4", ".mp3")
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "mp3", audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

from openai import OpenAI

def ai_response(prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Lecture Voice to Notes", layout="centered")

st.title("🎙️ Lecture Voice to Notes Generator")
st.write("Upload **audio or video lecture** and get notes, summary & quiz")

uploaded_file = st.file_uploader(
    "Upload Audio or Video",
    type=["mp3", "wav", "m4a", "mp4"]
)

if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    if uploaded_file.name.endswith(".mp4"):
        st.info("Extracting audio from video...")
        audio_path = extract_audio(file_path)
    else:
        audio_path = file_path

    if st.button("Generate Notes"):
        with st.spinner("Transcribing lecture..."):
            transcript = transcribe_audio(audio_path)

        st.subheader("📝 Transcription")
        st.text_area("Lecture Text", transcript, height=200)

        with st.spinner("Generating summary..."):
            summary = ai_response(
                f"Summarize the following lecture notes:\n{transcript}"
            )

        st.subheader("📌 Summary")
        st.write(summary)

        with st.spinner("Generating quiz..."):
            quiz = ai_response(
                f"Create 5 quiz questions from this lecture:\n{transcript}"
            )

        st.subheader("❓ Quiz")
        st.write(quiz)
