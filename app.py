APP.PY

import streamlit as st
import streamlit_extras as ste
import os
import whisper
from transformers import pipeline
from moviepy import VideoFileClip
import yt_dlp
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.let_it_rain import rain
from streamlit_extras.stylable_container import stylable_container
from streamlit_lottie import st_lottie
import json
import textwrap

# Setup paths
os.environ["PATH"] += os.pathsep + r"C:\Users\PC\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

# Function to extract audio from local video
def extract_audio(video_path, audio_path="audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    return audio_path

# Function to download YouTube audio
def download_youtube_audio(youtube_url, output_path="youtube_audio"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path + ".mp3"

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# Function to summarize using HuggingFace transformers
def summarize_text(text, chunk_size=900):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    chunks = textwrap.wrap(text, chunk_size)
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=60, min_length=10, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return "\n".join(summaries)

# Load Lottie animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Lottie animation: techy header (you can replace with your own Lottie JSON file)
tech_anim = load_lottiefile("animations/tech.json") if os.path.exists("animations/tech.json") else None

# App UI/UX Begins
st.set_page_config(page_title="Smart Video Summarizer", layout="centered", page_icon="🎧")
rain(emoji="✨📸", font_size=25, falling_speed=3, animation_length="infinite")
if tech_anim:
    st_lottie(tech_anim, speed=1, height=250, key="header_anim")

st.markdown("<h1 style='text-align: center; color: #4FC3F7;'>🎥 AI-Powered Video Summarizer</h1>", unsafe_allow_html=True)
st.markdown("##### <center>Summarize YouTube or Local Videos in Seconds using AI.</center>", unsafe_allow_html=True)

add_vertical_space(2)
option = st.radio("Select Input Type", ("📁 Upload Local Video", "🔗 YouTube Video URL"))

add_vertical_space(1)

if option == "📁 Upload Local Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🎵 Extracting audio..."):
            audio_file = extract_audio("uploaded_video.mp4")

        with st.spinner("🧠 Transcribing with Whisper..."):
            transcript = transcribe_audio(audio_file)

        with st.spinner("📝 Summarizing with BART model..."):
            summary = summarize_text(transcript)

        st.success("✅ Done!")
        st.subheader("🔎 Summary:")
        st.write(summary)

elif option == "🔗 YouTube Video URL":
    youtube_url = st.text_input("Paste the YouTube video link below")

    if st.button("🔍 Analyze"):
        if youtube_url:
            with st.spinner("🎵 Downloading audio..."):
                audio_file = download_youtube_audio(youtube_url, "youtube_audio")

            with st.spinner("🧠 Transcribing with Whisper..."):
                transcript = transcribe_audio(audio_file)

            with st.spinner("📝 Summarizing with BART model..."):
                summary = summarize_text(transcript)

            st.success("✅ Analysis Complete!")
            st.subheader("🔎 Summary:")
            st.write(summary)
        else:
            st.warning("⚠️ Please enter a valid YouTube URL.")

# Footer
add_vertical_space(3)
st.markdown("---")
st.markdown("<center><small>Built with 💙 using Whisper, HuggingFace, and Streamlit</small></center>", unsafe_allow_html=True)
