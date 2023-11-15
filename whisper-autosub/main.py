import json
import os

from moviepy.config import change_settings
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import torch

MODEL_ID = "distil-whisper/distil-medium.en"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if torch.backends.mps.is_available() else torch.float32


def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    audio.write_audiofile(audio_path)
    return audio_path


def transcribe_audio(audio_path):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        torch_dtype=DTYPE,
        device=DEVICE,
    )
    transcription = pipe(audio_path)
    return transcription["text"]


def add_subtitles_to_video(video_path, words):
    video = VideoFileClip(video_path)
    clips = [video]

    last_end = 0
    for word in words:
        # Generate a TextClip for each piece of transcription
        # print(word["word"])
        start_time = word.get("start", last_end + 0.01)
        end_time = word.get("end", start_time + 0.3)

        txt_clip = TextClip(word["word"], fontsize=24, color="white")
        txt_clip = (
            txt_clip.set_position("bottom")
            .set_duration(end_time - start_time)
            .set_start(start_time)
        )
        clips.append(txt_clip)
        last_end = end_time

    # Overlay the TextClips on the video
    final_video = CompositeVideoClip(clips)
    final_video_path = video_path.rsplit(".", 1)[0] + "_subtitled.mp4"
    final_video.write_videofile(final_video_path, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
    change_settings({"FFMPEG_BINARY": ffmpeg_path})

    video_path = "./data/eth-phd-1.mov"
    # audio_path = extract_audio(video_path)
    audio_path = "./data/eth-phd-1.wav"
    # transcription = transcribe_audio(audio_path)
    # print(transcription)

    # Align with gentle

    with open("./data/align.json", "r") as f:
        script = json.load(f)

    add_subtitles_to_video(video_path, script["words"])

    a = 0
