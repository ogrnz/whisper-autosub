# Whisper autosub

This small experiments takes a video and uses [distil-whisper](https://github.com/huggingface/distil-whisper) to transcribe audio into text.

Under the hood, the script uses `FFMPEG` and `moviepy` to handle video & audio, `torch` and Hugging Face `transformers` to use the model and `gentle` to align text with the audio.

## Installation
```shell
git clone https://github.com/ogrnz/whisper-autosub.git
cd whisper-autosub
# Create and activate your venv
pip install -r requirements.txt
```

## Usage
Check out the script in `whisper-autosub/main.py` and simply provide the path to your video. For now, you need to manually align the text with the audio with [gentle](https://github.com/lowerquality/gentle). The script expects the **json** exported file.
