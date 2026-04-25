# whisper-transcribe — Speech Recognition

Local and API-based speech-to-text transcription using [OpenAI Whisper](https://github.com/openai/whisper). Supports offline transcription with locally-downloaded models.

## Capabilities

- **transcribe** — Transcribe an audio file to text
- **translate** — Translate non-English audio to English text
- **stream** — Real-time transcription from microphone input
- **batch** — Transcribe all audio files in a directory
- **models** — List and download available Whisper models
- **srt** — Generate SRT subtitle files from audio

## Setup

```bash
# Local Whisper (requires Python 3.8+)
pip install openai-whisper

# For microphone streaming
pip install sounddevice numpy

# For SRT generation
pip install openai-whisper
```

## Usage

```bash
# Transcribe an audio file
python3 whisper_transcribe.py transcribe audio.mp3

# Transcribe with a specific model
python3 whisper_transcribe.py transcribe audio.mp3 --model medium --language en

# Translate non-English audio to English
python3 whisper_transcribe.py translate recording.mp3

# Generate SRT subtitles
python3 whisper_transcribe.py srt video.mp4 --output subs.srt

# Batch transcribe a directory
python3 whisper_transcribe.py batch ./audio_files/ --output-dir ./transcripts/

# Real-time microphone transcription
python3 whisper_transcribe.py stream --model tiny

# List available models
python3 whisper_transcribe.py models
```

## Models

| Model    | Parameters | VRAM   | Speed    | Accuracy |
|----------|-----------|--------|----------|----------|
| tiny     | 39M       | ~1GB   | ~32x     | Good     |
| base     | 74M       | ~1GB   | ~16x     | Better   |
| small    | 244M      | ~2GB   | ~6x      | Great    |
| medium   | 769M      | ~5GB   | ~2x      | Excellent|
| large-v3 | 1550M     | ~10GB  | 1x       | Best     |

## Output

- Transcriptions saved to `transcripts/`
- Subtitles saved to `subtitles/`
- Models cached in `~/.cache/whisper/`
