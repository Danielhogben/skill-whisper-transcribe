#!/usr/bin/env python3
"""whisper-transcribe — Speech recognition powered by OpenAI Whisper."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
C = "\033[96m"
W = "\033[0m"
BOLD = "\033[1m"

SKILL_DIR = Path(__file__).parent
CONFIG_FILE = SKILL_DIR / "config.json"
TRANSCRIPTS_DIR = SKILL_DIR / "transcripts"
SUBTITLES_DIR = SKILL_DIR / "subtitles"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".mp4", ".webm", ".opus"}


def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def check_whisper():
    try:
        import whisper
        return True
    except ImportError:
        print(f"{R}openai-whisper not installed.{W}")
        print(f"  Install: {C}pip install openai-whisper{W}")
        print(f"  For microphone streaming: {C}pip install sounddevice numpy{W}")
        return False


def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def load_model(model_name):
    import whisper
    print(f"{C}Loading model:{W} {model_name}")
    print(f"{Y}(First download may take a while...){W}")
    model = whisper.load_model(model_name)
    print(f"{G}Model loaded.{W}")
    return model


async def cmd_transcribe(args):
    if not check_whisper():
        sys.exit(1)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"{R}File not found:{W} {audio_path}")
        sys.exit(1)

    cfg = load_config()
    model_name = args.model or cfg.get("model", "base")
    language = args.language or cfg.get("language", None)

    model = load_model(model_name)

    print(f"\n{C}Transcribing:{W} {audio_path}")
    if language:
        print(f"{C}Language:{W} {language}")

    transcribe_opts = {"verbose": args.verbose}
    if language:
        transcribe_opts["language"] = language

    result = model.transcribe(str(audio_path), **transcribe_opts)

    text = result["text"]
    detected_lang = result.get("language", "unknown")

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    out_name = audio_path.stem + "_transcript.txt"
    out_path = TRANSCRIPTS_DIR / out_name
    out_path.write_text(text)

    # Also save full result as JSON
    json_out = TRANSCRIPTS_DIR / (audio_path.stem + "_result.json")
    json_data = {
        "file": str(audio_path),
        "model": model_name,
        "language": detected_lang,
        "text": text,
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ],
        "timestamp": datetime.now().isoformat(),
    }
    json_out.write_text(json.dumps(json_data, indent=2))

    print(f"\n{G}Transcription complete:{W}")
    print(f"{Y}Language:{W} {detected_lang}")
    print(f"{Y}Segments:{W} {len(result.get('segments', []))}")
    print(f"\n{text[:500]}{'...' if len(text) > 500 else ''}")
    print(f"\n{G}Saved:{W} {out_path}")
    print(f"{G}JSON:{W} {json_out}")


async def cmd_translate(args):
    if not check_whisper():
        sys.exit(1)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"{R}File not found:{W} {audio_path}")
        sys.exit(1)

    cfg = load_config()
    model_name = args.model or cfg.get("model", "base")

    model = load_model(model_name)

    print(f"\n{C}Translating to English:{W} {audio_path}")

    result = model.transcribe(str(audio_path), task="translate")

    text = result["text"]
    detected_lang = result.get("language", "unknown")

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    out_name = audio_path.stem + "_translated.txt"
    out_path = TRANSCRIPTS_DIR / out_name
    out_path.write_text(text)

    print(f"\n{G}Translation complete:{W}")
    print(f"{Y}Source language:{W} {detected_lang}")
    print(f"\n{text[:500]}{'...' if len(text) > 500 else ''}")
    print(f"\n{G}Saved:{W} {out_path}")


async def cmd_srt(args):
    if not check_whisper():
        sys.exit(1)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"{R}File not found:{W} {audio_path}")
        sys.exit(1)

    cfg = load_config()
    model_name = args.model or cfg.get("model", "base")
    language = args.language

    model = load_model(model_name)

    print(f"\n{C}Generating SRT:{W} {audio_path}")

    transcribe_opts = {}
    if language:
        transcribe_opts["language"] = language

    result = model.transcribe(str(audio_path), **transcribe_opts)

    segments = result.get("segments", [])
    if not segments:
        print(f"{Y}No segments found — cannot generate SRT.{W}")
        return

    SUBTITLES_DIR.mkdir(exist_ok=True)
    srt_name = args.output or (audio_path.stem + ".srt")
    srt_path = SUBTITLES_DIR / srt_name

    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")

    srt_path.write_text("\n".join(srt_lines))

    print(f"{G}SRT file created:{W} {srt_path}")
    print(f"{Y}Segments:{W} {len(segments)}")

    # Show preview
    preview = "\n".join(srt_lines[:12])
    print(f"\n{Y}Preview:{W}\n{preview}")


async def cmd_batch(args):
    if not check_whisper():
        sys.exit(1)

    audio_dir = Path(args.directory)
    if not audio_dir.is_dir():
        print(f"{R}Directory not found:{W} {audio_dir}")
        sys.exit(1)

    audio_files = sorted(
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(f"{Y}No audio files found in {audio_dir}{W}")
        return

    cfg = load_config()
    model_name = args.model or cfg.get("model", "base")
    output_dir = Path(args.output_dir) if args.output_dir else TRANSCRIPTS_DIR

    print(f"{C}Found {len(audio_files)} audio files{W}")
    print(f"{C}Model:{W} {model_name}")
    print(f"{C}Output:{W} {output_dir}\n")

    model = load_model(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{BOLD}[{i}/{len(audio_files)}] {audio_file.name}{W}")

        try:
            result = model.transcribe(str(audio_file))
            text = result["text"]
            lang = result.get("language", "unknown")

            out_file = output_dir / (audio_file.stem + "_transcript.txt")
            out_file.write_text(text)

            results.append({
                "file": str(audio_file),
                "language": lang,
                "text": text[:200],
                "output": str(out_file),
                "status": "ok",
            })
            print(f"  {G}OK{W} ({lang}): {text[:80]}...")

        except Exception as e:
            results.append({
                "file": str(audio_file),
                "status": "error",
                "error": str(e),
            })
            print(f"  {R}ERROR:{W} {e}")

    summary = {
        "total": len(audio_files),
        "success": sum(1 for r in results if r["status"] == "ok"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    summary_file = output_dir / "batch_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print(f"\n{G}Batch complete:{W} {summary['success']}/{summary['total']} succeeded")
    print(f"{G}Summary:{W} {summary_file}")


async def cmd_stream(args):
    if not check_whisper():
        sys.exit(1)

    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print(f"{R}sounddevice/numpy not installed.{W}")
        print(f"  Install: {C}pip install sounddevice numpy{W}")
        sys.exit(1)

    cfg = load_config()
    model_name = args.model or cfg.get("model", "tiny")
    sample_rate = 16000
    chunk_duration = args.chunk_seconds

    model = load_model(model_name)

    print(f"\n{C}Real-time transcription started{W}")
    print(f"{C}Model:{W} {model_name}")
    print(f"{C}Chunk size:{W} {chunk_duration}s")
    print(f"{Y}Press Ctrl+C to stop{W}\n")

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    transcript_lines = []

    try:
        while True:
            print(f"{C}Listening...{W}", end="\r")
            audio = sd.rec(
                int(chunk_duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            audio_data = audio.flatten()

            # Skip silence
            if np.max(np.abs(audio_data)) < 0.01:
                continue

            result = model.transcribe(audio_data, fp16=False)
            text = result["text"].strip()

            if text:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"  {Y}[{timestamp}]{W} {text}")
                transcript_lines.append(f"[{timestamp}] {text}")

    except KeyboardInterrupt:
        print(f"\n\n{G}Transcription stopped.{W}")

        if transcript_lines:
            out_path = TRANSCRIPTS_DIR / f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            out_path.write_text("\n".join(transcript_lines))
            print(f"{G}Saved:{W} {out_path}")


async def cmd_models(args):
    if args.list or not args.download:
        print(f"{BOLD}Available Whisper models:{W}\n")
        models = [
            ("tiny", "39M", "~1GB", "~32x", "Good for quick drafts"),
            ("base", "74M", "~1GB", "~16x", "Good balance of speed/quality"),
            ("small", "244M", "~2GB", "~6x", "Great quality, moderate speed"),
            ("medium", "769M", "~5GB", "~2x", "Excellent quality"),
            ("large-v3", "1550M", "~10GB", "1x", "Best quality, slowest"),
            ("turbo", "809M", "~6GB", "~8x", "Fast + high quality (recommended)"),
        ]
        print(f"  {'Model':<12} {'Params':<8} {'VRAM':<8} {'Speed':<8} {'Notes'}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*30}")
        for name, params, vram, speed, notes in models:
            print(f"  {name:<12} {params:<8} {vram:<8} {speed:<8} {notes}")

        # Check which are downloaded
        cache_dir = Path.home() / ".cache" / "whisper"
        if cache_dir.exists():
            downloaded = {f.stem.replace(".pt", "") for f in cache_dir.glob("*.pt")}
            if downloaded:
                print(f"\n  {G}Downloaded:{W} {', '.join(sorted(downloaded))}")

    if args.download:
        if not check_whisper():
            sys.exit(1)
        model_name = args.download
        print(f"{C}Downloading model: {model_name}{W}")
        model = load_model(model_name)
        print(f"{G}Model {model_name} ready.{W}")


async def cmd_config(args):
    cfg = load_config()

    if args.model:
        cfg["model"] = args.model
        print(f"{G}Default model set to {args.model}.{W}")

    if args.language:
        cfg["language"] = args.language
        print(f"{G}Default language set to {args.language}.{W}")

    if args.model or args.language:
        save_config(cfg)

    if not any([args.model, args.language]):
        print(f"{BOLD}Current configuration:{W}")
        if cfg:
            for k, v in cfg.items():
                print(f"  {Y}{k}:{W} {v}")
        else:
            print(f"  {Y}(empty — using defaults){W}")


async def main():
    parser = argparse.ArgumentParser(
        description="Speech recognition with OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # transcribe
    p = sub.add_parser("transcribe", help="Transcribe audio file to text")
    p.add_argument("audio", help="Path to audio file")
    p.add_argument("--model", "-m", help="Model name (tiny/base/small/medium/large-v3/turbo)")
    p.add_argument("--language", "-l", help="Language code (en, es, fr, etc.)")
    p.add_argument("--verbose", "-v", action="store_true")

    # translate
    p = sub.add_parser("translate", help="Translate audio to English text")
    p.add_argument("audio", help="Path to audio file")
    p.add_argument("--model", "-m", help="Model name")

    # srt
    p = sub.add_parser("srt", help="Generate SRT subtitles from audio")
    p.add_argument("audio", help="Path to audio/video file")
    p.add_argument("--output", "-o", help="Output SRT filename")
    p.add_argument("--model", "-m", help="Model name")
    p.add_argument("--language", "-l", help="Language code")

    # batch
    p = sub.add_parser("batch", help="Batch transcribe a directory")
    p.add_argument("directory", help="Directory with audio files")
    p.add_argument("--output-dir", "-o", help="Output directory")
    p.add_argument("--model", "-m", help="Model name")

    # stream
    p = sub.add_parser("stream", help="Real-time microphone transcription")
    p.add_argument("--model", "-m", default="tiny", help="Model name")
    p.add_argument("--chunk-seconds", type=float, default=5.0, help="Chunk duration")

    # models
    p = sub.add_parser("models", help="List/download available models")
    p.add_argument("--list", action="store_true", default=True, help="List models")
    p.add_argument("--download", "-d", help="Download a specific model")

    # config
    p = sub.add_parser("config", help="Configure defaults")
    p.add_argument("--model", "-m", help="Set default model")
    p.add_argument("--language", "-l", help="Set default language")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        "transcribe": cmd_transcribe,
        "translate": cmd_translate,
        "srt": cmd_srt,
        "batch": cmd_batch,
        "stream": cmd_stream,
        "models": cmd_models,
        "config": cmd_config,
    }
    await cmds[args.command](args)


if __name__ == "__main__":
    asyncio.run(main())
