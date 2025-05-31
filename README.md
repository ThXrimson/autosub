# autosub

Automatically generate subtitles for videos and audios using OpenAI Whisper.

## Features

- Transcribe audio and video files to subtitles (SRT)
- Supports batch processing of files and directories
- Supports multiple Whisper models and languages
- Optionally embed subtitles into video files
- GPU acceleration (if available)
- CLI interface with flexible options

## Installation

Install the dependencies (requires Python 3.13+):

```sh
pip install .
```

Or use your preferred tool (e.g., `uv`, `pip`, `hatch`).

## Usage

```sh
autosub [OPTIONS] INPUTS...
```

**Examples:**

Transcribe a single video:

```sh
autosub -m base your_video.mp4
```

Transcribe all videos in a directory and embed subtitles:

```sh
autosub -m small --embed /path/to/videos/
```

Specify output directory:

```sh
autosub -m medium -o ./subs/ video1.mp4 video2.mp4
```

List available models:

```sh
autosub --list-models
```

## Options

- `-o, --output-dir PATH` Directory to save the transcriptions.
- `-l, --list-models` List available Whisper models and exit.
- `-m, --model [MODEL]` Whisper model to use for transcription. **(required)**
- `--whisper-root PATH` Directory to download Whisper models.
- `--gpu / --no-gpu` Use GPU for transcription if available. (default: enabled)
- `--task [transcribe|translate]` Task to perform: transcribe or translate. (default: transcribe)
- `--language [LANGUAGE|auto]` Language of the input audio. (default: auto)
- `--initial-prompt TEXT` Initial prompt for the transcription.
- `--overwrite` Overwrite existing transcription files.
- `-v, --verbose` Enable verbose output.
- `--embed` Embed subtitles into the video files.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
