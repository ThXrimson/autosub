import mimetypes
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterator, List, TextIO, Union

import ffmpeg
import whisper
from numpy.typing import NDArray

StrOrPath = Union[str, Path]


def is_video(path: StrOrPath) -> bool:
    type, _ = mimetypes.guess_type(path)
    return type is not None and type.startswith("video/")


def is_audio(path: StrOrPath) -> bool:
    type, _ = mimetypes.guess_type(path)
    return type is not None and type.startswith("audio/")


def collect_video_paths(dir: StrOrPath) -> List[Path]:
    """This function will not recursively search for videos."""
    return [path for path in Path(dir).iterdir() if is_video(path)]


def extract_audio(path: StrOrPath) -> NDArray:
    with tempfile.TemporaryDirectory() as dir:
        if is_audio(path):
            audio = whisper.load_audio(Path(path).as_posix())
        else:
            output_path = Path(dir) / Path(path).name
            ffmpeg.input(str(path)).output(
                output_path.as_posix(), acodec="pcm_s16le", ac=1, ar="16k"
            ).run(quiet=True, overwrite_output=True)
            audio = whisper.load_audio(output_path.as_posix())
    return audio


def transcribe(
    audio: NDArray, output_path: StrOrPath, transcribe_func: Callable[[NDArray], Dict]
):
    result = transcribe_func(audio)
    with open(output_path, "w", encoding="utf-8") as srt:
        write_srt(result["segments"], file=srt)


def format_timestamp(seconds: float, always_include_hours: bool = False) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(transcript: Iterator[dict], file: TextIO):
    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            + f"{format_timestamp(segment['start'], always_include_hours=True)} --> "
            + f"{format_timestamp(segment['end'], always_include_hours=True)}\n"
            + f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def embed_subtitle(video_path: StrOrPath, srt_path: StrOrPath, output_path: StrOrPath):
    video = ffmpeg.input(str(video_path))
    audio = video.audio
    ffmpeg.concat(
        video.filter(
            "subtitles",
            str(srt_path),
            force_style="OutlineColour=&H40000000,BorderStyle=3",
        ),
        audio,
        v=1,
        a=1,
    ).output(str(output_path)).run(quiet=True, overwrite_output=True)


def get_output_dir(input_path: StrOrPath, output_dir: StrOrPath) -> Path:
    return Path(output_dir) if output_dir is not None else Path(input_path).parent
