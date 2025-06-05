import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict

import torch
import whisper
from tqdm.auto import tqdm

from .utils import (
    collect_video_paths,
    embed_subtitle,
    extract_audio,
    get_output_dir,
    is_audio,
    is_video,
    transcribe,
)


def autosub(
    inputs: tuple[str],
    recursive: bool,
    output_dir: str,
    model_name: str,
    whisper_root: str,
    cpu: bool,
    task: str,
    language: str,
    initial_prompt: str,
    overwrite: bool,
    verbose: bool,
    embed: bool,
):
    """Transcribe videos and embed subtitles into them.

    Args:
        inputs (tuple[str]):    Paths to video or audio files (directories) to transcribe.
        output_dir (str):       Directory to save the outputs.
        model_name (str):       Names of the Whisper model to use.
        download_root (str):    Directory to download Whisper models. Defaults to ~/.cache/whisper.
        cpu (bool):             Whether to use the CPU for inference.
        task (str):             Whether to perform X->X speech recognition (`transcribe`) or X->English translation (`translate`).
        language (str):         What is the origin language of the video? If unset, it is detected automatically.
        simplified (bool):      Whether to output the simplified Chinese.
        initial_prompt (str):   Initial prompt for the transcription.
        overwrite (bool):       Whether to overwrite the existing files.
        verbose (bool):         Whether to print out the progress and debug messages.
        embed (bool):           Whether to embed the subtitles into the video.
    """

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    logging.info("Loading Whisper model...")
    model_args = {
        "task": task,
        "verbose": verbose,
        "initial_prompt": initial_prompt,
    }

    if model_name.endswith(".en"):
        logging.warning(
            f"{model_name} is an English-only model, forcing English detection."
        )
        language = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        model_args["language"] = language

    model = whisper.load_model(
        model_name,
        device=torch.device("cpu" if cpu else "cuda"),
        download_root=whisper_root,
    )
    transcribe_func = partial(model.transcribe, **model_args)

    # Get input paths and corresponding output srt paths
    input_paths_to_srt_paths: Dict[Path, Path] = {}
    for input_path in inputs:
        input_path = Path(input_path)
        if input_path.is_dir():
            input_paths_to_srt_paths.update(
                {
                    video_path: get_output_dir(video_path, output_dir)
                    / video_path.with_suffix(".srt").name
                    for video_path in collect_video_paths(input_path, recursive)
                }
            )
        else:
            if not is_video(input_path) and not is_audio(input_path):
                continue
            input_paths_to_srt_paths[input_path] = (
                get_output_dir(input_path, output_dir)
                / input_path.with_suffix(".srt").name
            )

    logging.info("Transcribing videos...")
    for video_path, srt_path in input_paths_to_srt_paths.items():
        if not overwrite and srt_path.exists():
            logging.info(
                "Skipping video %s as the output already exists.", video_path.as_posix()
            )
            continue
        logging.info("Extracting audio from video %s...", video_path.as_posix())
        audio = extract_audio(video_path)
        logging.info("Transcribing video %s...", video_path.as_posix())
        transcribe(audio, srt_path, transcribe_func)

    if embed:
        logging.info("Embedding subtitles into the video...")
        for video_path, srt_path in tqdm(
            input_paths_to_srt_paths.items(), desc="Embedding subtitles"
        ):
            output_path = srt_path.with_stem(srt_path.stem + "_transcript").with_suffix(
                video_path.suffix
            )
            if not overwrite and output_path.exists():
                logging.info(
                    "Skipping video %s as the output already exists.",
                    video_path.as_posix(),
                )
                continue
            embed_subtitle(video_path, srt_path, output_path)
