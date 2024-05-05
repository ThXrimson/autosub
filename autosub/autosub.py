import logging
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
import whisper

from .utils import (collect_video_paths, embed_subtitle, extract_audio,
                    get_output_dir, transcribe)


def autosub(inputs: List[str],
            output_dir: str,
            model_name: str,
            gpu: bool,
            task: str,
            language: str,
            simplified: bool,
            initial_prompt: str,
            verbose: bool,
            embed: bool):
    """Transcribe videos and embed subtitles into them.

    Args:
        inputs (List[str]):     Paths to video files (directories) to transcribe.
        output_dir (str):       Directory to save the outputs.
        model_name (str):       Names of the Whisper model to use.
        gpu (bool):             Whether to use the GPU for inference.
        task (str):             Whether to perform X->X speech recognition (`transcribe`) or X->English translation (`translate`).
        language (str):         What is the origin language of the video? If unset, it is detected automatically.
        simplified (bool):      Whether to output the simplified Chinese.
        initial_prompt (str):   Initial prompt for the transcription.
        verbose (bool):         Whether to print out the progress and debug messages.
        embed (bool):           Whether to embed the subtitles into the video.
    """

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    if not inputs:
        os.makedirs(output_dir, exist_ok=True)

    logging.info('Loading Whisper model...')
    model_args = {
        'task': task,
        'verbose': verbose,
    }

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        language = 'en'
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        model_args['language'] = language

    model_args['initial_prompt'] = initial_prompt

    if language in ('zh', 'Chinese') and simplified:
        model_args['initial_prompt'] += '以下是普通话句子'

    model = whisper.load_model(
        model_name, device=torch.device('cuda' if gpu else 'cpu'))
    transcribe_func = partial(model.transcribe, **model_args)

    # Get input paths and corresponding output srt paths
    input_paths_to_srt_paths: Dict[Path, Path] = {}
    for input_path in inputs:
        input_path = Path(input_path)
        if input_path.is_dir():
            input_paths_to_srt_paths.update({
                video_path: Path(get_output_dir(video_path, output_dir)) /
                video_path.with_suffix('.srt').name
                for video_path in collect_video_paths(input_path)
            })
        else:
            input_paths_to_srt_paths[input_path] = Path(get_output_dir(
                input_path, output_dir)) / input_path.with_suffix('.srt').name

    logging.info('Transcribing videos...')
    for video_path, srt_path in input_paths_to_srt_paths.items():
        audio = extract_audio(video_path)
        transcribe(audio, srt_path, transcribe_func)

    if embed:
        logging.info('Embedding subtitles into the video...')
        for video_path, srt_path in input_paths_to_srt_paths.items():
            output_path = srt_path.with_suffix(video_path.suffix)
            embed_subtitle(video_path, srt_path, output_path)
