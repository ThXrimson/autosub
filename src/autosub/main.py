import logging
import os
import click
import torch
import whisper
import whisper.tokenizer

from .autosub import autosub

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "inputs",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help="Directory to save the transcriptions. Defaults to the directory of each file.",
)
@click.option(
    "-l",
    "--list-models",
    default=False,
    is_flag=True,
    help="List available Whisper models and exit.",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(whisper.available_models()),
    required=True,
    metavar="MODEL",
    help="Whisper model to use for transcription.",
)
@click.option(
    "--whisper-root",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=None,
    help="Directory to download Whisper models. Defaults to environment variable AUTOSUB_WHISPER_ROOT or ~/.cache/whisper.",
)
@click.option(
    "--gpu",
    is_flag=True,
    default=True,
    help="Use GPU for transcription if available.",
)
@click.option(
    "--task",
    type=click.Choice(["transcribe", "translate"]),
    default="transcribe",
    help="Task to perform: 'transcribe' for speech recognition or 'translate' for translation to English.",
)
@click.option(
    "--language",
    type=click.Choice(
        [*whisper.tokenizer.LANGUAGES]
        + [*whisper.tokenizer.LANGUAGES.values()]
        + ["auto"]
    ),
    default="auto",
    metavar="LANGUAGE",
    help="Language of the input audio (language code or name). Set to 'auto' for automatic detection.",
)
@click.option(
    "--initial-prompt",
    type=str,
    default=None,
    help="Initial prompt for the transcription. Useful for guiding the model on specific topics.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing transcription files.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
@click.option(
    "--embed",
    is_flag=True,
    default=False,
    help="Embed subtitles into the video files.",
)
def main(
    inputs,
    output_dir,
    list_models,
    model,
    whisper_root,
    gpu,
    task,
    language,
    initial_prompt,
    overwrite,
    verbose,
    embed,
):
    """Transcribe videos and audios with Whisper.

    INPUTS can be a list of video or audio files or directories containing them.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.disable(logging.NOTSET if verbose else logging.INFO)
    if list_models:
        click.echo("Available Whisper models:")
        for model in whisper.available_models():
            click.echo(f"  {model}")
        return
    if not inputs:
        raise click.UsageError("No input files or directories specified.")
    if not torch.cuda.is_available() and gpu:
        logging.warning(
            "GPU is not available. Using CPU for transcription. "
            "This may be slow for large files."
        )
        gpu = False
    if whisper_root is None:
        whisper_root = (
            click.get_app_dir("autosub", roaming=True)
            if "AUTOSUB_WHISPER_ROOT" not in os.environ
            else os.environ["AUTOSUB_WHISPER_ROOT"]
        )
    autosub(
        inputs,
        output_dir,
        model,
        whisper_root,
        gpu,
        task,
        language,
        initial_prompt,
        overwrite,
        verbose,
        embed,
    )


if __name__ == "__main__":
    main()
