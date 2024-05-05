import argparse

import whisper

from .autosub import autosub


def main():

    parser = argparse.ArgumentParser(description='Transcribe videos')

    general_group = parser.add_argument_group('Helper arguments')
    general_group.add_argument('--list-models', action='store_true',
                        default=False, help='list available Whisper models and exit')

    io_group = parser.add_argument_group('IO arguments')
    io_group.add_argument('-i', '--inputs', nargs='+', type=str,
                        help='paths to video files (directories) to transcribe')
    io_group.add_argument('-o', '--output-dir', type=str,
                        help='directory to save the outputs')

    whisper_group = parser.add_argument_group('Whisper arguments')
    whisper_group.add_argument('-m', '--model', type=str, default='small', metavar='MODEL',
                        choices=whisper.available_models(), help='names of the Whisper model to use')
    whisper_group.add_argument('-g', '--gpu', action='store_true', default=False,
                        help='whether to use the GPU for inference')

    transcribe_group = parser.add_argument_group('Transcription arguments')
    transcribe_group.add_argument('-t', '--task', type=str, default='transcribe', choices=[
                        'transcribe', 'translate'], help='whether to perform X->X speech recognition (\'transcribe\') or X->English translation (\'translate\')')
    transcribe_group.add_argument('-l', '--language', type=str, default='auto',
                        help='What is the origin language of the video? If unset, it is detected automatically.')
    transcribe_group.add_argument('-s', '--simplified', action='store_true',
                        default=False, help='whether to output the simplified Chinese')
    transcribe_group.add_argument('-p', '--initial-prompt', type=str, default='',
                        help='initial prompt for the transcription')
    transcribe_group.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='whether to print out the progress and debug messages')

    subtitle_group = parser.add_argument_group('Subtitle arguments')
    subtitle_group.add_argument('--embed', '-e', action='store_true', default=False,
                        help='whether to embed the subtitles into the video')

    opts = parser.parse_args()
    if opts.list_models:
        print('Available Whisper models:')
        for model in whisper.available_models():
            print(f'  {model}')
        return

    inputs = opts.inputs
    output_dir = opts.output_dir
    model = opts.model
    gpu = opts.gpu
    task = opts.task
    verbose = opts.verbose
    language = opts.language
    simplified = opts.simplified
    initial_prompt = opts.initial_prompt
    embed = opts.embed

    autosub(inputs, output_dir, model, gpu, task,
            language, simplified, initial_prompt, verbose, embed)


if __name__ == '__main__':
    main()
