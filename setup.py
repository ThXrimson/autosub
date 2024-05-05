from setuptools import setup

setup(
    version="1.0",
    name="autosub",
    packages=['autosub'],
    author="Haoxiang Tian",
    install_requires=[
        'torch==2.3.0',
        'ffmpeg-python',
        'openai-whisper',
    ],
    description="Automatically generate and subtitles with Whisper.",
    entry_points={
        'console_scripts': ['autosub=autosub.__main__:main'],
    },
    include_package_data=True,
)