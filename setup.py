from setuptools import find_packages, setup

setup(name="jeanne",
    version="0.1",
    description="seq2seq in pytorch (minimum version) + best saint waifu",
    author="Andros Tjandra",
    author_email='andros.tjandra@gmail.com',
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    license="BSD",
    url="",
    packages=find_packages(),
    install_requires=['tamamo', 'librosa', 'tqdm', 'tabulate', 'asr-evaluation']);
