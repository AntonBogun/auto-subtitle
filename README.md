# Automatic subtitles in your videos

This repository is a fix of a [fork](https://github.com/Irvingouj/auto-subtitle) that is a fix of the original [auto-subtitle](https://github.com/m1guelpf/auto-subtitle) repository. 

The original repository did not handle filenames correctly so the fork fixed that using pathlib.
Then, the fork did not handle quoting correctly meaning the ffmpeg command that burns the subtitles into the video would fail if the filename had single quotes or similar.

This fork has the code to correctly escape the filename, but that code is commented out in favor of adding the subtitles as a subtitle track instead of burning them into the video. It also enables usage of NVIDIA GPU acceleration if available.

## Installation

The mandatory prerequisite is [`ffmpeg`](https://ffmpeg.org/), which is available from most package managers:

```bash
# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

The optional prerequisite is [`torch.cu***`](https://pytorch.org/get-started/locally/) which is available from the PyTorch website. Select the appropriate version for your system.

Note: you may need to uninstall the modules listed in the generated command for it to download the correct cuda-enabled version.

---

It is possible to install this fork like the original repository via:

    pip install git+https://github.com/AntonBogun/auto-subtitle.git

But, in case anything breaks and you have to edit the code, you can install it via:

    git clone https://github.com/AntonBogun/auto-subtitle
    cd auto-subtitle
    pip install -e .

## Usage

The following command will generate a `subtitled/video.mp4` file contained the input video with overlayed subtitles.

    auto_subtitle /path/to/video.mp4 -o subtitled/

The default setting (which selects the `small` model) works well for transcribing English. You can optionally use a bigger model for better results (especially with other languages). The available models are `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`.

    auto_subtitle /path/to/video.mp4 --model medium

Adding `--task translate` will translate the subtitles into English:

    auto_subtitle /path/to/video.mp4 --task translate

Run the following to view all available options:

    auto_subtitle --help

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.
