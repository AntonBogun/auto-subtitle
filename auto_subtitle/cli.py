import os
import subprocess
from typing import Dict, List
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from pathlib import Path
from .utils import filename, str2bool, write_srt
import re
import torch

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "video", nargs="+", type=str, help="paths to video files to transcribe"
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=whisper.available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=".",
        help="directory to save the outputs",
    )
    parser.add_argument(
        "--output_srt",
        type=str2bool,
        default=False,
        help="whether to output the .srt file along with the video files",
    )
    parser.add_argument(
        "--srt_only",
        type=str2bool,
        default=False,
        help="only generate the .srt file and not create overlayed video",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="whether to print out the progress and debug messages",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        choices=[
            "auto",
            "af",
            "am",
            "ar",
            "as",
            "az",
            "ba",
            "be",
            "bg",
            "bn",
            "bo",
            "br",
            "bs",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fo",
            "fr",
            "gl",
            "gu",
            "ha",
            "haw",
            "he",
            "hi",
            "hr",
            "ht",
            "hu",
            "hy",
            "id",
            "is",
            "it",
            "ja",
            "jw",
            "ka",
            "kk",
            "km",
            "kn",
            "ko",
            "la",
            "lb",
            "ln",
            "lo",
            "lt",
            "lv",
            "mg",
            "mi",
            "mk",
            "ml",
            "mn",
            "mr",
            "ms",
            "mt",
            "my",
            "ne",
            "nl",
            "nn",
            "no",
            "oc",
            "pa",
            "pl",
            "ps",
            "pt",
            "ro",
            "ru",
            "sa",
            "sd",
            "si",
            "sk",
            "sl",
            "sn",
            "so",
            "sq",
            "sr",
            "su",
            "sv",
            "sw",
            "ta",
            "te",
            "tg",
            "th",
            "tk",
            "tl",
            "tr",
            "tt",
            "uk",
            "ur",
            "uz",
            "vi",
            "yi",
            "yo",
            "zh",
        ],
        help="What is the origin language of the video? If unset, it is detected automatically.",
    )

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")

    video_pathes = args.pop("video")
    good_video_pathes = [Path(p).resolve() for p in video_pathes]

    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection."
        )
        args["language"] = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language

    #use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = whisper.load_model(model_name)
    model = whisper.load_model(model_name, device=device)

    audios = get_audio(good_video_pathes)
    subtitles_path = get_subtitles(
        audios,
        output_srt or srt_only,
        output_dir,
        lambda audio_path: model.transcribe(str(audio_path), **args),
    )

    if srt_only:
        return

    for path, srt_path in subtitles_path.items():
        srt_path = Path(srt_path).resolve()
        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

        if not (os.path.exists(srt_path) and os.path.exists(path)):
            print("Missing srt or source file, something went wrong")
            continue

        command = build_ffmpeg_command(path, srt_path, out_path)
        execute_ffmpeg_command(command)


def to_raw_path(path: str) -> str:
    return path.replace("\\", "/")


def get_audio(source_file_path_arr: List[Path]) -> Dict[Path, Path]:
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for source_file in source_file_path_arr:
        print(f"Extracting audio from {os.path.basename(source_file)}...")
        output_path = Path(temp_dir) / f"{Path(source_file).stem}.wav"

        ffmpeg.input(filename=str(source_file)).output(
            str(output_path), acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)

        audio_paths[source_file] = output_path

    return audio_paths


def get_subtitles(
    audio_paths: Dict[Path, Path],
    output_srt: bool,
    output_dir: str,
    transcribe: callable,
) -> Dict[Path, Path]:
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_dir = Path(output_dir) if output_srt else Path(tempfile.gettempdir())
        srt_path = srt_dir / f"{Path(path).stem}.srt"

        # Skip if file already exists
        if srt_path.exists():
            print(f"Subtitle file for {Path(path).name} already exists ({srt_path}). Skipping...")
            subtitles_path[path] = str(srt_path)
            continue

        print(f"Generating subtitles for {Path(path).name}... This might take a while.")

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)

        subtitles_path[path] = str(srt_path)

    return subtitles_path


def execute_ffmpeg_command(command):
    try:
        # Run the command and wait for it to complete
        subprocess.run(command, check=True, shell=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        # Handle errors in the subprocess
        print(f"An error occurred: {e}")


# def build_ffmpeg_command(video_path, subtitle_path, output_path):
#     video_path = str(video_path)
#     subtitle_path = str(subtitle_path)
#     output_path = str(output_path)
#     # Replace backslashes with forward slashes
#     subtitle_path = subtitle_path.replace("\\", "\\\\")
#     if ":" in subtitle_path:
#         subtitle_path = subtitle_path.replace(":", "\\:")

#     command = (
#         f'ffmpeg -i "{video_path}" -filter_complex '
#         f'"[0:v]subtitles=\'{subtitle_path}\'[v]" -map "[v]" -map "0:a" -y "{output_path}"'
#     )

#     return command
# def escape_ffmpeg_path(path: str) -> str:
#     # Escape backslashes and single quotes
#     return path.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
def escape_ffmpeg_path(path: str) -> str:
    """
    Properly escape a path for use in FFmpeg filter_complex arguments.
    
    This handles the multiple levels of escaping required:
    1. First level: Escaping characters within the filter option value
    2. Second level: Escaping for the filtergraph description
    3. For Windows paths, convert backslashes to forward slashes or escape properly
    
    Args:
        path: The file path to escape
        
    Returns:
        Properly escaped path for FFmpeg filter_complex
    
    References:
        http://underpop.online.fr/f/ffmpeg/ffmpeg-utils.html.gz#quoting_005fand_005fescaping
        http://underpop.online.fr/f/ffmpeg/help/notes-on-filtergraph-escaping.htm.gz
    """
    if os.name != "nt":
        raise NotImplementedError("Only Windows is supported for now")
    # return (r"\'".join(("'"+part+"'" if len(part)>0 else "") for part in path.split("'") if len(part)>0)) if len(path)>0 else "''"
    # For Windows paths, we have two options:
    # 1. Convert backslashes to forward slashes (simpler)
    path = path.replace('\\', '/')
    
    # Option 2 (alternative): Escape backslashes properly (more complex)
    # path = path.replace('\\', '\\\\')
    
    # First level escaping: escape ':
    path = re.sub(r"([:'\\])", r"\\\1", path)
    #enclose to keep spaces
    path = f"'{path}'"
    # Second level escaping: escape filter graph special chars [],;'
    path = re.sub(r"([,\[\];'\\])", r"\\\1", path)
    #enclose a final time
    return f"'{path}'"


# def escape_commandline_path(path:str)->str:
#     #just need to escape " and \ for command line
#     return '"' + path.replace('\\', '\\\\').replace('"', '\\"') + '"'
def build_ffmpeg_command(video_path, subtitle_path, output_path):
    video_path = str(video_path)
    subtitle_path = str(subtitle_path)
    # subtitle_path_escaped = escape_ffmpeg_path(subtitle_path)
    output_path = str(output_path)
    
    # Use proper escaping for the subtitle path
    # Double escape for Windows paths
    # subtitle_path_escaped = subtitle_path.replace('\\', '\\\\').replace('\'', '\\\'').replace(':', '\\:')
    
    # For ffmpeg subtitles filter, we need to escape single quotes properly
    command = (
        # f'ffmpeg -i "{video_path}" -filter_complex '
        # f'"[0:v]subtitles={subtitle_path_escaped}" -map 0:v -map 0:a -y "{output_path}"'
        f'ffmpeg -i "{video_path}" -i "{subtitle_path}" -c:v copy -c:a copy -c:s mov_text -y "{output_path}"'
    )
    
    return command


if __name__ == "__main__":
    main()
