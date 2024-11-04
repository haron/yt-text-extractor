#!/usr/bin/env python

import json
import logging as log
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import pytesseract
from configargparse import ArgumentDefaultsRawHelpFormatter, ArgumentParser
from json_repair import repair_json
from openai import OpenAI
from PIL import Image
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError  # noqa

default_opts = {
    "format_sort": ["res:360"],
    "fragment_retries": 10,
    "ignoreerrors": "only_download",
    "no_warnings": True,
    "noprogress": True,
    "retries": 10,
    "writeinfojson": True,
    "outtmpl": {"default": "video.%(ext)s"},
    "paths": {},
}


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-d", "--debug", env_var="DEBUG", action="store_true")
    parser.add_argument("-t", "--openrouter-token", env_var="OPENROUTER_API_KEY", required=True)
    parser.add_argument("url", help="Youtube URL")
    return parser.parse_args()


def system(cmd, ignore_errors=False):
    log.debug(f"Running command: {cmd}")
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        log.debug(e)
        if not ignore_errors:
            raise (e)


def configure_logging(args):
    log_level = log.ERROR
    if args.verbose:
        log_level = log.INFO
    if args.debug:
        log_level = log.DEBUG
    log.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    if log.getLogger().getEffectiveLevel() <= log.INFO:
        default_opts["verbose"] = True
    else:
        default_opts["quiet"] = True


def download(temp_dir, url):
    log.info(f"Downloading {url} with yt-dlp to {temp_dir}")
    opts = deepcopy(default_opts)
    opts["paths"]["home"] = str(temp_dir)
    log.debug(f"yt-dlp options: {opts}")
    ydl = YoutubeDL(opts)
    ydl.download(url)
    log.debug(f"Downloaded files: {list(temp_dir.glob('*'))=}")
    json_file = temp_dir / "video.info.json"
    video_file = list(set(temp_dir.glob("video.*")) - set([json_file]))[0]
    info = json.load(open(json_file))
    log.info(f"Download done: {video_file=}")
    return video_file, info


def extract(url):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        video_file, info = download(temp_dir, url)
        images = make_screenshots(video_file)
        pairs = ocr(images, lang=info.get("language"))
        check(pairs, info)
        return pairs


def make_screenshots(video_file):
    system(f'ffmpeg -i "{video_file}" -r 0.33 "{video_file.parent}/%07d.jpg"')
    screenshots = list(video_file.parent.glob("*.jpg"))
    log.debug(f"Found {len(screenshots)=}, {screenshots[:2]=}...")
    return screenshots


def ocr(images, lang=None):
    ocr_lang = partial(ocr_file, lang=lang)
    pairs = sorted(zip(images, ThreadPoolExecutor().map(ocr_lang, images)))
    pairs = [(fn, text.strip()) for fn, text in pairs if text]
    log.debug(f"Got {len(pairs)=}, {pairs[:2]=}...")
    return pairs


def ocr_file(fn, lang=None):
    return pytesseract.image_to_string(Image.open(fn), lang=lang)


def ocr_llm(files, title):
    return call_llm(
        files,
        f"These are screenshots from Youtube video '{title}'. Answer with JSON dictionary using file names as keys and text from images as values.",
    )


def call_llm(prompt, system_prompt, model="meta-llama/llama-3.2-11b-vision-instruct:free"):
    client = OpenAI(base_url="https://openrouter.ai/api/v1")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": system_prompt},
        ],
    )
    answer = completion.choices[0].message.content
    return repair_json(answer)


def check(pairs, info):
    data = {fn.stem: text for fn, text in pairs}
    log.debug(f"Data to check: {data}")
    title = info["fulltitle"]
    system_prompt = f"This is JSON dictionary with following structure: image ID as key and strings as values. Text is recognized from these images with OCR. Images are derived from Youtube video '{title}'. Find strings looking like OCR errors and return a JSON-encoded list with IDs of these images. Important: return just the ID list, nothing more."  # noqa
    answer = call_llm([json.dumps(data)], system_prompt)
    files = [p[0] for p in pairs if p[0].stem in answer]
    print(ocr_llm(files, title))


def main():
    args = parse_args()
    os.environ["OPENAI_API_KEY"] = args.openrouter_token
    configure_logging(args)
    log.debug(args)
    print("\n\n".join([text for _, text in extract(args.url)]))


if __name__ == "__main__":
    main()
