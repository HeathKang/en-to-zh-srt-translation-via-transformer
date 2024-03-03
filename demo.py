from typing import Generator
from transformers import pipeline
from srt import parse, compose, Subtitle


def load_file() -> Generator[Subtitle, None, None]:
    with open("demo-en.srt", "r") as f:
        subtitles_str = f.read()
        subtitles = parse(subtitles_str)

    return subtitles


def save_file(contents: str):
    with open("demo-zh.srt", "w") as f:
        f.write(contents)


pipe = pipeline("translation", "Helsinki-NLP/opus-mt-en-zh", device="cuda:0")

subtitles = load_file()


zh_subtitles = []
for subtitle in subtitles:
    res = pipe(subtitle.content)
    zh_single_subtitle = Subtitle(
        subtitle.index,
        subtitle.start,
        subtitle.end,
        content=res[-1].get("translation_text", ""),
    )
    print(zh_single_subtitle)
    zh_subtitles.append(zh_single_subtitle)

parsed_str = compose(zh_subtitles)
save_file(parsed_str)
