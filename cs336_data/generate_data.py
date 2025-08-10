from cs336_data.extractor import extract_texts_from_warc
import pathlib
import time
from collections.abc import Generator
from cs336_data.common import DATA_DIR, WIKI_PATH
from cs336_data.identifier import language_identification, nsfw_detection, hate_detection
from cs336_data.masker import mask_all
from cs336_data.quality_filter import gopher_filter

def data_generator(warc_path: str | pathlib.Path) -> Generator[str, None, None]:
    for item in extract_texts_from_warc(warc_path):
        if gopher_filter(item):
            continue
        language, lang_score = language_identification(item)
        if language != "en":
            continue
        elif lang_score < 0.9:
            continue
        nsfw, nsfw_score = nsfw_detection(item)
        if nsfw != "non-nsfw":
            continue
        elif nsfw_score < 0.95:
            continue
        hate, hate_score = hate_detection(item)
        if hate != "non-toxic":
            continue
        elif hate_score < 0.95:
            continue
        result = mask_all(item)
        yield result["text"]

if __name__ == "__main__":
    for item in data_generator(DATA_DIR / "data_20.warc.gz"):
        # write to WIKI_PATH
        now = time.strftime('%Y%m%d%H%M%S')
        with open(WIKI_PATH / f"data_{now}.txt", "w") as f:
            f.write(item)

    

