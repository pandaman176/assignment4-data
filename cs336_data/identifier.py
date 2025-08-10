import fasttext
import pathlib
from cs336_data.common import LANGUAGE_MODEL_PATH, DATA_DIR, HATE_MODEL_PATH, NSFW_MODEL_PATH
from cs336_data.extractor import extract_texts_from_warc

def language_identification(text: str, model_path: str | pathlib.Path = LANGUAGE_MODEL_PATH):
    """
    take a unicode string and return a pair containing an identifier of the language
    and a confidence score
    """

    model = fasttext.load_model(str(model_path))
    processed_text = text.replace("\n", " ").strip() # remove '\n'
    result = model.predict(processed_text)
    return result[0][0][9:], result[1][0]

def nsfw_detection(text: str, model_path: str | pathlib.Path = NSFW_MODEL_PATH):
    """
    take a unicode string and return a pair containing an identifier of the language
    and a confidence score
    """
    model = fasttext.load_model(str(model_path))
    processed_text = text.replace("\n", " ").strip() # remove '\n'
    result = model.predict(processed_text)
    return result[0][0][9:], result[1][0]

def hate_detection(text: str, model_path: str | pathlib.Path = HATE_MODEL_PATH):
    """
    take a unicode string and return a pair containing an identifier of the language
    and a confidence score
    """
    model = fasttext.load_model(str(model_path))
    processed_text = text.replace("\n", " ").strip() # remove '\n'
    result = model.predict(processed_text)
    return result[0][0][9:], result[1][0]

if __name__ == "__main__":
    i = 0
    for item in extract_texts_from_warc(DATA_DIR / "CC" / "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"):
        print(item[:100])
        print("-"*100)
        print(language_identification(item))
        print(nsfw_detection(item))
        print(hate_detection(item))
        print("="*100)
        i += 1
        if i > 20:
            break
