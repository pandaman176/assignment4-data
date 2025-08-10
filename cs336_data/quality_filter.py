import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from cs336_data.extractor import extract_texts_from_warc
from cs336_data.train import perplexity_of_text, tokenize_english
import pickle
import os
import cs336_data.common as common
from typing import Any


def gopher_filter(text: str) -> bool:
    """判断是否通过gopher quality检测
        Contain less than 50 or more than 100,000 words.
        Have a mean word length outside the range of 3 to 10 characters.
        Have more than 30% of lines ending with an ellipsis (“...”).
        Contain less than 80% of words with at least one alphabetic character.
    """
    tokens = word_tokenize(text)
    length = len(tokens)
    if length < 50 or length > 100_000:
        return False

    mean_word_length = sum(len(token) for token in tokens) / length
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    lines = text.splitlines()
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    num_ellipsisi_ended_lines = sum(1 for line in non_empty_lines if line.endswith("..."))
    percent_ellipsis_ended_lines = num_ellipsisi_ended_lines / len(non_empty_lines)
    if percent_ellipsis_ended_lines > 0.3:
        return False
    
    # 判断是否包含字母字符
    def contains_alphabetic(token: str) -> bool:
        return any(char.isalpha() for char in token)
    
    alphabetic_words = sum(1 for token in tokens if contains_alphabetic(token))
    if alphabetic_words / length < 0.8:
        return False
    
    return True

def classify_quality(text: str) -> tuple[Any, float]:
    # load model
    if os.path.exists(common.ASSETS_PATH / "model.pkl"):
        with open(common.ASSETS_PATH / "model.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        print("No model found. Please run perplexity_of_file.py first.")
        exit()
    n = 3  # trigram 推荐起点
    tokenizer = tokenize_english
    ppl = perplexity_of_text(model, tokenizer(text), n)
    # TODO: provide real quality score instead of just using placeholder 0.9
    if ppl < 500:
        return "wiki", 0.9
    else:
        return "cc", 0.9
    
    
if __name__ == "__main__":
    for item in extract_texts_from_warc("../data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"):
        if gopher_filter(item):
            print("================>PASSED")
        else:
            print("FAILED")