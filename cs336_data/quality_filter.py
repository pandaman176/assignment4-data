import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from cs336_data.extractor import extract_texts_from_warc


def gopher_filter(text: str) -> bool:
    """判断是否通过gopher quality检测
        Contain less than 50 or more than 100,000 words.
        Have a mean word length outside the range of 3 to 10 characters.
        Have more than 30% of lines ending with an ellipsis (“...”).
        Contain less than 80% of words with at least one alphabetic character.
    """
    tokens = word_tokenize(text)
    length = len(tokens)
    if length < 50 or length > 100000:
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

if __name__ == "__main__":
    for item in extract_texts_from_warc("../data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"):
        if gopher_filter(item):
            print("================>PASSED")
        else:
            print("FAILED")