import pathlib

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
LANGUAGE_MODEL_PATH = DATA_DIR / "classifiers" / "lid.176.bin"
HATE_MODEL_PATH = DATA_DIR / "classifiers" / "jigsaw_fasttext_bigrams_hatespeech_final.bin"
NSFW_MODEL_PATH = DATA_DIR / "classifiers" / "jigsaw_fasttext_bigrams_nsfw_final.bin"
WIKI_PATH = DATA_DIR / "wiki"
TEST_PATH = DATA_DIR / "test_txts"
ASSETS_PATH = pathlib.Path(__file__).parent / "assets"