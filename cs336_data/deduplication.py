import os
from collections import Counter, defaultdict
import hashlib
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    """
    Perform exact line deduplication on a set of input files.
    1. count the number of occurrences of each line in the input files
    2. useing hash to reduce memory
    3. rewrite each file with the unique lines
    """

    # 1. count the number of occurrences of each line in the input files
    line_counts = Counter()
    for input_file in input_files:
        with open(input_file) as f:
            for line in f:
                # 2. useing hash to reduce memory
                line_counts[hash(line.strip())] += 1 if line.strip() else 0
    
    # 3. rewrite each file with the unique lines
    for input_file in input_files:
        with open(input_file) as f:
            with open(os.path.join(output_directory, os.path.basename(input_file)), "w") as out:
                for line in f:
                    if line.strip() and line_counts[hash(line.strip())] == 1:
                        out.write(line)

def get_ngrams(text: str, ngrams: int) -> set[tuple[str, ...]]:
    """generate ngrams set from text """
    tokens = word_tokenize(text)
    return set(tuple(tokens[i:i+ngrams]) for i in range(len(tokens)-ngrams+1))

def jaccard_similarity(ngrams_s1: set[tuple[str, ...]], ngrams_s2: set[tuple[str, ...]]) -> float:
    """calculate jaccard similarity between two ngrams sets"""
    return len(ngrams_s1 & ngrams_s2) / len(ngrams_s1 | ngrams_s2)

def get_signature( ngs: set[tuple[str, ...]], num_hashes: int) -> list[int]:
    """get signature from ngrams set"""
    signature = [float('inf') ] * num_hashes
    for ngram in ngs:
        for i in range(num_hashes):
            hash_value = int(hashlib.md5((str(i) + " ".join(ngram)).encode()).hexdigest(), 16)
            signature[i] = min(signature[i], hash_value)
    return signature

def signature_similarity(sgn1: list[int], sgn2: list[int]) -> float:
    """calculate signature similarity between two signature"""
    assert len(sgn1) == len(sgn2)
    same_col_num = 0
    for i in range(len(sgn1)):
        if sgn1[i] == sgn2[i]:
            same_col_num += 1
    return same_col_num / len(sgn1)

def get_bands(signature: list[int], num_bands: int) -> tuple[list[tuple[int, ...]], int]:
    """get bands and band length from signature"""
    bands = []
    assert len(signature) % num_bands == 0
    band_len = len(signature) // num_bands
    for i in range(num_bands):
        bands.append(tuple(signature[i*band_len:(i+1)*band_len]))
    return bands, band_len
    
class LSH:

    def __init__(self, num_bands: int):
        self.num_bands = num_bands
        self.buckets = [ defaultdict(list) for _ in range(num_bands) ]

    def insert(self, signature: set[tuple[str, ...]], doc_name: str):
        """insert ngram into LSH"""
        bands, _ = get_bands(signature, self.num_bands)
        for i in range(self.num_bands):
            self.buckets[i][hash(bands[i])].append(doc_name)

    def query(self, signature: list[int], file_name: str) -> set[str]:
        """return similar documents names from ngrams set"""
        bands, _ = get_bands(signature, self.num_bands)
        candidates = set()
        for i in range(self.num_bands):
            for doc_name in self.buckets[i][hash(bands[i])]:
                candidates.add(doc_name)
        candidates.remove(file_name)
        return candidates


def minhash_deduplication(
        input_files: list[os.PathLike],
        num_hashes: int,
        num_bands: int,
        ngrams: int,
        jaccard_threshold: float,
        output_directory: os.PathLike,
):
    """
    File Content -> N-grams Set S := [s_1, s_2, ..., s_m] , s_1 := ("a", "b", "c")
    N-grams Set -> signature := [minhash(h_1, S), minhash(h_2, S), ..., minhash(h_k, S)]
    signature ~> jaccard similarity ( the proportion of columns with the same minhash value)
    """

    lsh = LSH(num_bands)

    # First pass: insert all ngrams into LSH
    signatures = {}
    for input_file in input_files:
        file_name = os.path.basename(input_file)
        with open(input_file) as f:
            # generate ngrams set from file
            ngs = get_ngrams(f.read(), ngrams)
            sgn = get_signature(ngs, num_hashes)
            signatures[file_name] = sgn
            lsh.insert(sgn, file_name)

    # print("LSH built")

    # Second pass: find the similar candidates
    unique_docs = []
    seen = set()
    for input_file in input_files:
        file_name = os.path.basename(input_file)
        # is this file already been seen(similar to some other file)
        if file_name in seen:
            continue
        sgn = signatures[file_name]
        candidates: set[str] = lsh.query(sgn, file_name)
        # if no similar documents, add itself to unique_docs
        if not candidates:
            unique_docs.append(file_name)
            continue

        for candidate in candidates:
                unique_docs.append(file_name)
                file_sgn = signatures[file_name]
                candidate_sgn = signatures[candidate]
                if signature_similarity(file_sgn, candidate_sgn) > jaccard_threshold:
                    seen.add(candidate)

    # print(f"unique_docs: {len(unique_docs)}")

    # write unique_docs to output_directory
    input_directory = os.path.dirname(input_files[0])
    for unique_file in unique_docs:
        file_path = os.path.join(input_directory, unique_file)
        with open(file_path) as f:
            with open(os.path.join(output_directory, unique_file), "w") as out:
                out.write(f.read())


if __name__ == "__main__":
    test_text = "hello world, this is a test?"
    test_text2 = "hello world, this is a test."
    ngs1 = get_ngrams(test_text, 2)
    ngs2 = get_ngrams(test_text2, 2)
    print("  jaccard similarity: ", jaccard_similarity(ngs1, ngs2))
    for k in range(1, 2002, 500):
        sign1 = get_signature(ngs1, k)
        sign2 = get_signature(ngs2, k)
        print("signature similarity: ", signature_similarity(sign1, sign2), f" k={k}")

    sign1 = get_signature(ngs1, 10)
    print(sign1)
    bands1, bd1_len = get_bands(sign1, 5)
    print(bands1)

    print("==="*10)

    input_dir = "./documents_line_deduplicated"
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    minhash_deduplication(
        input_files=input_files,
        output_directory="./output",
        num_hashes=100,
        num_bands=10,
        ngrams=5,
        jaccard_threshold=0.8,
    )

