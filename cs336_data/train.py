"""
Using wikipedia data to train a n-gram model
that can calculate the perplexity of a document
to do qulification detection.
"""

import os
import numpy as np
import random
import nltk
import pickle
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.lm import KneserNeyInterpolated  # 平滑良好
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.util import ngrams
import math
from cs336_data import common


# 英文简单分词（或者用 nltk.word_tokenize）
def tokenize_english(text):
    return word_tokenize(text)

# 按文件夹读取所有 .txt 文档并分词（根据语言选择 tokenizer）
def load_corpus_from_dir(dir_path, tokenizer):
    docs = []
    for fn in os.listdir(dir_path):
        if not fn.endswith(".txt"):
            continue
        path = os.path.join(dir_path, fn)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer(text)
        if tokens:
            docs.append(tokens)
    return docs

# 训练 n-gram 模型 (Kneser-Ney)
def train_ngram_model(tokenized_texts, n):
    train_data, vocab = padded_everygram_pipeline(n, tokenized_texts)
    model = KneserNeyInterpolated(n)   # 推荐：Kneser-Ney
    model.fit(train_data, vocab)
    return model

# 用模型计算单篇文本的困惑度（手工计算以避免 API 细节差异）
def perplexity_of_text(model, tokens, n, epsilon=1e-12):
    # tokens: list of tokens (未加 pad)
    padded = list(pad_both_ends(tokens, n))
    N = 0
    log_prob_sum = 0.0
    # 从第 (n-1) 个位置开始取每个 n-gram 的最后一个词和其上下文
    for i in range(n-1, len(padded)):
        context = tuple(padded[i-(n-1):i])
        word = padded[i]
        p = model.score(word, context)  # 概率
        if p <= 0:
            p = epsilon
        log_prob_sum += math.log(p)
        N += 1
    # 避免除0
    if N == 0:
        return float("inf")
    ppl = math.exp(-log_prob_sum / N)
    return ppl

def sample_sentences_from_file(filepath, sample_size=10):
    with open(filepath, encoding='utf-8') as f:
        text = f.read()
    sentences = nltk.sent_tokenize(text)  # 拆句子
    if len(sentences) <= sample_size:
        return sentences  # 文本太短就返回全部
    return random.sample(sentences, sample_size)  # 随机抽样

def perplexity_of_file(model, filepath, tokenizer, n):
    samples = sample_sentences_from_file(filepath, sample_size=10)
    perplexitiys = []
    for sample in samples:
        tokens = tokenizer(sample)
        if tokens:
            perplexity = perplexity_of_text(model, tokens, n)
            perplexitiys.append(perplexity)
    
    return sum(perplexitiys) / len(perplexitiys)

if __name__ == "__main__":
    n = 3  # trigram 推荐起点
    wiki_dir = common.WIKI_PATH      # 训练用 Wiki 文档目录 (.txt)
    test_dir = common.TEST_PATH   # 可选：用于验证/设阈值的非-Wiki 文档
    tokenizer = tokenize_english

    if os.path.exists(common.ASSETS_PATH / "model.pkl"):
        print("Loading model...")
        with open(common.ASSETS_PATH / "model.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        # 参数
        
        # 载入语料（以列表形式：每个元素为一个 token 列表）
        wiki_texts = load_corpus_from_dir(wiki_dir, tokenizer)
        if not wiki_texts:
            print("No training texts found.")
            exit()
        # 训练模型
        model = train_ngram_model(wiki_texts, n)
        # Save the model
        with open(common.ASSETS_PATH / "model.pkl", "wb") as f:
            pickle.dump(model, f)

    
    for fn in os.listdir(test_dir):
        if not fn.endswith(".txt"):
            continue
        path = os.path.join(test_dir, fn)
        ppl = perplexity_of_file(model, path, tokenizer, n)
        print(f"Perplexity of {fn}: {ppl}")
    
    # 建议：用一个验证集来选阈值
    # 1) 对一批已知 Wiki 文档算 ppl_wiki（求均值或中位数）
    # 2) 对一批非 Wiki 文档算 ppl_nonwiki
    # 3) 选择能最好区分两者的阈值（比如 ROC/PR 或简单的区分度）
