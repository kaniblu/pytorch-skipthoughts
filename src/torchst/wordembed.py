import os
import tempfile
import subprocess

import torch
import numpy as np


class FastText(object):
    def __init__(self, fasttext_path, model_path):
        self.fasttext_path = fasttext_path
        self.model_path = model_path
        self.args = [fasttext_path, "print-word-vectors", model_path]
        self.process = None

    def query(self, word):
        q = (word + "\n").encode("utf-8")
        self.process.stdin.write(q)
        self.process.stdin.flush()

        res = self.process.stdout.readline()
        res = np.fromstring(res, dtype=float, sep=' ')

        return res

    def __enter__(self):
        self.process = subprocess.Popen(self.args,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.kill()


def load_embeddings(path, word_dim):
    embeddings = {}

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            tokens = line.split()
            word = " ".join(tokens[:-word_dim])
            values = [float(v) for v in tokens[-word_dim:]]
            vector = np.array(values)
            embeddings[word] = vector

    return embeddings


def preinitialize_embeddings(model, vocab, embeddings):
    for word, vector in embeddings.items():
        if word not in vocab.f2i:
            continue

        idx = vocab.f2i[word]
        model.embeddings.weight.data[idx] = (torch.FloatTensor(vector))


def load_fasttext_embeddings(words, fasttext_path, model_path):
    with FastText(fasttext_path, model_path) as fasttext:
        embeddings = {w: fasttext.query(w) for w in words}

    return embeddings


def preinitialize_glove_embeddings(model, vocab, embed_path):
    embeddings = load_embeddings(embed_path, model.word_dim)
    preinitialize_embeddings(model, vocab, embeddings)


def preinitialize_fasttext_embeddings(model, vocab, fasttext_path, model_path):
    embeddings = load_fasttext_embeddings(vocab.words, fasttext_path, model_path)
    preinitialize_embeddings(model, vocab, embeddings)