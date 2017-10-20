import os
import pickle
import logging
import subprocess
import itertools
import multiprocessing
import multiprocessing.pool as mp

import torch
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
from torch.autograd import Variable
from torchtextutils import Vocabulary
import numpy as np
import tqdm
import yaap


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


def _mp_initialize(wd):
    global word_dim
    word_dim = wd


def _mp_process(lines):
    global word_dim
    ret = {}

    for line in lines:
        tokens = line.split()
        word = " ".join(tokens[:-word_dim])
        values = [float(v) for v in tokens[-word_dim:]]
        vector = np.array(values)
        ret[word] = vector

    return ret


def aggregate_dicts(*dicts):
    ret = {}
    for d in dicts:
        ret.update(d)

    return ret


def chunks(it, n, k):
    buffer = [[] for _ in range(n)]
    buf_it = iter(itertools.cycle(buffer))

    for item in it:
        buf_item = next(buf_it)

        if len(buf_item) == k:
            yield buffer
            buffer = [[] for _ in range(n)]
            buf_it = iter(itertools.cycle(buffer))
            buf_item = next(buf_it)

        buf_item.append(item)

    if all(buffer):
        yield buffer


def load_embeddings_mp(path, word_dim, processes=None):

    if processes is None:
        processes = multiprocessing.cpu_count()

    pool = mp.Pool(processes, initializer=_mp_initialize,
                   initargs=(word_dim,))

    with open(path, "r") as f:
        iterator = chunks(f, n=processes,
                          k=processes * 10000)
        ret = {}
        for batches in iterator:
            results = pool.map_async(_mp_process, batches)
            results = results.get()
            results = aggregate_dicts(*results)

            ret.update(results)

        return ret


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


class WordEmbeddingTranslator(nn.Module):
    def __init__(self, word_dim):
        super(WordEmbeddingTranslator, self).__init__()
        self.word_dim = word_dim
        self.W = nn.Linear(word_dim, word_dim)

    def forward(self, src):
        batch_size, word_dim = src.size()

        assert word_dim == self.word_dim, \
            "word dimension must be equal to the module's"

        return self.W(src)


class WordEmbeddingTranslatorTrainer(object):
    def __init__(self, model, data_generator, epochs, loss):
        self.epochs = epochs
        self.model = model
        self.data_generator = data_generator
        self.loss = loss

        if loss == "smoothl1":
            self.loss_fn = F.smooth_l1_loss
        elif loss == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("Unrecognized loss type: {}".format(loss))

    def run_loss(self, src, target):
        o = self.model(src)
        batch_size, word_dim = src.size()
        batch_size_t, word_dim_t = target.size()

        assert batch_size == batch_size_t, "batch sizes must be equal"
        assert word_dim == word_dim_t == self.model.word_dim, \
            "word dimensions must be equal"

        loss = self.loss_fn(o, target)

        return loss

    def train(self):
        optimizer = O.Adam(self.model.parameters())
        t = tqdm.tqdm()

        for epoch_id in range(self.epochs):
            for x, y in self.data_generator:
                if self.model.W.weight.is_cuda:
                    x = x.cuda()
                    y = y.cuda()

                optimizer.zero_grad()
                loss = self.run_loss(x, y)
                loss.backward()
                optimizer.step()

                loss_val = loss.data[0]
                t.set_description("loss: {}".format(loss_val))
                t.update()


class WordEmbeddingTranslationGenerator(object):
    def __init__(self, src, target, shuffle=True, batch_size=32, pin_memory=True):
        """
        Arguments:
            src: Tensor
            target: Tensor
        """
        assert src.size() == target.size(), "two embeddings must have the" \
                                            "same size"

        self.vocab_size, self.word_dim = src.size()
        self.src = src
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def __iter__(self):
        return self.generate()

    def generate(self):
        if self.shuffle:
            idx = np.random.permutation(np.arange(self.vocab_size))
            idx = torch.LongTensor(idx)
            src = self.src[idx]
            target = self.target[idx]
        else:
            src = self.src
            target = self.target

        for i in range(0, self.vocab_size, self.batch_size):
            b_idx = i
            e_idx = i + self.batch_size
            batch_x = src[b_idx:e_idx].contiguous()
            batch_y = target[b_idx:e_idx].contiguous()

            batch_x = Variable(batch_x)
            batch_y = Variable(batch_y)

            yield batch_x, batch_y


class ModelWordEmbedding(nn.Module):
    def __init__(self, vocab, word_dim):
        super(ModelWordEmbedding, self).__init__()
        self.word_dim = word_dim
        self.embeddings = nn.Embedding(len(vocab), word_dim)

    def load_embeddings(self, state_dict):
        self_state_dict = self.state_dict()
        self_states = set(self_state_dict.keys())
        states = set(state_dict)

        assert self_states & states, "Given state dict does not contain " \
                                     "word embedding params"

        for name, param in state_dict.items():
            if name not in self_state_dict:
                continue

            if isinstance(param, nn.Parameter):
                param = param.data

            self_state_dict[name].copy_(param)


def build_wordembed_dict(embeddings, vocab):
    """Builds a wordembedding dictionary (word->np vector) from tensor"""
    ret = {}
    for i, embedding in enumerate(embeddings):
        word = vocab.i2f[i]
        ret[word] = embedding.numpy()

    return ret


def join_embeddings(src_we, target_we):
    """joins and filters words not in common and produces two tensors"""
    src_w = set(src_we.keys())
    target_w = set(target_we.keys())
    common_w = src_w & target_w

    src_tensor = []
    target_tensor = []
    for w in common_w:
        src_tensor.append(src_we[w])
        target_tensor.append(target_we[w])

    src_tensor = torch.Tensor(np.stack(src_tensor))
    target_tensor = torch.Tensor(np.stack(target_tensor))

    return src_tensor, target_tensor


def build_wordembed_tensor(embeddings, vocab):
    """Builds a wordembedding tensor from word->numpy dictionaries"""
    embeddings_rearr = []
    reserved = set(vocab.reserved.values())
    for i in range(len(vocab)):
        word = vocab.i2f[i]
        if word in reserved:
            continue
        embeddings_rearr.append(embeddings[word])

    embeddings = torch.Tensor(np.stack(embeddings_rearr))

    return embeddings


def train():
    parser = yaap.ArgParser(allow_config=True)
    parser.add("--word-dim", type=int, required=True)
    parser.add("--ckpt-path", type=yaap.path, required=True)
    parser.add("--vocab-path", type=yaap.path, required=True)
    parser.add("--save-path", type=yaap.path, required=True)
    parser.add("--wordembed-type", type=str, required=True,
               choices=["glove", "fasttext"])
    parser.add("--wordembed-path", type=yaap.path, required=True)
    parser.add("--fasttext-path", type=yaap.path, default=None,
               help="Path to FastText binary.")
    parser.add("--wordembed-processes", type=int, default=1)
    parser.add("-v", "--verbose", action="store_true", default=False)

    group = parser.add_group("Training Options")
    group.add_argument("--epochs", type=int, default=10)
    group.add_argument("--batch-size", type=int, default=32)
    group.add_argument("--method", type=str, default="pytorch",
                       choices=["pytorch", "sklearn"])
    group.add_argument("--loss", type=str, default="smoothl1",
                       choices=["smoothl1", "l2", "l1"])
    group.add_argument("--gpu", action="store_true", default=False)
    group.add_argument("--no-shuffle", action="store_true", default=False)

    args = parser.parse_args()

    if args.verbose:
        loglvl = logging.INFO
    else:
        loglvl = logging.CRITICAL

    logging.basicConfig(level=loglvl)
    logging.info("initializing...")

    word_dim = args.word_dim
    assert os.path.exists(os.path.dirname(args.save_path)), "base directory"\
        "for saving translation weights does not exist"
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    assert args.wordembed_processes >= 1, "number of processes must be " \
                                          "larger than or equal to 1"

    if args.wordembed_processes == 1:
        embedding_loader = load_embeddings
    else:
        def embedding_loader(path, word_dim):
            return load_embeddings_mp(path, word_dim,
                                       processes=args.wordembed_processes)

    logging.info("training translation model of word_dim={}".format(word_dim))

    logging.info("loading target word embeddings...")

    model_we = ModelWordEmbedding(vocab, word_dim)
    model_we.load_embeddings(torch.load(args.ckpt_path))
    target_we = model_we.embeddings.weight.data
    target_we = build_wordembed_dict(target_we, vocab)

    logging.info("loading source word embeddings...")
    if args.wordembed_type == "glove":
        glove_path = args.wordembed_path
        src_embeddings = embedding_loader(glove_path, word_dim)
    elif args.wordembed_type == "fasttext":
        fasttext_path = args.fasttext_path
        assert fasttext_path is not None, "if wordembed type is fasttext, "\
            "you must provide path to fasttext binary"

        model_path = args.wordembed_path
        src_embeddings = load_fasttext_embeddings(vocab, fasttext_path, model_path)
    else:
        raise ValueError("Unrecognized wordembed type: {}".format(
            args.wordembed_type
        ))

    logging.info("joining source and target word embeddings...")
    src_we, target_we = join_embeddings(src_embeddings, target_we)

    logging.info("preparing training environment...")
    if args.method == "sklearn":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression(n_jobs=7)
    elif args.method == "pytorch":
        model = WordEmbeddingTranslator(word_dim)
    else:
        assert False

    logging.info("beginning training...")
    if args.method == "sklearn":
        src_we = src_we.numpy()
        target_we = target_we.numpy()
        model.fit(src_we, target_we)
    elif args.method == "pytorch":
        data_generator = WordEmbeddingTranslationGenerator(
            src=src_we, target=target_we, shuffle=not args.no_shuffle,
            batch_size=args.batch_size
        )
        trainer = WordEmbeddingTranslatorTrainer(
            model=model, data_generator=data_generator,
            epochs=args.epochs,
            loss=args.loss
        )
        trainer.train()
    else:
        assert False

    logging.info("saving results...")
    if args.method == "sklearn":
        pickle.dump(model, open(args.save_path, "wb"))
    elif args.method == "pytorch":
        torch.save(model.state_dict(), args.save_path)

    logging.info("done!")


if __name__ == "__main__":
    train()