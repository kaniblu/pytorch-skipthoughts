import io
import sys
import pickle
import logging

import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torchtextutils import BatchPreprocessor
from torchtextutils import Generator
from torchtextutils import SplitWordIterator
from torchtextutils import FunctionMapper
from torchtextutils import MemoryPinner
from torchtextutils import SentenceWordTokenizer
from torchtextutils import Vocabulary
from torchtextutils.common import ensure_dir_exists
from yaap import path
from yaap import ArgParser
from configargparse import YAMLConfigFileParser

from .model import MultiContextSkipThoughts
from .wordembed import FastText
from .wordembed import load_embeddings


def parse_args():
    parser = ArgParser(allow_config=True,
                       config_file_parser_class=YAMLConfigFileParser)
    parser.add("--ckpt-path", type=path, required=True)
    parser.add("--vocab-path", type=path, required=True)
    parser.add("--data-path", type=path, default=None, required=False)
    parser.add("--vector-path", type=path, default=None, required=False)
    parser.add("--batch-size", type=int, default=32)
    parser.add("--gpu", action="store_true", default=False)
    parser.add("--verbose", action="store_true", default=False)
    parser.add("--flush_char", type=str, default=chr(0x05))

    group = parser.add_argument_group("Word Expansion Options")
    group.add("--wordembed-type", type=str, default=None,
              choices=["glove", "fasttext", None])
    group.add("--wordembed-path", type=path, default=None)
    group.add("--fasttext-path", type=path, default=None)

    group = parser.add_argument_group("Model Parameters")
    group.add("--encoder-cell", type=str, default="gru",
              choices=["lstm", "gru", "sru"])
    group.add("--decoder-cell", type=str, default="gru",
              choices=["lstm", "gru", "sru"])
    group.add("--before", type=int, default=1)
    group.add("--after", type=int, default=1)
    group.add("--predict-self", action="store_true", default=False)
    group.add("--word-dim", type=int, default=100)
    group.add("--hidden-dim", type=int, default=100)
    group.add("--layers", type=int, default=2)
    group.add("--bidirectional", action="store_true", default=False)
    group.add("--reverse-encoder", action="store_true", default=False)

    args = parser.parse_args()

    return args


class EmbedDict(object):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def query(self, word):
        return self.embeddings.get(word)


def file_reader(data_path):
    if data_path is None:
        for line in sys.stdin:
            yield line.strip()
    else:
        with io.open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()


class Vectorizer(object):
    def __init__(self, model, use_gpu=False):
        self.model = model
        self.is_cuda = use_gpu

        if use_gpu:
            self.model = model.cuda()

    def prepare_batch(self, x, x_lens):
        x_lens, x_idx = torch.sort(x_lens, 0, True)
        _, x_ridx = torch.sort(x_idx)
        x = x[x_idx]

        x_var = Variable(x, volatile=True)
        x_lens = Variable(x_lens, volatile=True)
        x_ridx = Variable(x_ridx.long(), volatile=True)

        if self.is_cuda:
            x_var = x_var.cuda()
            x_lens = x_lens.cuda()
            x_ridx = x_ridx.cuda()

        return x_var, x_lens, x_ridx

    def vectorize(self, x, x_lens):
        x, x_lens, r_idx = self.prepare_batch(x, x_lens)

        if len(x.size()) == 2:
            vectors = self.model.encode(x, x_lens)
        elif len(x.size()) == 3:
            vectors = self.model.encode_embed(x, x_lens)
        else:
            raise Exception()

        vectors = torch.index_select(vectors, 0, r_idx)

        if self.is_cuda:
            vectors = vectors.cpu()

        return vectors.data


def vectorize(vectorizer, data_generator, save_path, verbose):

    for x, x_lens in tqdm.tqdm(data_generator,
                               desc="vectorizing",
                               disable=not verbose):
        vectors = vectorizer.vectorize(x, x_lens)
        vectors = vectors.numpy()

        if save_path is None:
            np.savetxt(sys.stdout.buffer, vectors)
        else:
            with open(save_path, "ab") as f:
                np.savetxt(f, vectors)


class BatchGenerator(Generator):
    def __init__(self, items, batch_size,
                 allow_residual=True,
                 stop_char=chr(0x05)):
        self.items = items
        self.batch_size = batch_size
        self.allow_residual = allow_residual
        self.stop_char = stop_char

    def generate(self):
        batch = []

        for item in self.items:
            if item:
                c = item[0]
            else:
                c = None

            if c != self.stop_char:
                batch.append(item)

            if c != self.stop_char and len(batch) < self.batch_size:
                continue

            if batch:
                yield batch

            del batch
            batch = []

        if self.allow_residual and batch:
            yield batch
            del batch


# Custom Batch Preprocessor
class EmbeddingBatchPreprocessor(BatchPreprocessor):
    def __init__(self, model, we, *args, **kwargs):
        super(EmbeddingBatchPreprocessor, self).__init__(*args, **kwargs)

        self.wordembed = we
        self.embeddings = model.embeddings.weight.data.cpu()
        self.unk = self.vocab.unk
        self.pad = self.vocab.pad
        self.bos = self.vocab.bos
        self.eos = self.vocab.eos

    def __call__(self, batch):
        batch_new = []

        for words in batch:
            if self.add_bos:
                words = [self.bos] + words
            if self.add_eos:
                words = words + [self.eos]

            batch_new.append(words)

        batch = batch_new
        batch_new = []
        lens = [len(w) for w in batch]
        max_len = max(lens)

        for words in batch:
            if len(words) < max_len:
                words += [self.pad] * (max_len - len(words))

            words_embed = []

            for w in words:
                if w in self.vocab.f2i:
                    index = self.vocab.f2i[w]
                    words_embed.append(self.embeddings[index])
                else:
                    vector = self.wordembed.query(w)

                    if vector is None:
                        words_embed.append(self.embeddings[self.unk_idx])
                    else:
                        vector = torch.Tensor(vector)
                        words_embed.append(vector)

            sent = torch.stack(words_embed)
            batch_new.append(sent)

        batch = torch.stack(batch_new)
        lens = torch.LongTensor(lens)

        return batch, lens


def create_generator_ae(sents, batch_size, preprocessor,
                        pin_memory=True, allow_residual=True, max_length=None,
                        word_iterator=SplitWordIterator,
                        flush_char=chr(0x05)):
    sent_tokens = SentenceWordTokenizer(sents, max_length,
                                        word_iterator=word_iterator)
    batches = BatchGenerator(sent_tokens, batch_size,
                             allow_residual=allow_residual)
    prep_batches = FunctionMapper(batches, preprocessor)

    if pin_memory:
        ret = MemoryPinner(prep_batches)
    else:
        ret = prep_batches

    return ret


def main():
    args = parse_args()
    n_decoders = args.before + args.after + (1 if args.predict_self else 0)

    if not args.verbose:
        logging.basicConfig(level=logging.CRITICAL)

    if args.vector_path is not None:
        ensure_dir_exists(args.vector_path)

    logging.info("Loading vocabulary...")
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    logging.info("Loading model...")
    model_cls = MultiContextSkipThoughts
    model = model_cls(vocab, args.word_dim, args.hidden_dim,
                      encoder_cell=args.encoder_cell,
                      decoder_cell=args.decoder_cell,
                      n_decoders=n_decoders,
                      n_layers=args.layers,
                      bidirectional=args.bidirectional,
                      reverse_encoder=args.reverse_encoder,
                      batch_first=True)
    model.load_state_dict(torch.load(args.ckpt_path))

    logging.info("Loading word embeddings...")
    we_type = args.wordembed_type

    logging.info("Preparing vectorizing environment...")
    vectorizer = Vectorizer(model,
                            use_gpu=args.gpu)

    if we_type is not None:
        if we_type == "glove":
            embeddings = load_embeddings(args.wordembed_path, args.word_dim)
            we = EmbedDict(embeddings)
        elif we_type == "fasttext":
            we = FastText(args.fasttext_path, args.wordembed_path)
        else:
            raise ValueError("Unrecognized word embed type: {}".format(
                we_type
            ))

        preprocessor = EmbeddingBatchPreprocessor(
            model=model,
            we=we,
            vocab=vocab
        )
    else:
        preprocessor = BatchPreprocessor(
            vocab=vocab,
        )

    reader = file_reader(args.data_path)
    data_generator = create_generator_ae(reader,
                                         batch_size=args.batch_size,
                                         preprocessor=preprocessor,
                                         pin_memory=True,
                                         allow_residual=True,
                                         max_length=None,
                                         flush_char=args.flush_char)

    logging.info("Vectorizing...")
    vectorize(vectorizer, data_generator, args.vector_path, args.verbose)

    logging.info("Done!")

if __name__ == '__main__':
    main()