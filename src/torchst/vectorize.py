import io
import sys
import pickle
import logging

import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torchtextutils import create_generator_ae
from torchtextutils import Vocabulary
from torchtextutils import BatchPreprocessor
from torchtextutils.common import ensure_dir_exists
from yaap import path
from yaap import ArgParser

from .model import MultiContextSkipThoughts


def parse_args():
    parser = ArgParser(allow_config=True)
    parser.add("--ckpt-path", type=path, required=True)
    parser.add("--vocab-path", type=path, required=True)
    parser.add("--data-path", type=path, default=None, required=False)
    parser.add("--vector-path", type=path, default=None, required=False)
    parser.add("--batch-size", type=int, default=32)
    parser.add("--gpu", action="store_true", default=False)
    parser.add("--verbose", action="store_true", default=False)

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
        vectors = self.model.encode(x, x_lens)
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

    logging.info("Preparing vectorizing environment...")
    vectorizer = Vectorizer(model,
                            use_gpu=args.gpu)
    preprocessor = BatchPreprocessor(vocab)
    reader = file_reader(args.data_path)
    data_generator = create_generator_ae(reader,
                                         batch_size=args.batch_size,
                                         preprocessor=preprocessor,
                                         pin_memory=True,
                                         allow_residual=True,
                                         max_length=None)

    logging.info("Vectorizing...")
    vectorize(vectorizer, data_generator, args.vector_path, args.verbose)

    logging.info("Done!")

if __name__ == '__main__':
    main()