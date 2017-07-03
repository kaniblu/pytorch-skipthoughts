import io
import os
import pickle

import tqdm
import torch
from torch.autograd import Variable
import torch.utils.data as tchdata
import numpy as np

from model import SkipThoughts
from utils import ensure_dir_exists
from utils.vocab import Vocabulary
from utils.argparser import ArgParser
from utils.argparser import path
from utils.preprocessor import Preprocessor
from utils.generator import AutoencodingDataGenerator
from utils.generator import TextFileReader


def parse_args():
    parser = ArgParser(allow_config=True)
    parser.add("--ckpt_path", type=path, required=True)
    parser.add("--vocab_path", type=path, required=True)
    parser.add("--data_path", type=path, required=True)
    parser.add("--vector_path", type=path, required=True)
    parser.add("--batch_size", type=int, default=32)
    parser.add("--gpu", action="store_true", default=False)

    group = parser.add_argument_group("Model Parameters")
    group.add("--encoder_cell", type=str, default="lstm",
              choices=["lstm", "gru"])
    group.add("--decoder_cell", type=str, default="gru",
              choices=["lstm", "gru"])
    group.add("--n_before", type=int, default=1)
    group.add("--n_after", type=int, default=1)
    group.add("--predict_self", type=int, default=0)
    group.add("--word_dim", type=int, default=100)
    group.add("--hidden_dim", type=int, default=100)
    group.add("--n_layers", type=int, default=2)
    group.add("--bidirectional", type=int, default=1)

    args = parser.parse_args()

    return args


def load_data(data_path):
    with io.open(data_path, "r", encoding="utf-8") as f:
        sents = f.readlines()

    sents = [sent.strip() for sent in sents]
    sents = [sent for sent in sents if sent]

    return sents


def create_data_loader(sents, preprocessor, batch_size):
    def _collate_fn(batch):
        lens = torch.LongTensor([len(s) + 1 for s in batch])
        batch = torch.LongTensor(preprocessor(batch))

        lens, idx = torch.sort(lens, dim=0, descending=True)
        batch = batch[idx]

        batch = batch.pin_memory()

        return batch, lens

    data_loader = tchdata.DataLoader(sents, batch_size, collate_fn=_collate_fn)
    return data_loader


def vectorize(model, data_loader, gpu):
    tvs = []

    for x, x_lens, _, _ in tqdm.tqdm(data_loader, desc="Encoding"):
        x_lens, x_idx = torch.sort(x_lens, 0, True)
        _, x_ridx = torch.sort(x_idx)
        x = x[x_idx]

        x_var = Variable(x, volatile=True)
        x_lens = Variable(x_lens, volatile=True)

        if gpu:
            x_var = x_var.cuda()
            x_lens = x_lens.cuda()

        tv = model.encode(x_var, x_lens)
        tv = tv.cpu().data
        tv = tv[x_ridx]
        tv = tv.numpy()

        tvs.append(tv)

    return np.vstack(tvs)


def main():
    args = parse_args()
    n_decoders = args.n_before + args.n_after + (1 if args.predict_self else 0)

    ensure_dir_exists(args.vector_path)

    print("Loading vocabulary...")
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    print("Loading model...")
    model_cls = SkipThoughts
    model = model_cls(vocab, args.word_dim, args.hidden_dim,
                      encoder_cell=args.encoder_cell,
                      decoder_cell=args.decoder_cell,
                      n_decoders=n_decoders,
                      n_layers=args.n_layers,
                      bidirectional=args.bidirectional,
                      batch_first=True)
    model.load_state_dict(torch.load(args.ckpt_path))

    if args.gpu:
        model = model.cuda()

    print("Vectorizing...")
    preprocessor = Preprocessor(vocab)
    reader = TextFileReader(args.data_path)
    data_loader = AutoencodingDataGenerator(reader, vocab, args.batch_size,
                                            max_length=9999999,
                                            preprocessor=preprocessor,
                                            pin_memory=True,
                                            add_input_noise=False,
                                            allow_residual=True)
    vectors = vectorize(model, data_loader, args.gpu)

    print("Saving vectors...")
    np.savetxt(args.vector_path, vectors)

    print("Done!")


if __name__ == '__main__':
    main()