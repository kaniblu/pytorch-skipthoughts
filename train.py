import os
import pickle
import datetime

import torch
import torch.optim as O
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

from utils.argparser import ArgParser
from utils.argparser import path
from utils.vocab import Vocabulary
from utils.generator import DataGenerator
from utils.generator import TextFileReader
from utils.preprocessor import Preprocessor
from utils.visdom import Visdom
from model import SkipThoughts
from model import compute_loss


def parse_args():
    parser = ArgParser(allow_config=True)

    parser.add("--name", type=str, default="noname")
    parser.add("--data_dir", type=path, required=True)
    parser.add("--vocab_path", type=path, required=True)
    parser.add("--save_dir", type=path, required=True)
    parser.add("--word_embeddings_path", type=path, default=None)

    group = parser.add_group("Training Options")
    group.add("--n_epochs", type=int, default=3)
    group.add("--batch_size", type=int, default=32)
    group.add("--omit_prob", type=float, default=0.05)
    group.add("--swap_prob", type=float, default=0.05)
    group.add("--val_period", type=int, default=100)
    group.add("--save_period", type=int, default=1000)
    group.add("--max_len", type=int, default=30)
    group.add("--visdom_buffer_size", type=int, default=10)

    group = parser.add_group("Model Parameters")
    group.add("--n_before", type=int, default=1)
    group.add("--n_after", type=int, default=1)
    group.add("--predict_self", type=int, default=0)
    group.add("--word_dim", type=int, default=100)
    group.add("--hidden_dim", type=int, default=100)
    group.add("--n_layers", type=int, default=2)
    group.add("--bidirectional", type=int, default=1)
    group.add("--dropout_prob", type=float, default=0.05)

    args = parser.parse_args()

    return args


def load_word_embeddings(model, vocab, path):
    with open(path, 'r') as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]

            if word not in vocab:
                continue

            idx = vocab[word]
            values = [float(v) for v in tokens[1:]]
            model.embeddings.weight.data[idx] = (torch.FloatTensor(values))


def calculate_loss(model, x, x_lens, ys, ys_lens, ys_idx, gpu=True):
    """inputs must be sorted"""
    x = Variable(x)
    x_lens = x_lens.tolist()
    ys_i = Variable(ys[..., :-1]).contiguous()
    ys_t = Variable(ys[..., 1:]).contiguous()
    ys_lens_var = Variable(ys_lens - 1)
    ys_lens = ys_lens_var.data.tolist()
    ys_idx = Variable(ys_idx)

    if gpu:
        x = x.cuda(async=True)
        ys_i = ys_i.cuda(async=True)
        ys_t = ys_t.cuda(async=True)
        ys_lens_var = ys_lens_var.cuda(async=True)
        ys_idx = ys_idx.cuda(async=True)

    model.zero_grad()
    logits, tv = model(x, x_lens, ys_i, ys_lens, ys_idx)
    dec_losses = [compute_loss(y, y_t, y_lens)
                  for y, y_t, y_lens in zip(logits, ys_t, ys_lens_var)]

    return logits, dec_losses


def train(model, data_loader_fn, n_epochs, viz, save_dir, save_period,
          val_period, vocab):
    model = model.cuda()
    model.reset_parameters()
    optimizer = O.Adam(model.parameters())

    step = 0

    legend = ["Average"] + ["Decoder_{}".format(i) for i in range(model.n_decoders)]

    for eid in range(n_epochs):
        data_loader = iter(data_loader_fn())

        while True:
            try:
                x, x_lens, ys, ys_lens = next(data_loader)
            except StopIteration:
                break

            step += 1

            x_lens, x_idx = torch.sort(x_lens, dim=0, descending=True)
            x = x[x_idx]

            ys_lens, ys_idx = torch.sort(ys_lens, dim=1, descending=True)
            for i in range(ys.size(0)):
                ys[i] = ys[i][ys_idx[i]]

            _, declosses = calculate_loss(model, x, x_lens, ys, ys_lens, ys_idx)
            decloss_vals = [loss.data[0] for loss in declosses]
            loss = sum(declosses) / model.n_decoders
            loss_val = loss.data[0]
            plot_X = [step] * (model.n_decoders + 1)
            plot_Y = [loss_val] + decloss_vals

            loss.backward()
            clip_grad_norm(model.parameters(), 3)
            optimizer.step()

            viz.plot(
                X=[plot_X],
                Y=[plot_Y],
                opts=dict(
                    legend=legend,
                    title="Training Loss"
                )
            )

            if step % 10 == 0:
                print("[{}]: loss={}".format(step, loss_val))

            if step % val_period == 0:
                try:
                    x, x_lens, ys, ys_lens = next(data_loader)
                except StopIteration:
                    break

                x_lens, x_idx = torch.sort(x_lens, dim=0, descending=True)
                x = x[x_idx]

                ys_lens, ys_idx = torch.sort(ys_lens, dim=1, descending=True)
                for i in range(ys.size(0)):
                    ys[i] = ys[i][ys_idx[i]]

                logits, declosses = calculate_loss(model, x, x_lens, ys, ys_lens, ys_idx)
                loss_vals = [loss.data[0] for loss in declosses]
                total_loss = sum(declosses)
                total_loss_val = total_loss.data[0]

                plot_X = [step] * (model.n_decoders + 1)
                plot_Y = [total_loss_val] + loss_vals

                viz.plot(
                    X=[plot_X],
                    Y=[plot_Y],
                    opts=dict(
                        legend=legend,
                        title="Training Loss"
                    )
                )

                x = x[:10]
                ys = ys[:, :10, 1:].transpose(1, 0)
                ys_lens = ys_lens.transpose(1, 0)

                preds = logits[:, :10].transpose(1, 0).max(3)[1].squeeze()
                x_sents = [" ".join(vocab[int(e[i])] for i in range(l)) for e, l
                           in zip(x, x_lens)]
                t_sents = [[" ".join(vocab[int(e[i])] for i in range(l)) for e, l
                           in zip(y, y_lens)] for y, y_lens in zip(ys, ys_lens - 1)]
                p_sents = [[
                    " ".join(vocab[int(e[i].data[0])] for i in range(l - 1)) for
                    e, l in zip(pred, y_lens)] for pred, y_lens in zip(preds, ys_lens - 1)]

                title = "Iter #{}".format(step)
                body = "\n".join(
                    "({})\nEncoder   Input:\t{}\n{}".format(
                        i + 1, x_sent, "\n".join(
                            "Decoder#{0} Output:\t{1}\nDecoder#{0} Target:\t{2}".format(
                                i + 1, p, t
                            ) for i, (t, p) in enumerate(zip(t_sent, p_sent))
                        )
                    ) for i, (x_sent, t_sent, p_sent) in
                    enumerate(zip(x_sents, t_sents, p_sents)))

                print(title)
                print(body)

                viz.code(
                    text=title + "\n" + body,
                    opts=dict(
                        title="Validation Check"
                    )
                )

            if step % save_period == 0:
                model_filename = "model-{}-{:.4f}".format(step, loss_val)
                path = os.path.join(save_dir, model_filename)
                torch.save(model.state_dict(), path)
                viz.save([save_dir])


def main():
    a = parse_args()
    n_decoders = a.n_before + a.n_after + (1 if a.predict_self else 0)

    assert os.path.exists(a.vocab_path)

    print("Loading vocabulary...")
    with open(a.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    dt = datetime.datetime.now()
    save_basename = dt.strftime("%Y%m%d") + "-{}".format(a.name)
    save_dir = os.path.join(a.save_dir, save_basename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Initializing model...")
    model_cls = SkipThoughts
    model = model_cls(len(vocab), a.word_dim, a.hidden_dim,
                      n_decoders=n_decoders,
                     n_layers=a.n_layers,
                     bidirectional=a.bidirectional,
                     dropout_prob=a.dropout_prob,
                     pad_idx=vocab[vocab.pad],
                     bos_idx=vocab[vocab.bos],
                     eos_idx=vocab[vocab.eos])

    if a.word_embeddings_path is not None:
        print("Loading word embeddings...")
        load_word_embeddings(model, vocab, a.word_embeddings_path)

    env = os.path.basename(save_dir)
    viz = Visdom(buffer_size=a.visdom_buffer_size, env=env)

    viz.code(
        text=str(a)[10:-1].replace(", ", "\n"),
        opts=dict(
            title="Arguments"
        )
    )

    def _data_loader_fn():
        preprocessor = Preprocessor(vocab, a.omit_prob, a.swap_prob)
        file_line_reader = TextFileReader(a.data_dir, shuffle_files=True)
        return DataGenerator(file_line_reader, vocab, a.batch_size, a.max_len,
                              preprocessor, n_before=a.n_before,
                             n_after=a.n_after, predict_self=a.predict_self)

    print("Beginning training...")
    train(model, _data_loader_fn, a.n_epochs, viz, save_dir, a.save_period,
          a.val_period, vocab)

    print("Done!")


if __name__ == '__main__':
    main()
