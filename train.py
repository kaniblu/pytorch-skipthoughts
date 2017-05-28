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


def forward(model, x, x_lens, ys, ys_lens, gpu=True, volatile=False):
    x = Variable(x, volatile=volatile)
    x_lens = Variable(x_lens, volatile=volatile)
    ys_i = Variable(ys[..., :-1], volatile=volatile).contiguous()
    ys_t = Variable(ys[..., 1:], volatile=volatile).contiguous()
    ys_lens = Variable(ys_lens - 1, volatile=volatile)

    if gpu:
        x = x.cuda(async=True)
        x_lens = x_lens.cuda(async=True)
        ys_i = ys_i.cuda(async=True)
        ys_t = ys_t.cuda(async=True)
        ys_lens = ys_lens.cuda(async=True)

    model.zero_grad()
    logits, tv = model(x, x_lens, ys_i, ys_lens)

    # Clip target seq_len by output length
    ys_t = [y_t[:, :y.size(1)].contiguous() for y, y_t in zip(logits, ys_t)]

    decoder_losses = [compute_loss(y, y_t, y_lens)
                  for y, y_t, y_lens in zip(logits, ys_t, ys_lens)]

    del x, x_lens, ys_i, ys_t, ys_lens

    return logits, decoder_losses


def viz_val_text(inp_sent, out_sents, tar_sents):
    assert len(out_sents) == len(tar_sents)

    decoder_str = "\n".join(
        "Decoder#{0} Output:\t{1}\nDecoder#{0} Target:\t{2}".format(
            i + 1, out_sent, tar_sent
        ) for i, (out_sent, tar_sent) in enumerate(zip(out_sents, tar_sents))
    )
    encoder_str = "Encoder   Input:\t{}".format(inp_sent)
    str = encoder_str + "\n" + decoder_str

    return str


def to_sent(data, length, vocab):
    return " ".join(vocab[data[i]] for i in range(length))


def train(model, data_loader_fn, n_epochs, viz, save_dir, save_period,
          val_period):
    model = model.cuda()
    vocab = model.vocab
    model.reset_parameters()
    optimizer = O.Adam(model.parameters())

    step = 0

    legend = ["Average"] + ["Decoder_{}".format(i) for i in
                            range(model.n_decoders)]

    for eid in range(n_epochs):
        data_loader = iter(data_loader_fn())

        for x, x_lens, ys, ys_lens in data_loader:
            step += 1

            if step % val_period == 0:
                logits, dec_losses = forward(model, x, x_lens, ys, ys_lens,
                                             volatile=True)
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

                try:
                    inputs = x[:10]
                    input_lens = x_lens[:10]
                    targets = ys[:, :10, 1:].transpose(1, 0)
                    target_lens = (ys_lens.transpose(1, 0) - 1)
                    logits = torch.cat([logit[:10].unsqueeze(0) for logit in logits], 0)
                    preds = logits.transpose(1, 0).max(3)[1].squeeze().data.cpu()

                    inp_sents = [to_sent(x_, l, vocab)
                                 for x_, l in zip(inputs, input_lens)]
                    tar_sents = [[to_sent(y_, l, vocab) for y_, l in zip(y, y_lens)]
                                 for y, y_lens in zip(targets, target_lens)]
                    out_sents = [[to_sent(y_, l, vocab) for y_, l in zip(y, y_lens)]
                                 for y, y_lens in zip(preds, target_lens)]

                    title = "Iter #{}".format(step)
                    body = "\n".join(
                        "({})\n{}".format(i + 1, viz_val_text(inp, out, tar))
                        for i, (inp, tar, out) in enumerate(zip(inp_sents, tar_sents, out_sents))
                    )

                    print(title)
                    print(body)

                    viz.code(
                        text=title + "\n" + body,
                        opts=dict(
                            title="Validation Check"
                        )
                    )
                except:
                    pass

                continue

            _, declosses = forward(model, x, x_lens, ys, ys_lens)
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

            if step % save_period == 0:
                model_filename = "model-{}-{:.4f}".format(step, loss_val)
                path = os.path.join(save_dir, model_filename)
                torch.save(model.state_dict(), path)
                viz.save([save_dir])


def main():
    args = parse_args()
    n_decoders = args.n_before + args.n_after + (1 if args.predict_self else 0)

    assert os.path.exists(args.vocab_path)

    print("Loading vocabulary...")
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    dt = datetime.datetime.now()
    save_basename = dt.strftime("%Y%m%d") + "-{}".format(args.name)
    save_dir = os.path.join(args.save_dir, save_basename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Initializing model...")
    model_cls = SkipThoughts
    model = model_cls(vocab, args.word_dim, args.hidden_dim,
                      n_decoders=n_decoders,
                      n_layers=args.n_layers,
                      bidirectional=args.bidirectional,
                      dropout_prob=args.dropout_prob,
                      batch_first=True)

    if args.word_embeddings_path is not None:
        print("Loading word embeddings...")
        load_word_embeddings(model, vocab, args.word_embeddings_path)

    env = os.path.basename(save_dir)
    viz = Visdom(buffer_size=args.visdom_buffer_size, env=env)

    viz.code(
        text=str(args)[10:-1].replace(", ", "\n"),
        opts=dict(
            title="Arguments"
        )
    )

    def _data_loader_fn():
        preprocessor = Preprocessor(vocab, args.omit_prob, args.swap_prob)
        file_line_reader = TextFileReader(args.data_dir, shuffle_files=True)
        return DataGenerator(file_line_reader, vocab, args.batch_size,
                             max_length=args.max_len,
                             preprocessor=preprocessor,
                             n_before=args.n_before,
                             n_after=args.n_after,
                             predict_self=args.predict_self)

    print("Beginning training...")
    train(model, _data_loader_fn, args.n_epochs,
          viz=viz,
          save_dir=save_dir,
          save_period=args.save_period,
          val_period=args.val_period)

    print("Done!")


if __name__ == '__main__':
    main()
