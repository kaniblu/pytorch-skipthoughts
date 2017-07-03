import os
import pickle
import shutil
import tempfile
import datetime
import subprocess
import multiprocessing.pool as mp

import tqdm
import torch
import torch.optim as O
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

from utils.argparser import ArgParser
from utils.argparser import path
from utils.vocab import Vocabulary
from utils.generator import ContextDataGenerator
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
    parser.add("--gpu", action="store_true", default=False)
    parser.add("--n_previews", type=int, default=10)

    group = parser.add_group("Word Embedding Options")
    group.add("--wordembed_type", type=str, default="none",
              choices=["glove", "fasttext", "none"])
    group.add("--wordembed_path", type=path, default=None)
    group.add("--fasttext_path", type=path, default=None)

    group = parser.add_group("Training Options")
    group.add("--n_epochs", type=int, default=3)
    group.add("--batch_size", type=int, default=32)
    group.add("--omit_prob", type=float, default=0.05)
    group.add("--swap_prob", type=float, default=0.05)
    group.add("--val_period", type=int, default=100)
    group.add("--save_period", type=int, default=1000)
    group.add("--max_len", type=int, default=30)

    group = parser.add_group("Visdom Options")
    group.add("--visdom_host", type=str, default="localhost")
    group.add("--visdom_port", type=int, default=8097)
    group.add("--visdom_buffer_size", type=int, default=10)

    group = parser.add_group("Model Parameters")
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
    group.add("--dropout_prob", type=float, default=0.05)

    args = parser.parse_args()

    return args


def load_glove_embeddings(model, vocab, path):
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            tokens = line.split()
            word = " ".join(tokens[:-model.word_dim])

            if word not in vocab.f2i:
                continue

            idx = vocab.f2i[word]
            values = [float(v) for v in tokens[-model.word_dim:]]
            model.embeddings.weight.data[idx] = (torch.FloatTensor(values))


def load_fasttext_embeddings(model, vocab, fasttext_path, embedding_path):
    fasttext_path = os.path.abspath(fasttext_path)
    temp_dir = tempfile.gettempdir()
    query_path = os.path.join(temp_dir, "queries.txt")
    out_path = os.path.join(temp_dir, "vecs.txt")

    with open(query_path, "w") as f:
        f.write("\n".join(vocab.words))

    p1 = subprocess.Popen(["cat", query_path], stdout=subprocess.PIPE)
    output = subprocess.check_output([fasttext_path,
                                      "print-word-vectors",
                                      embedding_path],
                                     stdin=p1.stdout)
    with open(out_path, "w") as f:
        f.write(output.decode("utf-8"))

    load_glove_embeddings(model, vocab, out_path)


def prepare_batch(x, x_lens, ys, ys_lens, gpu=False, volatile=False):
    x_lens, x_idx = torch.sort(x_lens, 0, True)
    _, x_ridx = torch.sort(x_idx)
    ys_lens, ys_idx = torch.sort(ys_lens, 1, True)

    x_ridx_exp = x_ridx.unsqueeze(0).expand_as(ys_idx)
    xys_idx = torch.gather(x_ridx_exp, 1, ys_idx)

    x = x[x_idx]
    ys = torch.gather(ys, 1, ys_idx.unsqueeze(-1).expand_as(ys))

    x = Variable(x, volatile=volatile)
    x_lens = Variable(x_lens, volatile=volatile)
    ys_i = Variable(ys[..., :-1], volatile=volatile).contiguous()
    ys_t = Variable(ys[..., 1:], volatile=volatile).contiguous()
    ys_lens = Variable(ys_lens - 1, volatile=volatile)
    xys_idx = Variable(xys_idx, volatile=volatile)

    if gpu:
        x = x.cuda(async=True)
        x_lens = x_lens.cuda(async=True)
        ys_i = ys_i.cuda(async=True)
        ys_t = ys_t.cuda(async=True)
        ys_lens = ys_lens.cuda(async=True)
        xys_idx = xys_idx.cuda(async=True)

    return x, x_lens, ys_i, ys_t, ys_lens, xys_idx


def calculate_loss(x, x_lens, ys_i, ys_t, ys_lens, xys_idx, dec_logits):
    losses = []

    for logits, y, lens in zip(dec_logits, ys_t, ys_lens):
        loss = compute_loss(logits, y, lens)
        losses.append(loss)

    loss = sum(losses) / len(losses)
    return loss, losses


def val_text(x_sents, yi_sents, yt_sents, o_sents):
    text = ""

    for x_sent, yi_sent, yt_sent, o_sent in \
            zip(x_sents, yi_sents, yt_sents, o_sents):
        text += "Encoder    Input: {}\n".format(x_sent)

        for i, (si, st, so) in enumerate(zip(yi_sent, yt_sent, o_sent)):
            text += "Decoder_{} Input:  {}\n".format(i, si)
            text += "Decoder_{} Target: {}\n".format(i, st)
            text += "Decoder_{} Output: {}\n".format(i, so)

    return text


def val_sents(x, x_lens, ys_i, ys_t, ys_lens, xys_idx,
              dec_logits, vocab, n_previews):
    _, xys_ridx = torch.sort(xys_idx, 1)
    xys_ridx_exp = xys_ridx.unsqueeze(-1).expand_as(ys_i)
    ys_i = torch.gather(ys_i, 1, xys_ridx_exp)
    ys_t = torch.gather(ys_t, 1, xys_ridx_exp)
    dec_logits = [torch.index_select(logits, 0, xy_ridx)
                  for logits, xy_ridx in zip(dec_logits, xys_ridx)]
    ys_lens = torch.gather(ys_lens, 1, xys_ridx)

    x, x_lens = x[:n_previews], x_lens[:n_previews]
    ys_i, ys_t = ys_i[:, :n_previews], ys_t[:, :n_previews]
    dec_logits = torch.cat([logits[:n_previews].max(2)[1].squeeze(-1).unsqueeze(0)
                            for logits in dec_logits], 0)
    ys_lens = ys_lens[:, :n_previews]

    ys_i, ys_t = ys_i.transpose(1, 0), ys_t.transpose(1, 0)
    dec_logits, ys_lens = dec_logits.transpose(1, 0), ys_lens.transpose(1, 0)

    x, x_lens = x.data.tolist(), x_lens.data.tolist()
    ys_i, ys_t = ys_i.data.tolist(), ys_t.data.tolist()
    dec_logits, ys_lens = dec_logits.data.tolist(), ys_lens.data.tolist()

    def to_sent(data, length, vocab):
        return " ".join(vocab.i2f[data[i]] for i in range(length))

    def to_sents(data, lens, vocab):
        return [to_sent(d, l, vocab) for d, l in zip(data, lens)]

    x_sents = to_sents(x, x_lens, vocab)
    yi_sents = [to_sents(yi, y_lens, vocab) for yi, y_lens in zip(ys_i, ys_lens)]
    yt_sents = [to_sents(yt, y_lens, vocab) for yt, y_lens in zip(ys_t, ys_lens)]
    o_sents = [to_sents(dec_logit, y_lens, vocab)
               for dec_logit, y_lens in zip(dec_logits, ys_lens)]

    return x_sents, yi_sents, yt_sents, o_sents


def train(model, data_loader_fn, n_epochs, viz_pool, save_dir, save_period,
          val_period, n_previews):
    vocab = model.vocab
    optimizer = O.Adam(model.parameters())
    legend = ["average"] + ["decoder_{}".format(i) for i in
                            range(model.n_decoders)]
    step = 0
    t = tqdm.tqdm()

    for _ in range(n_epochs):
        data_loader = data_loader_fn()

        for data in data_loader:
            step += 1

            if step % val_period == 0:
                processed_data = prepare_batch(*data, model.is_cuda, True)
                x, x_lens, ys_i, ys_t, ys_lens, xys_idx = processed_data
                dec_logits, h = model(x, x_lens, ys_i, ys_lens, xys_idx)
                loss, losses = calculate_loss(*processed_data, dec_logits)
                losses_val = [l.data[0] for l in losses]
                loss_val = loss.data[0]

                plot_X = [step] * (model.n_decoders + 1)
                plot_Y = [loss_val] + losses_val

                viz_pool.apply_async(viz_run, ("plot", tuple(), dict(
                    X=[plot_X],
                    Y=[plot_Y],
                    opts=dict(
                        legend=legend,
                        title="Validation Loss"
                    )
                )))

                sents = val_sents(*processed_data, dec_logits, vocab,
                                  n_previews)
                text = val_text(*sents)

                viz_pool.apply_async(viz_run, ("code", tuple(), dict(
                    text="Step {}\n".format(step) + text,
                    opts=dict(
                        title="Validation Text".format(step)
                    )
                )))

            else:
                processed_data = prepare_batch(*data, model.is_cuda, False)
                x, x_lens, ys_i, ys_t, ys_lens, xys_idx = processed_data
                dec_logits, h = model(x, x_lens, ys_i, ys_lens, xys_idx)
                loss, losses = calculate_loss(*processed_data, dec_logits)
                losses_val = [l.data[0] for l in losses]
                loss_val = loss.data[0]

                plot_X = [step] * (model.n_decoders + 1)
                plot_Y = [loss_val] + losses_val

                loss.backward()
                clip_grad_norm(model.parameters(), 3)
                optimizer.step()

                viz_pool.apply_async(viz_run, ("plot", tuple(), dict(
                    X=[plot_X],
                    Y=[plot_Y],
                    opts=dict(
                        legend=legend,
                        title="Training Loss"
                    )
                )))

            if step % save_period == 0:
                model_filename = "model-{}-{:.4f}".format(step, loss_val)
                path = os.path.join(save_dir, model_filename)
                torch.save(model.state_dict(), path)
                viz.save([save_dir])

            t.set_description("[{}]: loss={:.4f}".format(step, loss_val))
            t.update()


def init_viz(args, kwargs):
    global viz

    viz = Visdom(*args, **kwargs)


def viz_run(f_name, args, kwargs):
    global viz

    getattr(viz, f_name).__call__(*args, **kwargs)


def main():
    args = parse_args()
    n_decoders = args.n_before + args.n_after + (1 if args.predict_self else 0)

    assert os.path.exists(args.vocab_path)

    print("Loading vocabulary...")
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_basename = timestamp + "-{}".format(args.name)
    save_dir = os.path.join(args.save_dir, save_basename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Initializing model...")
    model_cls = SkipThoughts
    model = model_cls(vocab, args.word_dim, args.hidden_dim,
                      encoder_cell=args.encoder_cell,
                      decoder_cell=args.decoder_cell,
                      n_decoders=n_decoders,
                      n_layers=args.n_layers,
                      bidirectional=args.bidirectional,
                      dropout_prob=args.dropout_prob,
                      batch_first=True)

    model.reset_parameters()

    print("Loading word embeddings...")
    if args.wordembed_type == "glove":
        load_glove_embeddings(model, vocab, args.wordembed_path)
    elif args.wordembed_type == "fasttext":
        load_fasttext_embeddings(model, vocab,
                                 fasttext_path=args.fasttext_path,
                                 embedding_path=args.wordembed_path)
    elif args.wordembed_type == "none":
        pass
    else:
        raise ValueError("Unrecognized word embedding type: {}".format(
            args.wordembed_type
        ))

    if args.gpu:
        model = model.cuda()

    viz_pool = mp.ThreadPool(1, initializer=init_viz, initargs=(tuple(), dict(
        buffer_size=args.visdom_buffer_size,
        server="http://{}".format(args.visdom_host),
        port=args.visdom_port,
        env=args.name,
        name=timestamp
    )))

    viz_pool.apply_async(viz_run, ("code", tuple(), dict(
        text=str(args)[10:-1].replace(", ", "\n"),
        opts=dict(
            title="Arguments"
        )
    )))

    config_path = os.path.join(save_dir, os.path.basename(args.config))
    shutil.copy(args.config, config_path)

    def _data_loader_fn():
        preprocessor = Preprocessor(vocab, args.omit_prob, args.swap_prob)
        file_line_reader = TextFileReader(args.data_dir, shuffle_files=True)
        return ContextDataGenerator(file_line_reader, vocab, args.batch_size,
                                    max_length=args.max_len,
                                    preprocessor=preprocessor,
                                    n_before=args.n_before,
                                    n_after=args.n_after,
                                    predict_self=args.predict_self)

    print("Beginning training...")
    train(model,
          data_loader_fn=_data_loader_fn,
          n_epochs=args.n_epochs,
          viz_pool=viz_pool,
          save_dir=save_dir,
          n_previews=args.n_previews,
          save_period=args.save_period,
          val_period=args.val_period)

    print("Done!")


if __name__ == '__main__':
    main()
