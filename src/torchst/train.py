import os
import pickle
import shutil
import logging
import datetime
import multiprocessing.pool as mp

import torch
import torch.optim as O
import tqdm
from torch.autograd import Variable
from torch.nn.parallel import data_parallel
from torch.nn.utils import clip_grad_norm
from torchtextutils import Vocabulary
from torchtextutils import BatchPreprocessor
from torchtextutils import DirectoryReader
from torchtextutils import OmissionNoisifier
from torchtextutils import SwapNoisifier
from torchtextutils import create_generator_st
from visdom_pooled import Visdom
from yaap import ArgParser
from yaap import path
from configargparse import YAMLConfigFileParser
from tensorboard import SummaryWriter

from .model import MultiContextSkipThoughts
from .model import compute_loss
from .wordembed import load_embeddings_mp
from .wordembed import load_embeddings
from .wordembed import load_fasttext_embeddings
from .wordembed import preinitialize_embeddings


def parse_args():
    parser = ArgParser(allow_config=True,
                       config_file_parser_class=YAMLConfigFileParser)
    parser.add("--name", type=str, default="main")
    parser.add("--data-path", type=path, action="append", required=True,
               help="Path to a sentence file or directory that contains "
                    "a set of sentence files where each line is a sentence, "
                    "in which tokens are separated by spaces.")
    parser.add("--vocab-path", type=path, required=True)
    parser.add("--save-dir", type=path, required=True)
    parser.add("--gpus", type=int, action="append")
    parser.add("--previews", type=int, default=10)
    parser.add("--batch-first", action="store_true", default=True,
               help="currently due to the limitation of DataParallel API,"
                    "it is impossible to operate without batch-first data")
    parser.add("--visualizer", type=str, default=None,
               choices=["visdom", "tensorboard"])

    group = parser.add_group("Word Embedding Options")
    group.add("--wordembed-type", type=str, default="none",
              choices=["glove", "fasttext", "none"])
    group.add("--wordembed-path", type=path, default=None)
    group.add("--fasttext-path", type=path, default=None,
              help="Path to FastText binary.")
    group.add("--wordembed-freeze", action="store_true", default=False)
    group.add("--wordembed-processes", type=int, default=4)

    group = parser.add_group("Training Options")
    group.add("--epochs", type=int, default=3)
    group.add("--batch-size", type=int, default=32)
    group.add("--omit-prob", type=float, default=0.05)
    group.add("--swap-prob", type=float, default=0.05)
    group.add("--val-period", type=int, default=100)
    group.add("--save-period", type=int, default=1000)
    group.add("--max-len", type=int, default=30)

    group = parser.add_group("Visdom Options")
    group.add("--visdom-host", type=str, default="localhost")
    group.add("--visdom-port", type=int, default=8097)
    group.add("--visdom-buffer-size", type=int, default=10)

    group = parser.add_group("Model Parameters")
    group.add("--reverse-encoder", action="store_true", default=False)
    group.add("--encoder-cell", type=str, default="lstm",
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
    group.add("--dropout-prob", type=float, default=0.05)

    args = parser.parse_args()

    return args


class DataGenerator(object):
    def __init__(self, data_paths, vocab, omit_prob, swap_prob, batch_size,
                 max_len, n_before, n_after, predict_self=False,
                 shuffle_files=True, batch_first=True, pin_memory=True,
                 allow_residual=True):
        self.data_paths = data_paths
        self.vocab = vocab
        self.omit_prob = omit_prob
        self.swap_prob = swap_prob
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_before = n_before
        self.n_after = n_after
        self.predict_self = predict_self
        self.shuffle_files = shuffle_files
        self.batch_first = batch_first
        self.pin_memory = pin_memory
        self.allow_residual = allow_residual

    def __iter__(self):
        return self.generate()

    def generate(self):
        preprocessor = BatchPreprocessor(self.vocab)
        line_reader = file_list_reader(self.data_paths,
                                       shuffle_files=self.shuffle_files)
        data_generator = create_generator_st(line_reader,
                                             batch_first=self.batch_first,
                                             batch_size=self.batch_size,
                                             preprocessor=preprocessor,
                                             pin_memory=self.pin_memory,
                                             allow_residual=self.allow_residual,
                                             max_length=self.max_len,
                                             n_before=self.n_before,
                                             n_after=self.n_after,
                                             predict_self=self.predict_self)
        noisifiers = []

        if self.omit_prob > 0:
            unk_idx = self.vocab[self.vocab.unk]
            omitter = OmissionNoisifier(self.omit_prob, unk_idx)
            noisifiers.append(omitter)

        if self.swap_prob > 0:
            swapper = SwapNoisifier(self.swap_prob)
            noisifiers.append(swapper)

        for in_data, in_lens, out_data, out_lens in data_generator:
            for nosifier in noisifiers:
                # in-place noisification
                nosifier((in_data, in_lens))

            yield in_data, in_lens, out_data, out_lens


class Trainer(object):
    def __init__(self, model, gpu_devices, data_generator, n_epochs,
                 logger, save_dir, save_period, val_period, previews,
                 batch_first=True):
        self.model = model
        self.gpu_devices = gpu_devices
        self.data_generator = data_generator
        self.n_epochs = n_epochs
        self.logger = logger
        self.save_dir = save_dir
        self.save_period = save_period
        self.val_period = val_period
        self.previews = previews
        self.batch_first = batch_first
        self.legend = ["average"] + ["decoder_{}".format(i)
                                     for i in range(self.model.n_decoders)]

        if gpu_devices:
            self.model = model.cuda()

    @property
    def is_cuda(self):
        return len(self.gpu_devices) > 0

    def prepare_batches(self, batch_data, chunks, **kwargs):
        x, x_lens, ys, ys_lens = batch_data
        batch_dim = 0 if self.batch_first else 1

        x_list = x.chunk(chunks, 0)
        x_lens_list = x_lens.chunk(chunks, 0)
        ys_list = ys.chunk(chunks, batch_dim)
        ys_lens_list = ys_lens.chunk(chunks, batch_dim)
        inp_list = [x_list, x_lens_list, ys_list, ys_lens_list]

        data_list = []
        for inp in zip(*inp_list):
            data = self.prepare_batch(inp, **kwargs)
            data_list.append(data)

        data_list = list(zip(*data_list))
        ret_list = []

        for data in data_list:
            data = [d.unsqueeze(0) for d in data]
            data = torch.cat(data)
            ret_list.append(data)

        return ret_list

    def merge_batches(self, batch_data):
        x, x_lens, ys_i, ys_t, ys_lens, xys_idx = batch_data
        n_devices = len(self.gpu_devices)
        sbatch_size = x.data.shape[1]

        xys_idx = xys_idx.chunk(n_devices)
        xys_idx = [xy_idx + i * sbatch_size for i, xy_idx in enumerate(xys_idx)]
        xys_idx = torch.cat(xys_idx)

        if not self.batch_first:
            ys_i = ys_i.transpose(0, 2, 1)
            ys_t = ys_t.transpose(0, 2, 1)
            ys_lens = ys_lens.transpose(0, 2, 1)
            xys_idx = xys_idx.transpose(0, 2, 1)

        data = [x, x_lens, ys_i, ys_t, ys_lens, xys_idx]
        x, x_lens, ys_i, ys_t, ys_lens, xys_idx = [torch.cat(d) for d in data]

        if not self.batch_first:
            ys_i = ys_i.transpose(1, 0)
            ys_t = ys_t.transpose(1, 0)
            ys_lens = ys_lens.transpose(1, 0)
            xys_idx = xys_idx.transpose(1, 0)

        data = [x, x_lens, ys_i, ys_t, ys_lens, xys_idx]
        data = [d.contiguous() for d in data]

        return data

    def prepare_batch(self, batch_data, volatile=False):
        x, x_lens, ys, ys_lens = batch_data
        batch_dim = 0 if self.batch_first else 1
        context_dim = 1 if self.batch_first else 0

        x_lens, x_idx = torch.sort(x_lens, 0, True)
        _, x_ridx = torch.sort(x_idx)
        ys_lens, ys_idx = torch.sort(ys_lens, batch_dim, True)

        x_ridx_exp = x_ridx.unsqueeze(context_dim).expand_as(ys_idx)
        xys_idx = torch.gather(x_ridx_exp, batch_dim, ys_idx)

        x = x[x_idx]
        ys = torch.gather(ys, batch_dim, ys_idx.unsqueeze(-1).expand_as(ys))

        x = Variable(x, volatile=volatile)
        x_lens = Variable(x_lens, volatile=volatile)
        ys_i = Variable(ys[..., :-1], volatile=volatile).contiguous()
        ys_t = Variable(ys[..., 1:], volatile=volatile).contiguous()
        ys_lens = Variable(ys_lens - 1, volatile=volatile)
        xys_idx = Variable(xys_idx, volatile=volatile)

        if self.is_cuda:
            x = x.cuda(async=True)
            x_lens = x_lens.cuda(async=True)
            ys_i = ys_i.cuda(async=True)
            ys_t = ys_t.cuda(async=True)
            ys_lens = ys_lens.cuda(async=True)
            xys_idx = xys_idx.cuda(async=True)

        return x, x_lens, ys_i, ys_t, ys_lens, xys_idx

    def calculate_loss(self, data, dec_logits):
        x, x_lens, ys_i, ys_t, ys_lens, xys_idx = data

        if self.batch_first:
            cdata = [ys_t, ys_lens, dec_logits]
            cdata = [d.transpose(1, 0).contiguous() for d in cdata]
            ys_t, ys_lens, dec_logits = cdata

        losses = []

        for logits, y, lens in zip(dec_logits, ys_t, ys_lens):
            loss = compute_loss(logits, y, lens)
            losses.append(loss)

        loss = sum(losses) / len(losses)
        return loss, losses

    def val_text(self, x_sents, yi_sents, yt_sents, o_sents):
        text = ""

        for x_sent, yi_sent, yt_sent, o_sent in \
                zip(x_sents, yi_sents, yt_sents, o_sents):
            text += "Encoder    Input: {}\n".format(x_sent)

            for i, (si, st, so) in enumerate(zip(yi_sent, yt_sent, o_sent)):
                text += "Decoder_{} Input:  {}\n".format(i, si)
                text += "Decoder_{} Target: {}\n".format(i, st)
                text += "Decoder_{} Output: {}\n".format(i, so)

        return text

    def val_sents(self, data, dec_logits):
        vocab, previews = self.model.vocab, self.previews
        x, x_lens, ys_i, ys_t, ys_lens, xys_idx = data

        if self.batch_first:
            cdata = [ys_i, ys_t, ys_lens, xys_idx, dec_logits]
            cdata = [d.transpose(1, 0).contiguous() for d in cdata]
            ys_i, ys_t, ys_lens, xys_idx, dec_logits = cdata

        _, xys_ridx = torch.sort(xys_idx, 1)
        xys_ridx_exp = xys_ridx.unsqueeze(-1).expand_as(ys_i)
        ys_i = torch.gather(ys_i, 1, xys_ridx_exp)
        ys_t = torch.gather(ys_t, 1, xys_ridx_exp)
        dec_logits = [torch.index_select(logits, 0, xy_ridx)
                      for logits, xy_ridx in zip(dec_logits, xys_ridx)]
        ys_lens = torch.gather(ys_lens, 1, xys_ridx)

        x, x_lens = x[:previews], x_lens[:previews]
        ys_i, ys_t = ys_i[:, :previews], ys_t[:, :previews]
        dec_logits = torch.cat(
            [logits[:previews].max(2)[1].squeeze(-1).unsqueeze(0)
             for logits in dec_logits], 0)
        ys_lens = ys_lens[:, :previews]

        ys_i, ys_t = ys_i.transpose(1, 0), ys_t.transpose(1, 0)
        dec_logits, ys_lens = dec_logits.transpose(1, 0), ys_lens.transpose(1,
                                                                            0)

        x, x_lens = x.data.tolist(), x_lens.data.tolist()
        ys_i, ys_t = ys_i.data.tolist(), ys_t.data.tolist()
        dec_logits, ys_lens = dec_logits.data.tolist(), ys_lens.data.tolist()

        def to_sent(data, length, vocab):
            return " ".join(vocab.i2f[data[i]] for i in range(length))

        def to_sents(data, lens, vocab):
            return [to_sent(d, l, vocab) for d, l in zip(data, lens)]

        x_sents = to_sents(x, x_lens, vocab)
        yi_sents = [to_sents(yi, y_lens, vocab) for yi, y_lens in
                    zip(ys_i, ys_lens)]
        yt_sents = [to_sents(yt, y_lens, vocab) for yt, y_lens in
                    zip(ys_t, ys_lens)]
        o_sents = [to_sents(dec_logit, y_lens, vocab)
                   for dec_logit, y_lens in zip(dec_logits, ys_lens)]

        return x_sents, yi_sents, yt_sents, o_sents

    def forward(self, inputs):
        return data_parallel(self.model, inputs,
                             device_ids=self.gpu_devices,
                             output_device=None,
                             dim=0,
                             module_kwargs=None)

    def step(self, step, batch_data, volatile=True, title=None):
        processed_data = self.prepare_batches(batch_data,
                                              chunks=len(self.gpu_devices),
                                              volatile=volatile)
        x, x_lens, ys_i, ys_t, ys_lens, xys_idx = processed_data
        inputs = (x, x_lens, ys_i, ys_lens, xys_idx)
        dec_logits, h = self.forward(inputs)
        merged_data = self.merge_batches(processed_data)
        loss, losses = self.calculate_loss(merged_data, dec_logits)
        losses_val = [l.data[0] for l in losses]
        loss_val = loss.data[0]

        plot_X = [step] * (self.model.n_decoders + 1)
        plot_Y = [loss_val] + losses_val

        self.logger.add_loss(title, **{
            t: p for t, p in zip(self.legend, plot_Y)
        })

        return merged_data, dec_logits, loss

    def step_val(self, step, batch_data):
        data, dec_logits, loss = self.step(step, batch_data,
                                           volatile=True,
                                           title="Validation Loss")
        sents = self.val_sents(data, dec_logits)
        text = self.val_text(*sents)

        self.logger.add_text("Validation Examples", text)

        return loss

    def step_train(self, step, batch_data):
        data, dec_logits, loss = self.step(step, batch_data,
                                           volatile=False,
                                           title="Training Loss")

        return loss

    def save(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)
        viz.save([self.save_dir])

    def train(self):
        optimizer = O.Adam([p for p in self.model.parameters()
                            if p.requires_grad])
        step = 0
        t = tqdm.tqdm()

        for epoch in range(self.n_epochs):
            for data in self.data_generator:
                step += 1

                self.model.zero_grad()
                optimizer.zero_grad()

                if step % self.val_period == 0:
                    loss = self.step_val(step, data)
                else:
                    loss = self.step_train(step, data)

                    loss.backward()
                    clip_grad_norm(self.model.parameters(), 3)
                    optimizer.step()

                loss_val = loss.data[0]

                if step % self.save_period == 0:
                    filename = "model-e{}-s{}-{:.4f}".format(
                        epoch, step, loss_val
                    )
                    self.save(filename)

                t.set_description("[{}|{}]: loss={:.4f}".format(
                    epoch, step, loss_val
                ))
                t.update()


def init_viz(args, kwargs):
    global viz

    viz = Visdom(*args, **kwargs)


def viz_run(f_name, args, kwargs):
    global viz

    getattr(viz, f_name).__call__(*args, **kwargs)


def file_list_reader(dir_or_paths, shuffle_files=False):
    if shuffle_files:
        import random
        random.shuffle(dir_or_paths)

    for x in dir_or_paths:
        if os.path.isfile(x):
            with open(x, "r") as f:
                for line in f:
                    yield line.strip()
        else:
            reader = DirectoryReader(x, shuffle_files=shuffle_files)
            for line in reader:
                yield line


def prod(*args):
    x = 1

    for a in args:
        x *= a

    return x


def count_parameters(model):
    counts = 0

    for param in model.parameters():
        if param.requires_grad:
            counts += prod(*param.size())

    return counts


class DataParallelSkipThoughts(MultiContextSkipThoughts):
    def __init__(self, *args, **kwargs):
        super(DataParallelSkipThoughts, self).__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        inputs = [d.squeeze(0) for d in inputs]
        return super(DataParallelSkipThoughts, self).forward(*inputs, **kwargs)


class TrainLogger(object):
    def __init__(self):
        self.step = 0

    def next(self):
        s = self.step
        self.step += 1

        return s

    def add_loss(self, prefix, **losses):
        raise NotImplementedError()

    def add_text(self, name, text):
        raise NotImplementedError()


class TensorboardTrainLogger(TrainLogger):
    def __init__(self, log_dir):
        super(TensorboardTrainLogger, self).__init__()

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def add_loss(self, prefix, **losses):
        for name, value in losses.items():
            name = "{}-{}".format(prefix, name)
            self.writer.add_scalar(name, value, self.next())

    def add_text(self, name, text):
        pass


class VisdomTrainLogger(TrainLogger):
    def __init__(self, viz_pool):
        super(VisdomTrainLogger, self).__init__()

        self.viz_pool = viz_pool

    def add_loss(self, name, **losses):
        self.viz_pool.apply_async(viz_run, ("plot", tuple(), dict(
            X=[self.next()] * len(losses),
            Y=[list(losses.values())],
            opts=dict(
                legend=list(losses.keys()),
                title=name
            )
        )))

    def add_text(self, name, text):
        self.viz_pool.apply_async(viz_run, ("code", tuple(), dict(
            text=text,
            opts=dict(
                title=name
            )
        )))


class DummyTrainLogger(TrainLogger):
    def add_loss(self, prefix, **losses):
        pass

    def add_text(self, name, text):
        pass


def main():
    args = parse_args()
    n_decoders = args.before + args.after + (1 if args.predict_self else 0)

    assert os.path.exists(args.vocab_path)

    logging.info("loading vocabulary...")
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_basename = timestamp + "-{}".format(args.name)
    save_dir = os.path.join(args.save_dir, save_basename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    logging.info("initializing model...")
    model_cls = DataParallelSkipThoughts
    model = model_cls(vocab, args.word_dim, args.hidden_dim,
                      reverse_encoder=args.reverse_encoder,
                      encoder_cell=args.encoder_cell,
                      decoder_cell=args.decoder_cell,
                      n_decoders=n_decoders,
                      n_layers=args.layers,
                      bidirectional=args.bidirectional,
                      dropout_prob=args.dropout_prob,
                      batch_first=args.batch_first)

    model.reset_parameters()
    n_params = count_parameters(model)
    logging.info("number of params: {}".format(n_params))

    logging.info("loading word embeddings...")
    assert args.wordembed_processes >= 1, \
        "number of processes must be larger than or equal to 1."

    if args.wordembed_processes > 1:
        def embedding_loader(path, word_dim):
            return load_embeddings_mp(path, word_dim,
                                      processes=args.wordembed_processes)
    else:
        embedding_loader = load_embeddings

    if args.wordembed_type == "glove":
        embeddings = embedding_loader(args.wordembed_path, model.word_dim)
        preinitialize_embeddings(model, vocab, embeddings)
    elif args.wordembed_type == "fasttext":
        fasttext_path = args.fasttext_path

        assert fasttext_path is not None, \
            "fasttext_path must specified when embed_type is fasttext."
        embeddings = load_fasttext_embeddings(vocab.words,
                                              fasttext_path,
                                              args.wordembed_path)
        preinitialize_embeddings(model, vocab, embeddings)
    elif args.wordembed_type == "none":
        pass
    else:
        raise ValueError("Unrecognized word embedding type: {}".format(
            args.wordembed_type
        ))

    if args.wordembed_freeze:
        model.embeddings.weight.requires_grad = False

    if args.visualizer is None:
        logger = DummyTrainLogger()
    elif args.visualizer == "tensorboard":
        logger = TensorboardTrainLogger(save_dir)
    elif args.visualizer == "visdom":
        viz_pool = mp.ThreadPool(1, initializer=init_viz, initargs=(tuple(), dict(
            buffer_size=args.visdom_buffer_size,
            server="http://{}".format(args.visdom_host),
            port=args.visdom_port,
            env=args.name,
            name=timestamp
        )))
        logger = VisdomTrainLogger(viz_pool)
    else:
        raise ValueError("Unrecognized visualizer type: {}".format(args.visualizer))

    logger.add_text("Arguments", str(args)[10:-1].replace(", ", "\n"))

    config_path = os.path.join(save_dir, os.path.basename(args.config))
    shutil.copy(args.config, config_path)

    logging.info("preparing training environment...")
    # Refer to torchtextutils.ContextDataGenerator
    batch_size = args.batch_size + args.before + args.after

    data_generator = DataGenerator(
        data_paths=args.data_path,
        vocab=vocab,
        omit_prob=args.omit_prob,
        swap_prob=args.swap_prob,
        batch_size=batch_size,
        max_len=args.max_len,
        n_before=args.before,
        n_after=args.after,
        predict_self=args.predict_self,
        shuffle_files=True,
        batch_first=args.batch_first,
        pin_memory=True,
        allow_residual=False
    )

    trainer = Trainer(
        model=model,
        gpu_devices=args.gpus,
        data_generator=data_generator,
        n_epochs=args.epochs,
        logger=logger,
        save_dir=save_dir,
        save_period=args.save_period,
        val_period=args.val_period,
        previews=args.previews,
        batch_first=args.batch_first
    )

    logging.info("training...")
    trainer.train()

    logging.info("done!")


if __name__ == '__main__':
    main()
