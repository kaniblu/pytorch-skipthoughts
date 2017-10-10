import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F
from torch.autograd import Variable

from torchsru import SRU


def pad_batch(x, n, after=True):
    pad_shape = x.data.shape[1:]
    pad = x.data.new(n, *pad_shape).zero_()
    pad = Variable(pad)
    l = [x, pad] if after else [pad, x]
    x = torch.cat(l)

    return x


def init_gru(cell, gain=1):
    cell.reset_parameters()

    # orthogonal initialization of recurrent weights
    for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            I.orthogonal(hh[i:i + cell.hidden_size], gain=gain)


def init_lstm(cell, gain=1):
    init_gru(cell, gain)

    # positive forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        l = len(ih_b)
        ih_b[l // 4:l // 2].data.fill_(1.0)
        hh_b[l // 4:l // 2].data.fill_(1.0)


def init_rnn_cell(cell, gain=1):
    if isinstance(cell, nn.LSTM):
        init_lstm(cell, gain)
    elif isinstance(cell, nn.GRU):
        init_gru(cell, gain)
    else:
        cell.reset_parameters()


def flatten_hidden_direction(h, directions=1):
    if directions < 2:
        return h
    else:
        layers = h.size(0) // directions
        batch_size, hidden_size = h.size()[1:]
        h = h.view(layers, directions, batch_size, hidden_size)
        h = h.permute(0, 2, 3, 1).contiguous()
        h = h.view(layers, batch_size, directions * hidden_size)

        return h.contiguous()


class CombinedCell(nn.Module):
    def __init__(self, rnn_cell_cls, input_size, hidden_size, num_layers,
                 dropout, batch_first):
        super(CombinedCell, self).__init__()

        self.rnn_cell_cls = rnn_cell_cls
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first

        params = dict(
            input_size=input_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first
        )

        self.uni_hidden_size = hidden_size // 2
        self.bi_hidden_size = hidden_size // 2 // 2

        self.uni_cell = rnn_cell_cls(
            bidirectional=False,
            hidden_size=self.uni_hidden_size,
            **params
        )

        self.bi_cell = rnn_cell_cls(
            bidirectional=True,
            hidden_size=self.bi_hidden_size,
            **params
        )

    def reset_parameters(self):
        init_rnn_cell(self.uni_cell)
        init_rnn_cell(self.bi_cell)

    def forward(self, *args, **kwargs):
        o_u, h_u = self.uni_cell(*args, **kwargs)
        o_b, h_b = self.bi_cell(*args, **kwargs)

        assert (isinstance(h_u, tuple) and isinstance(h_b, tuple)) or \
               (not isinstance(h_u, tuple) and not isinstance(h_b, tuple))

        if isinstance(h_u, tuple):
            # flatten direction
            h_b = [flatten_hidden_direction(x, 2) for x in h_b]
            h = tuple(torch.cat([x_u, x_b], 2) for x_u, x_b in zip(h_u, h_b))
        else:
            h_b = flatten_hidden_direction(h_b, 2)
            h = torch.cat([h_u, h_b], 2)

        assert (isinstance(o_u, R.PackedSequence) and isinstance(o_b, R.PackedSequence)) or \
               (not isinstance(o_u, R.PackedSequence) and not isinstance(o_b, R.PackedSequence))

        is_packed = isinstance(o_u, R.PackedSequence)

        if is_packed:
            o_u, lens_u = R.pad_packed_sequence(o_u, self.batch_first)
            o_b, _ = R.pad_packed_sequence(o_b, self.batch_first)
        else:
            lens_u = None

        o = torch.cat([o_u, o_b], 2)

        if is_packed:
            o = R.pack_padded_sequence(o, lens_u, self.batch_first)

        return o, h


class MultiContextSkipThoughts(nn.Module):
    @staticmethod
    def get_cell_cls(rnn_cell):
        if rnn_cell == "lstm":
            cell_cls = nn.LSTM
        elif rnn_cell == "gru":
            cell_cls = nn.GRU
        elif rnn_cell == "sru":
            cell_cls = SRU
        else:
            raise ValueError("Unrecognized rnn cell: {}".format(rnn_cell))

        return cell_cls

    def __init__(self, vocab, word_dim, hidden_dim, n_layers, n_decoders,
                 encoder_direction="uni",
                 batch_first=True,
                 dropout_prob=0.0,
                 reverse_encoder=False,
                 encoder_cell="gru",
                 decoder_cell="gru",
                 conditional_decoding=False):
        super(MultiContextSkipThoughts, self).__init__()

        self.is_cuda = False
        self.vocab = vocab
        self.vocab_size = vocab_size = len(vocab)
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.reverse_encoder = reverse_encoder
        self.conditional_decoding = conditional_decoding
        self.encoder_direction = encoder_direction

        if self.encoder_direction == "uni":
            self.bidirectional = False
        elif self.encoder_direction == "bi":
            self.bidirectional = True
        elif self.encoder_direction == "combine":
            self.bidirectional = None
        else:
            raise ValueError("Unrecognized direction: {}".format(
                self.encoder_direction
            ))

        self.encoder_cell_type = encoder_cell
        self.decoder_cell_type = decoder_cell
        self.batch_first = batch_first
        self.dropout_prob = dropout_prob
        self.n_layers = n_layers
        self.n_decoders = n_decoders
        self.bos_idx = vocab[vocab.bos]
        self.eos_idx = vocab[vocab.eos]
        self.pad_idx = vocab[vocab.pad]
        self.n_directions = 2 if self.bidirectional else 1
        self.encoder_hidden_size = hidden_dim // self.n_directions // n_layers
        self.decoder_hidden_size = hidden_dim // n_layers

        self.embeddings = nn.Embedding(vocab_size, word_dim,
                                       padding_idx=self.pad_idx)
        self.W_i = nn.Linear(word_dim, hidden_dim)
        self.W_o = nn.Linear(self.encoder_hidden_size * self.n_directions,
                             word_dim)

        enc_cls = self.get_cell_cls(encoder_cell)
        dec_cls = self.get_cell_cls(decoder_cell)

        encoder_params = dict(
            input_size=self.hidden_dim,
            hidden_size=self.encoder_hidden_size,
            num_layers=self.n_layers,
            dropout=dropout_prob,
            batch_first=True
        )

        if self.encoder_direction != "combine":
            encoder_params["bidirectional"] = self.bidirectional
            self.encoder = enc_cls(**encoder_params)
        else:
            self.encoder = CombinedCell(
                enc_cls, **encoder_params
            )

        if self.conditional_decoding:
            dec_input_size = hidden_dim * 2
        else:
            dec_input_size = hidden_dim

        for i in range(n_decoders):
            decoder = dec_cls(input_size=dec_input_size,
                              hidden_size=self.decoder_hidden_size,
                              num_layers=n_layers,
                              bidirectional=False,
                              dropout=dropout_prob,
                              batch_first=True)

            setattr(self, "decoder{}".format(i), decoder)

    def cuda(self, *args, **kwargs):
        ret = super(MultiContextSkipThoughts, self).cuda(*args, **kwargs)

        self.is_cuda = True

        return ret

    def cpu(self, *args, **kwargs):
        ret = super(MultiContextSkipThoughts, self).cpu(*args, **kwargs)

        self.is_cuda = False

        return ret

    def reset_parameters(self):
        I.normal(self.embeddings.weight.data, mean=0, std=0.01)
        I.xavier_normal(self.W_i.weight.data)
        I.xavier_normal(self.W_o.weight.data)

        init_rnn_cell(self.encoder)

        for i in range(self.n_decoders):
            decoder = getattr(self, "decoder{}".format(i))
            init_rnn_cell(decoder)

    def compose_decoder_hidden(self, cell_type, c, batch_size):
        if cell_type == "lstm":
            h = c.data.new(self.n_layers,
                           batch_size, self.decoder_hidden_size).zero_()
            h = Variable(h)

            return (h, c)
        elif cell_type == "gru":
            return c
        elif cell_type == "sru":
            return c
        else:
            raise ValueError("Unrecognized cell type: {}".format(
                cell_type
            ))

    def extract_hidden_state(self, cell_type, h):
        if cell_type == "lstm":
            return h[1]
        elif cell_type == "gru":
            return h
        elif cell_type == "sru":
            return h
        else:
            raise ValueError("Unrecognized cell type: {}".format(
                cell_type
            ))

    def encode(self, x, x_lens):
        x = self.embeddings(x)
        return self._encode_embed(x, x_lens)

    def encode_embed(self, x, x_lens):
        """Encodes raw word embeddings

        Arguments:
            x: [batch_size, seq_len, word_dim] FloatTensor
            x_lens: [batch_size] LongTensor
        """
        return self._encode_embed(x, x_lens)

    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        x_packed = R.pack_padded_sequence(x, x_lens,
                                          batch_first=self.batch_first)

        # Following line does not improve memory usage or computation efficiency
        # as claimed by pyTorch warning messages
        # cell.flatten_parameters()

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=self.batch_first)

        return output, h

    def reverse_sequence(self, x, x_lens):
        batch_size, seq_len, word_dim = x.size()

        inv_idx = Variable(torch.arange(seq_len - 1, -1, -1).long())
        shift_idx = Variable(torch.arange(0, seq_len).long())

        if x.is_cuda:
            inv_idx = inv_idx.cuda(x.get_device())
            shift_idx = shift_idx.cuda(x.get_device())

        inv_idx = inv_idx.unsqueeze(0).unsqueeze(-1).expand_as(x)
        shift_idx = shift_idx.unsqueeze(0).unsqueeze(-1).expand_as(x)
        shift = (seq_len + (-1 * x_lens)).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        shift_idx = shift_idx + shift
        shift_idx = shift_idx.clamp(0, seq_len - 1)

        x = x.gather(1, inv_idx)
        x = x.gather(1, shift_idx)

        return x

    def _encode_embed(self, x, x_lens):
        batch_size, seq_len, word_embed = x.size()

        if self.reverse_encoder:
            x = self.reverse_sequence(x, x_lens)

        x = x.view(-1, self.word_dim)
        x = F.tanh(self.W_i(x))
        x_enc = x.view(-1, seq_len, self.hidden_dim)

        _, h_n = self._run_rnn_packed(self.encoder, x_enc, x_lens.data.tolist())
        h_n = self.extract_hidden_state(self.encoder_cell_type, h_n)
        h_n = h_n.transpose(1, 0).contiguous().view(-1, self.hidden_dim)

        return h_n

    def _decode(self, dec_idx, h, x, x_lens):
        h_enc = h
        batch_size, seq_len = x.size()

        decoder = getattr(self, "decoder{}".format(dec_idx))

        assert decoder is not None

        h = h.view(-1, self.n_layers, self.decoder_hidden_size)
        h = h.transpose(1, 0).contiguous()
        h = self.compose_decoder_hidden(self.decoder_cell_type, h, batch_size)

        x = self.embeddings(x)
        x = x.view(-1, self.word_dim)
        x = F.tanh(self.W_i(x))
        x = x.view(batch_size, seq_len, self.hidden_dim)

        if self.conditional_decoding:
            h_exp = h_enc.unsqueeze(1).expand_as(x)
            x = torch.cat([x, h_exp], 2)

        o, h = self._run_rnn_packed(decoder, x, x_lens.data.tolist(), h)

        if o.size(1) < seq_len:
            pad = o.data.new(o.size(0), seq_len - o.size(1), o.size(2)).fill_(0)
            pad = Variable(pad)
            o = torch.cat([o, pad], 1)

        o = o.contiguous()
        o = o.view(-1, self.decoder_hidden_size)
        o = self.W_o(o)
        logits = torch.mm(o, self.embeddings.weight.t())
        logits = logits.view(batch_size, seq_len, self.vocab_size)

        return logits

    def forward(self, x, x_lens, ys, ys_lens, xys_idx):
        x = self.embeddings(x)
        h = self._encode_embed(x, x_lens)

        if self.batch_first:
            ys = ys.transpose(1, 0)
            ys_lens = ys_lens.transpose(1, 0)
            xys_idx = xys_idx.transpose(1, 0)

        logits_list = []

        for dec_idx, (y, y_lens, xy_idx) in enumerate(
                zip(ys, ys_lens, xys_idx)):
            h_dec = torch.index_select(h, 0, xy_idx)
            logits = self._decode(dec_idx, h_dec, y, y_lens)

            nil_batches = len(h_dec) - len(logits)
            if nil_batches:
                logits = pad_batch(logits, nil_batches, True)

            logits_list.append(logits.unsqueeze(0))

        logits = torch.cat(logits_list)

        if self.batch_first:
            logits = logits.transpose(1, 0)

        return logits, h


class SkipThoughts(MultiContextSkipThoughts):
    def __init__(self, *args, **kwargs):
        super(SkipThoughts, self).__init__(*args, n_decoders=2, **kwargs)


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = Variable(ranges)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


def compute_loss(logits, y, lens):
    batch_size, seq_len, vocab_size = logits.size()
    logits = logits.view(batch_size * seq_len, vocab_size)
    y = y.view(-1)

    logprobs = F.log_softmax(logits)
    losses = -torch.gather(logprobs, 1, y.unsqueeze(-1))
    losses = losses.view(batch_size, seq_len)
    mask = sequence_mask(lens, seq_len).float()
    losses = losses * mask
    loss_batch = losses.sum() / len(lens)
    loss_step = losses.sum() / lens.sum().float()

    return loss_batch, loss_step