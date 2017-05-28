import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F
from torch.autograd import Variable


class SkipThoughts(nn.Module):
    def __init__(self, vocab, word_dim, hidden_dim, n_layers, n_decoders,
                 bidirectional=True, batch_first=True, dropout_prob=0.0,
                 rnn_cell="gru"):
        super(SkipThoughts, self).__init__()

        self.vocab = vocab
        self.vocab_size = vocab_size = len(vocab)
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

        self.rnn_cell = rnn_cell
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob
        self.n_layers = n_layers
        self.n_decoders = n_decoders
        self.bos_idx = vocab[vocab.bos]
        self.eos_idx = vocab[vocab.eos]
        self.pad_idx = vocab[vocab.pad]
        self.n_directions = 2 if bidirectional else 1
        self.cell_hidden_size = hidden_dim // self.n_directions // n_layers

        self.embeddings = nn.Embedding(vocab_size, word_dim,
                                       padding_idx=self.pad_idx)
        self.W_i = nn.Linear(word_dim, hidden_dim)
        self.W_o = nn.Linear(self.cell_hidden_size * self.n_directions,
                             word_dim)

        self.encoder = nn.GRU(input_size=hidden_dim,
                              hidden_size=self.cell_hidden_size,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              dropout=dropout_prob, batch_first=True)

        for i in range(n_decoders):
            setattr(self, "decoder{}".format(i), nn.GRU(input_size=hidden_dim,
                                    hidden_size=self.cell_hidden_size,
                                    num_layers=n_layers,
                                    bidirectional=bidirectional,
                                    dropout=dropout_prob, batch_first=True))

    def reset_parameters(self):
        I.normal(self.embeddings.weight.data, mean=0, std=0.01)

        self.W_i.reset_parameters()
        self.W_o.reset_parameters()
        self.encoder.reset_parameters()
        for i in range(self.n_decoders):
            getattr(self, "decoder{}".format(i)).reset_parameters()

    def compose_hidden_state(self, h, batch_size):
        c = h.data.new(self.n_layers * self.n_directions,
                       batch_size, self.cell_hidden_size).zero_()

        return (h, c)

    def encode_tv(self, x, x_lens):
        return self._encode(x, x_lens)

    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        x_lens, x_idx = torch.sort(x_lens, 0, descending=True)
        _, x_ridx = torch.sort(x_idx)

        x = torch.index_select(x, 0, x_idx)
        x_packed = R.pack_padded_sequence(x, x_lens.data.cpu().tolist(),
                                          batch_first=self.batch_first)

        if h is not None:
            h = torch.index_select(h, 1, x_idx)
            output, ch = cell(x_packed, h)
        else:
            output, ch = cell(x_packed)

        if self.rnn_cell == "lstm":
            h, c = ch
            h = torch.index_select(h, 1, x_ridx)
            c = torch.index_select(c, 1, x_ridx)

            ch = h, c
        elif self.rnn_cell == "gru":
            ch = torch.index_select(ch, 1, x_ridx)

        output, _ = R.pad_packed_sequence(output, batch_first=self.batch_first)

        if self.batch_first:
            output = torch.index_select(output, 0, x_ridx)
        else:
            output = torch.index_select(output, 1, x_ridx)

        return output, ch

    def _encode(self, x, x_lens):
        batch_size, seq_len = x.size()

        x = self.embeddings(x)
        x = x.view(-1, self.word_dim)
        x = F.tanh(self.W_i(x))
        x_enc = x.view(-1, seq_len, self.hidden_dim)

        _, h_n = self._run_rnn_packed(self.encoder, x_enc, x_lens)
        h_n = h_n.transpose(1, 0).contiguous().view(-1, self.hidden_dim)

        return h_n

    def _decode(self, h, xs, xs_lens):
        """Decode
        
        Args:
            x: [n_decoders, batch_size, seq_len] LongTensor
            x_lens: [n_decoders, batch_size] list
        """
        n_decoders, batch_size, seq_len = xs.size()

        h_0 = h.view(-1, self.n_layers * self.n_directions,
                     self.cell_hidden_size)
        h_0 = h_0.transpose(1, 0).contiguous()

        xs = xs.view(-1, seq_len)
        xs = self.embeddings(xs)
        xs = xs.view(-1, self.word_dim)
        xs = F.tanh(self.W_i(xs))
        xs = xs.view(self.n_decoders, batch_size, seq_len, self.hidden_dim)
        xs = [x.squeeze(0) for x in torch.split(xs, 1)]

        decoders = [getattr(self, "decoder{}".format(i))
                    for i in range(self.n_decoders)]
        hs = [self._run_rnn_packed(decoder, x, x_lens, h_0)[0]
              for decoder, x, x_lens in zip(decoders, xs, xs_lens)]
        seq_lens = [h.size(1) for h in hs]
        hs = [self.W_o(h.view(-1, self.n_directions * self.cell_hidden_size))
              for h in hs]

        W_e = self.embeddings.weight.t()
        logits = [torch.mm(h, W_e) for h in hs]
        logits = [logit.view(batch_size, seq_len, self.vocab_size)
                  for logit, seq_len in zip(logits, seq_lens)]

        return logits

    def forward(self, x, x_lens, ys, ys_lens):
        h = self._encode(x, x_lens)
        logits = self._decode(h, ys, ys_lens)

        return logits, h


def _sequence_mask(sequence_length, max_len=None):
    """
    Credit: jhchoi@github 
    """
    if max_len is None:
        max_len = sequence_length.data.max()

    batch_size = len(sequence_length)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand).cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def compute_loss(logits, targets, length):
    """
    Credit: jhchoi@github
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = targets.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*targets.size())
    # mask: (batch, max_len)
    mask = _sequence_mask(sequence_length=length, max_len=targets.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.sum().float()
    return loss
