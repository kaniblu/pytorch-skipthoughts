import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F
from torch.autograd import Variable


class SkipThoughts(nn.Module):
    def __init__(self, vocab_size, word_dim, hidden_dim, n_layers, n_decoders,
                 bidirectional=True,
                 dropout_prob=0.0, pad_idx=0, bos_idx=1, eos_idx=2):
        super(SkipThoughts, self).__init__()

        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob
        self.n_layers = n_layers
        self.n_decoders = n_decoders
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.n_directions = 2 if bidirectional else 1
        self.cell_hidden_size = hidden_dim // self.n_directions // n_layers

        self.embeddings = nn.Embedding(vocab_size, word_dim,
                                       padding_idx=pad_idx)
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

    def _encode(self, x, x_lens):
        batch_size, seq_len = x.size()

        x = self.embeddings(x)
        x = x.view(-1, self.word_dim)
        x = F.tanh(self.W_i(x))
        x_enc = x.view(-1, seq_len, self.hidden_dim)

        x_packed = R.pack_padded_sequence(x_enc, x_lens, batch_first=True)

        _, h_n = self.encoder(x_packed)
        h_n = h_n.transpose(1, 0).contiguous().view(-1, self.hidden_dim)

        return h_n

    def _decode(self, h, xs, xs_lens, xs_idx):
        """Decode
        
        Args:
            x: [n_decoders, batch_size, seq_len] LongTensor
            x_lens: [n_decoders, batch_size] list
        """
        n_decoders, batch_size, seq_len = xs.size()

        h_0 = h.view(-1, self.n_layers * self.n_directions,
                     self.cell_hidden_size)
        hs_0 = [torch.index_select(h_0, 0, idx) for idx in xs_idx]
        hs_0 = [h_0.transpose(1, 0).contiguous() for h_0 in hs_0]

        xs = xs.view(-1, seq_len)
        xs = self.embeddings(xs)
        xs = xs.view(-1, self.word_dim)
        xs = F.tanh(self.W_i(xs))
        xs = xs.view(self.n_decoders, batch_size, seq_len, self.hidden_dim)
        xs = [x.squeeze(0) for x in torch.split(xs, 1)]

        packed_xs = [R.pack_padded_sequence(x, x_lens, batch_first=True)
                     for x, x_lens in zip(xs, xs_lens)]

        hs = [getattr(self, "decoder{}".format(i))(packed_x, h_0)[0] for i, (packed_x, h_0) in
              enumerate(zip(packed_xs, hs_0))]
        hs = [R.pad_packed_sequence(h, batch_first=True)[0] for h in hs]
        hs = [torch.cat([h, Variable(h.data.new(batch_size, seq_len - h.size(1), h.size(2)).zero_())], 1) if seq_len > h.size(1) else h for h in hs]
        hs = [h.unsqueeze(0) for h in hs]
        h = torch.cat(hs, 0)
        h = self.W_o(h.view(-1, self.n_directions * self.cell_hidden_size))

        W_e = self.embeddings.weight
        logits = torch.mm(h, W_e.t())
        logits = logits.view(n_decoders, batch_size, seq_len, self.vocab_size)

        return logits

    def forward(self, x, x_lens, ys, ys_lens, ys_idx):
        h = self._encode(x, x_lens)
        logits = self._decode(h, ys, ys_lens, ys_idx)

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
