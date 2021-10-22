import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
from layers.attention import MultiHeadedAttention
from layers.rezero import RezeroConnection
class Encoder(nn.Module):
    def __init__(self, src_embed_size, ans_embed_size, hidden_size, dropout, bidir, n_head):
        super(Encoder, self).__init__()
        self.ans_pooling = nn.MaxPool1d(4)
        atten_input_size = src_embed_size
        gru_hidden_size = hidden_size // (2 if bidir else 1)
        self.bigru = nn.GRU(src_embed_size, gru_hidden_size, 1, batch_first=True, dropout=dropout, bidirectional=bidir)
        self.multi_atten = MultiHeadedAttention(n_head, hidden_size, hidden_size, hidden_size, hidden_size)
        self.rezero_connection = RezeroConnection()
        self.decoder_init_proj = nn.Linear(gru_hidden_size, hidden_size)


    def forward(self, src_embed:torch.Tensor, src_mask, src_len, ans_embed):
        """
        :param src_embed: (B, src_len, embed)
        :param src_mask: (B, src_len)
        :param src_len: (B,)
        :param ans_embed: (B, ans_len, embed)
        :return:
        """
        packed = pack_padded_sequence(src_embed, src_len, batch_first=True)
        packed_memory, last_hidden = self.bigru(packed)
        memory, _ = pad_packed_sequence(packed_memory, batch_first=True)
        atten_mem = self.rezero_connection(memory, lambda x: self.multi_atten(x, x, x, src_mask))
        dec_init_hidden = torch.tanh(self.decoder_init_proj(last_hidden[1]))
        return atten_mem, dec_init_hidden