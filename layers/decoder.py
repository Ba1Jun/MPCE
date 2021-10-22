import torch.nn as nn
import torch
from torch_scatter import scatter_max
from log_utils import logger

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, attn_size, vocab_size, dropout, max_out_cpy):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.vocab_size = vocab_size
        self.max_out_cpy = max_out_cpy
        self.memory_attn_proj = nn.Linear(hidden_size, attn_size)
        self.gru_hidden_attn_proj = nn.Linear(hidden_size, attn_size, bias=False)
        self.engy_attn_proj = nn.Linear(attn_size, 1, bias=False)
        self.gru_input_proj = nn.Linear(embed_size + hidden_size, hidden_size)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.readout = nn.Linear(embed_size + hidden_size + hidden_size, hidden_size)
        self.readout_pooling = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.gen_proj = nn.Linear(hidden_size // 2, vocab_size)

    def concat_attn(self, query, key, value, mask):
        """
        :param query: (B, query_size)
        :param key: (B, src_len, attn_size)
        :param value: (B, src_len, value_size)
        :param mask: (B, src_len)
        :return:
        """
        query = self.gru_hidden_attn_proj(query).unsqueeze(1)  # (B, 1, attn_size)
        tmp = torch.add(key, query)  # (B, src_len, attn_size)
        e = self.engy_attn_proj(torch.tanh(tmp)).squeeze(2)  # (B, src_len)
        if mask is not None:
            e = e * mask + (1 - mask) * (-1000000)
        score = torch.softmax(e, dim=1)  # (B, src_len)
        logger.debug("\nattention score\n{}".format(score))
        ctxt = torch.bmm(score.unsqueeze(1), value).squeeze(1)  # (B, 1, value_size)
        return ctxt, score, e

    def forward(self, memory, memory_mask, src_extended_index, tgt, dec_init_hidden):
        """
        :param memory: (B, seq_len, hidden)
        :param memory_mask: (B, seq_len)
        :param src_extended_index: (B, seq_len)
        :param tgt: (tgt_len, B, embed)
        :param dec_init_hidden: (B, hidden)
        :return:
        """
        B, src_len, _ = memory.size()
        tgt = tgt[:-1]
        memory_for_attn = self.memory_attn_proj(memory)
        dec_hidden_tm1 = dec_init_hidden
        ctxt_tm1 = torch.zeros(B, self.hidden_size, device=memory.device)
        gen_probs = []
        atten_engy = []
        for tgt_tm1 in torch.split(tgt, 1, dim=0):
            tgt_tm1 = tgt_tm1.squeeze(0)
            gen_prob_t, dec_hidden_tm1, ctxt_tm1, atten_engy_t = self.decode_step(tgt_tm1,
                                                                    ctxt_tm1,
                                                                    dec_hidden_tm1,
                                                                    memory_for_attn,
                                                                    memory,
                                                                    src_extended_index,
                                                                    memory_mask)
            gen_probs.append(gen_prob_t)
            atten_engy.append(atten_engy_t)
        return torch.stack(gen_probs), torch.stack(atten_engy)  # (tgt_len, B, vocab_size), (tgt_len, B, src_len)

    def decode_step(self, tgt_tm1, ctxt_tm1, dec_hidden_tm1, memory_for_attn,
                    memory, src_extended_index=None, memory_mask=None):
        """
        :param tgt_tm1:
        :param ctxt_tm1:
        :param dec_hidden_tm1:
        :param memory_for_attn:
        :param memory:
        :param src_extended_index: (B, src_len)
        :param memory_mask:
        :return:
        """
        gru_input_tm1 = self.gru_input_proj(torch.cat([tgt_tm1, ctxt_tm1], dim=1))
        dec_hidden_t = self.gru_cell(gru_input_tm1, dec_hidden_tm1)
        ctxt_t, attn_score_t, attn_engy_t = self.concat_attn(dec_hidden_t, memory_for_attn, memory, memory_mask)
        r_t = self.readout(torch.cat([tgt_tm1, ctxt_t, dec_hidden_t], dim=-1))
        m_t = self.readout_pooling(r_t.unsqueeze(1)).squeeze(1)  # (B, hidden // 2)
        m_t = self.dropout(m_t)
        gen_t = self.gen_proj(m_t) # (B, vocab_size)
        final_gen_t = gen_t
        if self.max_out_cpy:
            oov_num = max(torch.max(src_extended_index).item() - self.vocab_size + 1, 0)
            placeholder = torch.zeros((gen_t.shape[0], oov_num), dtype=torch.float, device=gen_t.device)
            gen_t = torch.cat([gen_t, placeholder], dim=1)
            copy_t = torch.zeros_like(gen_t).fill_(-1000000)
            copy_t, _ = scatter_max(attn_engy_t, src_extended_index, dim=-1, out=copy_t)
            copy_t.masked_fill_(copy_t == -1000000, 0)
            final_gen_t = gen_t + copy_t
            copy_mask = torch.zeros_like(final_gen_t, dtype=torch.bool)
            copy_mask[:, self.vocab_size:] = 1
            final_gen_t = final_gen_t.masked_fill(((final_gen_t==0) & copy_mask), -1000000)
        return final_gen_t, dec_hidden_t, ctxt_t, attn_engy_t
