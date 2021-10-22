#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from layers.embedding import FeatureRichEmbedding
from layers.encoder import Encoder
from layers.decoder import Decoder
from instance import SquadBatch, Vocab, SquadInstance

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    def __init__(self, word_vocab:Vocab, bio_vocab:Vocab, feat_vocab:Vocab,
                 word_embed_size, bio_embed_size, feat_embed_size,
                 hidden_size, enc_bidir, dropout=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.word_vocab = word_vocab
        self.bio_vocab = bio_vocab
        self.feat_vocab = feat_vocab
        self.args = {
            'word_embed_size': word_embed_size,
            'bio_embed_size': bio_embed_size,
            'feat_embed_size': feat_embed_size,
            'hidden_size': hidden_size,
            'enc_bidir': enc_bidir,
            'dropout': dropout
        }
        self.embedding = FeatureRichEmbedding(len(word_vocab), word_embed_size,
                                              len(bio_vocab), bio_embed_size,
                                              len(feat_vocab), feat_embed_size)
        self.encoder = Encoder(word_embed_size + bio_embed_size + feat_embed_size * 3, hidden_size, dropout, enc_bidir)
        self.decoder_init_hidden_proj = nn.Linear(self.encoder.hidden_size, hidden_size)
        self.decoder = Decoder(word_embed_size, hidden_size, hidden_size, len(word_vocab), dropout)
        ### END YOUR CODE

    def batch_to_tensor(self, batch:SquadBatch):
        src_indexes = [torch.tensor(x, dtype=torch.long, device=self.device).transpose(0, 1) for x  in
                   (batch.src_index, batch.bio_index, batch.case_index, batch.ner_index, batch.pos_index)]
        src_len = torch.tensor(batch.src_len, dtype=torch.int, device=self.device)
        src_mask = self.generate_mask(src_len, src_indexes[0].size(0))
        tgt_index = torch.tensor(batch.tgt_index, dtype=torch.long, device=self.device).transpose(0, 1)
        return src_indexes, src_len, src_mask, tgt_index

    def forward(self, batch:SquadBatch) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        src_indexes, src_len, src_mask, tgt_index = self.batch_to_tensor(batch)
        src_embed = self.embedding(*src_indexes)
        memory , last_hidden = self.encoder(src_embed, src_len)
        memory = memory.transpose(0, 1)
        dec_init_hidden = torch.tanh(self.decoder_init_hidden_proj(last_hidden))
        tgt_embed = self.embedding.word_embeddings(tgt_index)  # (tgt_len, B, embed_size)
        gen_output = self.decoder(memory, src_mask, tgt_embed, dec_init_hidden)  # (tgt_len-1, B, hidden)
        return gen_output

    def generate_mask(self, length, max_length):
        mask = torch.zeros(length.size(0), max_length, dtype=torch.int, device=self.device)
        for i, x in enumerate(length):
            mask[i, x:] = 1
        return mask

    def beam_search(self, batch, beam_size, max_decoding_step):
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_indexes, src_len, src_mask, tgt_index = self.batch_to_tensor(batch)
        src_embed = self.embedding(*src_indexes)

        memory , last_hidden = self.encoder(src_embed, src_len)
        memory = memory.transpose(0, 1)
        memory_for_attn = self.decoder.memory_attn_proj(memory)

        dec_hidden_tm1 = torch.tanh(self.decoder_init_hidden_proj(last_hidden))
        ctxt_tm1 = torch.zeros_like(dec_hidden_tm1, device=self.device)


        hypotheses = [[self.word_vocab.SOS]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_step:
            t += 1
            hyp_num = len(hypotheses)

            memory_tm1 = memory.expand(hyp_num,
                                                     memory.size(1),
                                                     memory.size(2))

            memory_for_attn_tm1 = memory_for_attn.expand(hyp_num,
                                                         memory_for_attn.size(1),
                                                        memory_for_attn.size(2))

            prev_word = torch.tensor([self.word_vocab[hyp[-1]] for hyp in hypotheses], dtype=torch.long,
                                 device=self.device)  # (hpy_num, )
            tgt_tm1 = self.embedding.word_embeddings(prev_word)  # (hpy_num, e)

            gen_t, dec_hidden_t, ctxt_t = self.decoder.decode_step(tgt_tm1, ctxt_tm1, dec_hidden_tm1,
                                                memory_for_attn_tm1, memory_tm1)

            # log probabilities over target words
            log_p_t = F.log_softmax(gen_t, dim=-1)  # (hpy_num, vocab)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(
                -1)  # (hpy_num * src_len)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.word_vocab)
            hyp_word_ids = top_cand_hyp_pos % len(self.word_vocab)
            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.word_vocab.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == self.word_vocab.EOS:
                    completed_hypotheses.append((new_hyp_sent[1:-1], cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            dec_hidden_tm1 = dec_hidden_t[live_hyp_ids]
            ctxt_tm1 = ctxt_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        has_comp = True
        if len(completed_hypotheses) == 0:
            has_comp = False
            completed_hypotheses.append((hypotheses[0][1:], hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp[1], reverse=True)

        return completed_hypotheses, has_comp

    @property
    def device(self):
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.decoder_init_hidden_proj.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        word_vocab = Vocab.load(params['word_vocab'])
        bio_vocab = Vocab.load(params['bio_vocab'])
        feat_vocab = Vocab.load(params['feat_vocab'])
        model = NMT(word_vocab, bio_vocab, feat_vocab, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': self.args,
            'state_dict': self.state_dict()
        }
        params['word_vocab'] = self.word_vocab.state_dict()
        params['bio_vocab'] = self.bio_vocab.state_dict()
        params['feat_vocab'] = self.feat_vocab.state_dict()

        torch.save(params, path)