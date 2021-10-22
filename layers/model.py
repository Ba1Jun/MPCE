from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch
from instance import SquadBatch, Vocab, AlbertVocab, SquadInstance
from layers.decoder import Decoder
from layers.encoder import Encoder
from layers.embedding import FeatureRichEmbedding, AlbertFeatureRichEmbedding
from log_utils import logger


class QGModel(nn.Module):
    def __init__(self, word_vocab: Vocab, bio_vocab: Vocab, feat_vocab: Vocab, albert: bool,
                 word_embed_size, bio_embed_size, feat_embed_size,
                 hidden_size, dropout, enc_bidir, n_head, max_out_cpy: bool, **kwargs
                 ):
        super(QGModel, self).__init__()
        self.word_vocab = word_vocab
        self.bio_vocab = bio_vocab
        self.feat_vocab = feat_vocab
        self.args = {
            'albert': albert,
            'word_embed_size': word_embed_size,
            'bio_embed_size': bio_embed_size,
            'feat_embed_size': feat_embed_size,
            'hidden_size': hidden_size,
            'dropout': dropout,
            'enc_bidir': enc_bidir,
            'n_head': n_head,
            'max_out_cpy': max_out_cpy
        }
        self.args.update(kwargs)
        if albert:
            self.embedding = AlbertFeatureRichEmbedding(kwargs['albert_model_name'],
                                                        len(bio_vocab), bio_embed_size, len(feat_vocab), feat_embed_size,
                                                        kwargs['albert_cache_dir'])
            decoder_word_embed_size = kwargs['albert_word_embed_size']
        else:
            self.embedding = FeatureRichEmbedding(len(word_vocab), word_embed_size,
                                              len(bio_vocab), bio_embed_size,
                                              len(feat_vocab), feat_embed_size)
            decoder_word_embed_size = word_embed_size
        self.encoder = Encoder(word_embed_size + bio_embed_size + feat_embed_size * 3, word_embed_size, hidden_size,
                               dropout, enc_bidir, n_head)
        self.decoder = Decoder(decoder_word_embed_size, hidden_size, hidden_size, len(word_vocab), dropout, max_out_cpy)

    def batch_to_tensor(self, batch: SquadBatch):
        src_indexes = [torch.tensor(x, dtype=torch.long, device=self.device) for x in
                       (batch.src_index, batch.bio_index, batch.case_index, batch.ner_index, batch.pos_index)]
        src_ext_index = torch.tensor(batch.src_extended_index, dtype=torch.long, device=self.device)
        src_len = torch.tensor(batch.src_len, dtype=torch.int, device=self.device)
        src_mask = self.generate_mask(src_len, src_indexes[0].size(1))
        tgt_index = torch.tensor(batch.tgt_index, dtype=torch.long, device=self.device).transpose(0, 1)
        ans_len = torch.tensor(batch.ans_len, dtype=torch.int, device=self.device)
        ans_index = torch.tensor(batch.ans_index, dtype=torch.long, device=self.device)
        ans_mask = self.generate_mask(ans_len, ans_index.size(1))
        return src_indexes, src_ext_index, src_len, src_mask, tgt_index, ans_len, ans_index, ans_mask

    def encode(self, batch):
        src_indexes, src_ext_index, src_len, src_mask, tgt_index, ans_len, ans_index, ans_mask = self.batch_to_tensor(
            batch)
        src_embed, ans_embed = self.embedding(src_indexes[0], src_mask, src_len, ans_index, ans_mask, ans_len,
                                              src_indexes[1],
                                              *src_indexes[2:])  # (B, src_len, embed_size)
        tgt_embed = self.embedding.word_embeddings(tgt_index)  # (tgt_len, B, embed_size)
        memory, dec_init_hidden = self.encoder(src_embed, src_mask, src_len, ans_embed)
        # dec_init_hidden: (B, hidden)
        return memory, dec_init_hidden, tgt_embed, src_mask, src_ext_index

    def forward(self, batch: SquadBatch):
        memory, dec_init_hidden, tgt_embed, src_mask, src_ext_index = self.encode(batch)
        logger.debug("memory mask shape {}".format(src_mask))
        logger.debug("memory shape {}".format(memory.shape))

        gen_output, atten_output = self.decoder(memory, src_mask, src_ext_index, tgt_embed, dec_init_hidden)
        # (tgt_len - 1, B, vocab + oov), not probability
        return gen_output

    def generate_mask(self, length, max_length):
        mask = torch.zeros(length.size(0), max_length, device=self.device)
        for i, x in enumerate(length):
            mask[i, :x] = 1
        return mask

    def beam_search(self, batch: SquadBatch, beam_size, max_decoding_step):
        """
        :param batch: batch size is 1
        :param beam_size:
        :return:
        """
        oov_word = batch.oov_word[0]
        memory, dec_init_hidden, tgt_embed, src_mask, src_ext_index = self.encode(batch)
        # memory: (B, src_len, hidden)
        no_copy_hypothesis = [[self.word_vocab.SOS]]
        copy_hypothesis = [[self.word_vocab.SOS]]
        atten_engy = [[]]
        hyp_scores = torch.zeros(len(copy_hypothesis), dtype=torch.float, device=self.device)
        completed_hypothesis = []
        t = 0
        ctxt_tm1 = torch.zeros(len(copy_hypothesis), self.args['hidden_size'], device=self.device)
        dec_hidden_tm1 = dec_init_hidden
        memory_for_attn = self.decoder.memory_attn_proj(memory)
        while len(completed_hypothesis) < beam_size and t < max_decoding_step:
            t += 1
            hyp_num = len(copy_hypothesis)
            prev_word = [x[-1] for x in copy_hypothesis]
            tgt_tm1 = self.embedding.word_embeddings(torch.tensor(self.word_vocab.index(prev_word),
                                                                  dtype=torch.long,
                                                                  device=self.device))  # (B, word_embed_size)

            memory_for_attn_tm1 = memory_for_attn.expand((hyp_num, *memory_for_attn.shape[1:]))
            memory_tm1 = memory.expand((hyp_num, *memory.shape[1:]))
            gen_t, dec_hidden_t, ctxt_t, atten_engy_t = self.decoder.decode_step(tgt_tm1, ctxt_tm1, dec_hidden_tm1,
                                                                   memory_for_attn_tm1, memory_tm1, src_ext_index)
            gen_t = torch.log_softmax(gen_t, dim=-1)  # (B, vocab)
            live_hyp_num = beam_size - len(completed_hypothesis)
            continuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(gen_t) + gen_t).view(-1)  # (hyp_num * V)
            top_candi_scores, top_candi_position = torch.topk(continuating_hyp_scores, k=live_hyp_num)
            prev_hyp_indexes = top_candi_position / gen_t.shape[-1]
            hyp_word_indexes = top_candi_position % gen_t.shape[-1]

            new_copy_hypothesis = []
            new_no_copy_hypothesis = []
            new_atten_engy = []
            live_hyp_index = []
            new_hyp_scores = []
            num_unk = 0
            for prev_hyp_index, hyp_word_index, new_hyp_score in zip(prev_hyp_indexes, hyp_word_indexes,
                                                                     top_candi_scores):
                prev_hyp_index = prev_hyp_index.item()
                hyp_word_index = hyp_word_index.item()
                new_hyp_score = new_hyp_score.item()

                if hyp_word_index < len(self.word_vocab):
                    hyp_word = self.word_vocab.id2word[hyp_word_index]
                    copy_new_hypo = copy_hypothesis[prev_hyp_index] + [hyp_word]
                    no_copy_new_hypo = no_copy_hypothesis[prev_hyp_index] + [hyp_word]
                else:
                    hyp_word = oov_word[hyp_word_index - len(self.word_vocab)]
                    copy_new_hypo = copy_hypothesis[prev_hyp_index] + [hyp_word]
                    no_copy_new_hypo = no_copy_hypothesis[prev_hyp_index] + ['[COPY]']
                new_atten_hypo = atten_engy[prev_hyp_index] + [atten_engy_t[prev_hyp_index, :]]
                if hyp_word == self.word_vocab.EOS:
                    completed_hypothesis.append((copy_new_hypo[1:-1], no_copy_new_hypo[1:-1],
                                                 torch.stack(new_atten_hypo[:-1]).tolist(),
                                                 new_hyp_score))
                else:
                    new_copy_hypothesis.append(copy_new_hypo)
                    new_no_copy_hypothesis.append(no_copy_new_hypo)
                    new_atten_engy.append(new_atten_hypo)
                    live_hyp_index.append(prev_hyp_index)
                    new_hyp_scores.append(new_hyp_score)
            if len(completed_hypothesis) == beam_size:
                break
            live_hyp_index = torch.tensor(live_hyp_index, dtype=torch.long, device=self.device)
            dec_hidden_tm1 = dec_hidden_t[live_hyp_index]
            ctxt_tm1 = ctxt_t[live_hyp_index]

            copy_hypothesis = new_copy_hypothesis
            no_copy_hypothesis = new_no_copy_hypothesis
            atten_engy = new_atten_engy
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
        has_completed = True
        if len(completed_hypothesis) == 0:
            has_completed = False
            completed_hypothesis.append((copy_hypothesis[0][1:], no_copy_hypothesis[0][1:],
                                         torch.stack(atten_engy[0]).tolist(),
                                         hyp_scores[0].item()))
        completed_hypothesis.sort(key=lambda x: x[3], reverse=True)
        return completed_hypothesis, has_completed

    def nucleus_sampling(self, batch: SquadBatch, max_decoding_step, nucleus_p=0.9):
        """
        :param batch: batch size is 1
        :param beam_size:
        :return:
        """
        oov_word = batch.oov_word[0]
        memory, dec_init_hidden, tgt_embed, src_mask, src_ext_index = self.encode(batch)
        # memory: (B, src_len, hidden)
        copy_hypothesis = [[self.word_vocab.SOS]]
        no_copy_hypothesis = [[self.word_vocab.SOS]]
        hyp_score = [0]
        has_completed = False
        t = 0
        ctxt_tm1 = torch.zeros(len(copy_hypothesis), self.args['hidden_size'], device=self.device)
        dec_hidden_tm1 = dec_init_hidden
        memory_for_attn = self.decoder.memory_attn_proj(memory)
        while t < max_decoding_step:
            t += 1
            hyp_num = len(copy_hypothesis)
            prev_word = [x[-1] for x in copy_hypothesis]
            tgt_tm1 = self.embedding.word_embeddings(torch.tensor(self.word_vocab.index(prev_word),
                                                                  dtype=torch.long,
                                                                  device=self.device))  # (B, word_embed_size)

            memory_for_attn_tm1 = memory_for_attn.expand((hyp_num, *memory_for_attn.shape[1:]))
            memory_tm1 = memory.expand((hyp_num, *memory.shape[1:]))
            gen_t, dec_hidden_t, ctxt_t = self.decoder.decode_step(tgt_tm1, ctxt_tm1, dec_hidden_tm1,
                                                                   memory_for_attn_tm1, memory_tm1, src_ext_index)
            sorted, sorted_indexes = torch.sort(gen_t, dim=-1, descending=True)
            sorted_p = torch.softmax(sorted, dim=-1)
            cum_p = torch.cumsum(sorted_p, dim=-1)
            is_greater = torch.gt(cum_p, nucleus_p)  # (B, V+extended)
            le_index = torch.min(is_greater, dim=-1)[1].item()
            new_prob = cum_p / cum_p[0, le_index]
            random_v = torch.rand(1).item()
            sampled_index = torch.gt(new_prob, random_v).min(-1)[1].item()

            hyp_word_score = sorted_p[0, sampled_index].item()
            hyp_word_index = sorted_indexes[0, sampled_index].item()
            if hyp_word_index < len(self.word_vocab):
                hyp_word = self.word_vocab.id2word[hyp_word_index]
                copy_hypothesis[0].append(hyp_word)
                no_copy_hypothesis[0].append((hyp_word))
            else:
                hyp_word = oov_word[hyp_word_index - len(self.word_vocab)]
                copy_hypothesis[0].append(hyp_word)
                no_copy_hypothesis[0].append('[COPY]')
            if hyp_word == self.word_vocab.EOS:
                has_completed = True
                break
            hyp_score[0] += hyp_word_score
            ctxt_tm1 = ctxt_t
            dec_hidden_tm1 = dec_hidden_t
        if has_completed:
            return [(copy_hypothesis[0][1:-1], no_copy_hypothesis[0][1:-1], hyp_score[0])], has_completed
        else:
            return [(copy_hypothesis[0][1:], no_copy_hypothesis[0][1:], hyp_score[0])], has_completed

    @property
    def device(self):
        return self.decoder.memory_attn_proj.weight.device

    def save(self, path):
        directory = Path(path).parent
        directory.mkdir(parents=True, exist_ok=True)
        state_dict = {'word_vocab': self.word_vocab.state_dict(),
                      'bio_vocab': self.bio_vocab.state_dict(),
                      'feat_vocab': self.feat_vocab.state_dict(),
                      'args': self.args,
                      'model_state': self.state_dict()}
        torch.save(state_dict, path)

    @staticmethod
    def load(path, device):
        params = torch.load(path, map_location=lambda storage, loc: storage)
        if params['args']['albert']:
            word_vocab = AlbertVocab.load(params['word_vocab'])
        else:
            word_vocab = Vocab.load(params['word_vocab'])
        bio_vocab = Vocab.load(params['bio_vocab'])
        feat_vocab = Vocab.load(params['feat_vocab'])
        model = QGModel(word_vocab, bio_vocab, feat_vocab, **params['args'])  # type:nn.Module
        model.load_state_dict(params['model_state'])
        return model.to(device)
