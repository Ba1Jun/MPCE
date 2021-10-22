import torch.nn as nn
import torch
from transformers import AlbertModel


class FeatureRichEmbedding(nn.Module):
    def __init__(self, word_vocab_size, word_embed_size,
                       bio_vocab_size, bio_embed_size,
                       feat_vocab_size, feat_embed_size):
        super(FeatureRichEmbedding, self).__init__()
        self.word_embed_size = word_embed_size
        self.bio_embed_size = bio_embed_size
        self.feat_embed_size = feat_embed_size
        self.word_embeddings = nn.Embedding(word_vocab_size, word_embed_size)
        self.bio_embeddings = nn.Embedding(bio_vocab_size, bio_embed_size)
        self.feat_embeddings = nn.Embedding(feat_vocab_size, feat_embed_size)

    def forward(self, word_index, bio_index, *feat_index, **kwargs):
        word = self.word_embeddings(word_index)
        bio = self.bio_embeddings(bio_index)
        feat = torch.cat([self.feat_embeddings(index) for index in feat_index], dim=-1)
        return torch.cat([word, bio, feat], dim=-1)

class AlbertFeatureRichEmbedding(nn.Module):
    def __init__(self, albert_model_name,
                       bio_vocab_size, bio_embed_size,
                       feat_vocab_size, feat_embed_size,
                       albert_cache_dir=None):
        super(AlbertFeatureRichEmbedding, self).__init__()
        self.bio_embed_size = bio_embed_size
        self.feat_embed_size = feat_embed_size
        self.albert_embeddings = AlbertModel.from_pretrained(albert_model_name, cache_dir=albert_cache_dir)
        self.bio_embeddings = nn.Embedding(bio_vocab_size, bio_embed_size)
        self.feat_embeddings = nn.Embedding(feat_vocab_size, feat_embed_size)

    def forward(self, src_index, src_mask, src_len, ans_index, ans_mask, ans_len, bio_index, *feat_index, **kwargs):
        # word_index = torch.cat([src_index, ans_index], dim=1) #(B, src_len + ans_len)
        # attention_mask = torch.cat([src_mask, ans_mask], dim=1) # (B, src_len + ans_len)
        # token_type_ids = torch.cat([torch.zeros_like(src_mask, dtype=torch.long, device=src_index.device),
        #                             torch.ones_like(ans_mask, dtype=torch.long, device=src_index.device)], dim=1)
        word = self.albert_embeddings(src_index, attention_mask=src_mask)[0]
        # max_src_len = src_index.size(1)
        # max_ans_len = ans_index.size(1)
        # src, ans = torch.split(word, [max_src_len, max_ans_len], dim=1)

        bio = self.bio_embeddings(bio_index)
        feat = torch.cat([self.feat_embeddings(index) for index in feat_index], dim=-1)
        return torch.cat([word, bio, feat], dim=-1), None

    @property
    def word_embeddings(self):
        return self.albert_embeddings.embeddings.word_embeddings