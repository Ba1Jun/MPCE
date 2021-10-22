import torch
import torch.nn.functional as F
from typing import List, Any

def __pad(sents: List[List[Any]], pad_token: Any):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents: list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token: padding token
    @returns sents_padded: list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []
    sents_len = list(map(lambda x: len(x), sents))
    max_len = max(sents_len)
    for sent_len, sent in zip(list(sents_len), sents):
        sents_padded.append(sent + [pad_token] * (max_len - sent_len))

    return sents_padded


def lloss(lprobs, gold_index, ignore_index):
    """likelihood loss
    :param lprobs: (B, tgt_len, vocab_size)
    :param gold_index: (B, tgt_len)
    """
    vocab_size = lprobs.size(-1)
    return F.nll_loss(lprobs.view(-1, vocab_size), gold_index.view(-1), ignore_index=ignore_index, reduction='sum')

def ulloss(lprobs, target, ignore_index):
    """unlikelihood loss
    :param lprobs: (B, tgt_len, vocab_size)
    :param target: (B, tgt_len)
    """
    batch_size, tgt_len = lprobs.size(0), lprobs.size(1)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    target = target.view(-1)
    with torch.no_grad():
        # E.g. ABCCD, for token D, {A, B, C} are negtive target.
        # Make 'the triangle'.
        ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
        ctx_cands_ = (ctx_cands.tril(-1) + ignore_index)
        ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
        ctx_cands = ctx_cands.tril(-1) + ctx_cands_
        # Don't include the target for that timestep as a negative target.
        ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), ignore_index)
        # mask other batch
        for i in range(batch_size):
            cur_batch = slice(i*tgt_len, (i+1)*tgt_len)
            prev_batches = slice(0, i*tgt_len)
            next_batches = slice((i+1)*tgt_len, batch_size * tgt_len)
            ctx_cands[cur_batch, prev_batches] = ignore_index
            ctx_cands[cur_batch, next_batches] = ignore_index
        negative_targets = torch.zeros_like(lprobs).scatter_(1, ctx_cands, 1)

    # - compute loss
    one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)

    custom_loss = -torch.log(one_minus_probs) * negative_targets
    custom_loss = custom_loss.sum()
    return custom_loss

def ulloss_seq(lprobs, n_gram, seq_type='repeat', mask_p=0.1):
    pred_toks = torch.argmax(lprobs, dim=-1)  # (B, tgt_len)
    pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
    if seq_type == 'repeat':
        mask = ngram_repeat_mask(pred_toks, n_gram).type_as(lprobs)
    elif seq_type == 'random':
        mask = torch.bernoulli(torch.zeros_like(pred_toks, dtype=torch.float).fill_(mask_p))
    else:
        raise Exception("{} is not supported".format(seq_type))
    one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
    # (B, tgt_len)
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.sum()
    return loss

def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x)-n):
            ng = tuple(xl[j:j+n])
            if ng in seen:
                mask[i, j:j+n] = 1
            seen.add(ng)
    return mask
if __name__ == '__main__':
    batch_size = 3
    tgt_len = 6
    vocab_size = 20
    prob = torch.log_softmax(torch.rand((batch_size, tgt_len, vocab_size)), dim=-1)
    gold_index = torch.randint(0, vocab_size, (batch_size, tgt_len), dtype=torch.long)
    loss1 = lloss(prob, gold_index, ignore_index=0)
    loss2 = ulloss(prob, gold_index, ignore_index=1)
    print(loss1, loss2)




