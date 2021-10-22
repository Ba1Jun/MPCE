import random
import pickle
import json
import os
from typing import List, Any
from transformers import AlbertTokenizer
from instance import SquadBatch, SquadInstance, Vocab, PAD, UNK, SOS, EOS, AlbertVocab
from log_utils import logger

max_seq_len = 300


def load_config():
    config = json.load(open('config.json', 'r'))
    path_keys_to_expand = ['data_dir', 'save_dir', 'model_save_path', 'albert_cache_dir']
    for path_key in path_keys_to_expand:
        new_path = os.path.abspath(os.path.expanduser(config[path_key]))
        logger.debug("expand {} to {}".format(config[path_key], new_path))
        config[path_key] = new_path
    max_seq_len = config['max_seq_len']
    return config


def load_instances(path):
    instances = pickle.load(open(path, 'rb'))
    return instances


def load_word_vocab(path, vocab_size=20000):
    file = open(path, 'r')
    special = [PAD, UNK, SOS, EOS]
    words = [s.strip() for s in file.readlines()]
    for t in special:
        try:
            words.remove(t)
        except:
            pass
    special.extend(words)
    vocab = Vocab(words, pad=PAD, unk=UNK, sos=SOS, eos=EOS)
    return vocab


def load_bio_vocab(path):
    words = [PAD, 'O', 'B', 'I', UNK]
    vocab = Vocab(words, pad=PAD, unk=UNK)
    return vocab


def load_feat_vocab(path):
    file = open(path, 'r')
    special = [PAD, UNK]
    words = [s.strip() for s in file.readlines()]
    for t in special:
        try:
            words.remove(t)
        except:
            pass
    special.extend(words)
    vocab = Vocab(special, pad=PAD, unk=UNK)
    return vocab


def pad(sents: List[List[Any]], pad_token: Any):
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


def __process_instances_for_albert(instances: List[SquadInstance], albert_tokenizer: AlbertTokenizer):
    new_instances = []
    for instance in instances:
        src_limit = max_seq_len - 2
        if len(instance.src) > max_seq_len - 2:
            logger.info("src={} exceeds {}".format(len(instance.src), max_seq_len-2))
        src=[albert_tokenizer.bos_token] + instance.src[:src_limit] + [albert_tokenizer.eos_token]
        ans=instance.ans + [albert_tokenizer.eos_token]
        tgt=[albert_tokenizer.bos_token] + instance.tgt + [albert_tokenizer.eos_token]
        bio=['O'] + instance.bio[:src_limit] + ['O']
        ner=[PAD] + instance.ner[:src_limit] + [PAD]
        case=[PAD] + instance.case[:src_limit] + [PAD]
        pos=[PAD] + instance.pos[:src_limit] + [PAD]
        new_instances.append(SquadInstance(src=src, tgt=tgt, bio=bio, case=case, ner=ner, pos=pos, ans=ans))
    return new_instances


def __process_instances(instances):
    new_instances = []
    for instance in instances:
        new_instances.append(instance._replace(tgt=[SOS] + instance.tgt + [SOS]))
    return new_instances


def instances_to_batch(instances: List[SquadInstance], word_vocab, bio_vocab, feat_vocab):
    if isinstance(word_vocab, AlbertVocab):
       instances =  __process_instances_for_albert(instances, word_vocab.tokenizer)
    else:
        instances = __process_instances(instances)
    instances.sort(key=lambda x: len(x.src), reverse=True)
    data = dict(zip(SquadInstance._fields, zip(*instances)))
    data['ans_len'] = list(map(lambda t: len(t), data['ans']))
    data['ans'] = pad(data['ans'], word_vocab.PAD)
    data['ans_index'] = word_vocab.index(data['ans'])
    data['bio'] = pad(data['bio'], bio_vocab.PAD)
    data['bio_index'] = bio_vocab.index(data['bio'])
    for field in ('ner', 'pos', 'case'):
        data[field] = pad(data[field], feat_vocab.PAD)
        data[field + '_index'] = feat_vocab.index(data[field])
    data['src_len'] = list(map(lambda t: len(t), data['src']))
    data['src'] = pad(data['src'], word_vocab.PAD)
    data['tgt_len'] = list(map(lambda t: len(t), data['tgt']))
    data['tgt'] = pad(data['tgt'], word_vocab.PAD)
    src_index = []
    tgt_index = []
    src_extended_index = []
    tgt_extended_index = []
    oov_word = []
    for src, tgt in zip(data['src'], data['tgt']):
        this_oov_word = []
        this_src_index = []
        this_tgt_index = []
        this_src_extended_index = []
        this_tgt_extended_index = []
        for word in src:
            this_src_index.append(word_vocab.index(word))
            if word in word_vocab:
                this_src_extended_index.append(word_vocab.index(word))
            else:
                if word not in this_oov_word:
                    this_oov_word.append(word)
                this_src_extended_index.append(len(word_vocab) + this_oov_word.index(word))
        for word in tgt:
            this_tgt_index.append(word_vocab.index(word))
            if word in word_vocab or word not in this_oov_word:
                this_tgt_extended_index.append(word_vocab.index(word))
            else:
                this_tgt_extended_index.append(len(word_vocab) + this_oov_word.index(word))
        src_index.append(this_src_index)
        src_extended_index.append(this_src_extended_index)
        tgt_index.append(this_tgt_index)
        tgt_extended_index.append(this_tgt_extended_index)
        oov_word.append(this_oov_word)
    data['src_index'] = src_index
    data['tgt_index'] = tgt_index
    data['src_extended_index'] = src_extended_index
    data['tgt_extended_index'] = tgt_extended_index
    data['oov_word'] = oov_word
    return SquadBatch(**data)


def batch_iter(instances, batch_size, shuffle=True, **kwargs):
    if shuffle:
        random.shuffle(instances)
    x = 0
    while x < len(instances):
        yield instances_to_batch(instances=instances[x: x + batch_size], **kwargs)
        x += batch_size


if __name__ == '__main__':
    dev_instances = load_instances('squad_out/dev.ins')
    vocabs = {'word_vocab': load_word_vocab('squad_out/train.txt.vocab.word'),
              'bio_vocab': load_bio_vocab('squad_out/train.txt.vocab.bio'),
              'feat_vocab': load_feat_vocab('squad_out/train.txt.vocab.feat')}
    for x in batch_iter(dev_instances, 12, **vocabs):
        print(x)
        exit(0)
