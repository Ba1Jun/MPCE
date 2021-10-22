from collections import namedtuple
from transformers import AlbertTokenizer

SquadInstance = namedtuple('SquadInstance', ['src', 'tgt', 'bio', 'case', 'ner', 'pos', 'ans'])
SquadBatch = namedtuple('SquadBatch', ['src', 'tgt', 'ans', 'bio', 'case', 'ner', 'pos', 'oov_word',
                                       'src_len',  'src_index', 'src_extended_index',
                                       'tgt_len', 'tgt_index', 'tgt_extended_index',
                                       'ans_len','ans_index',
                                       'bio_index', 'case_index', 'ner_index', 'pos_index'])
PAD = '<PAD>'
UNK = '<UNK>'
SOS = '<S>'
EOS = '</S>'

class Vocab(object):

    def __init__(self, words, pad, unk, sos=None, eos=None):
        self.words = words
        self.word2id = {}
        for idx, word in enumerate(words):
            self.word2id[word] = idx
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.PAD = pad
        self.pad_idx = self.word2id[pad]
        assert pad in self.word2id and self.word2id[pad] == 0
        self.UNK = unk
        self.unk_idx = self.word2id[unk]
        assert unk in self.word2id
        self.SOS = sos
        if sos:
            assert sos in self.word2id
        self.EOS = eos
        if eos:
            assert eos in self.word2id

    def __contains__(self, item):
        return item in self.word2id

    def __getitem__(self, item):
        return self.word2id.get(item, self.unk_idx)

    def __len__(self):
        return len(self.word2id)

    def __str__(self):
        return "vocab size: {}".format(len(self))

    def index(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        elif type(sents) == list:
            return [self[w] for w in sents]
        else:
            return self[sents]

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def state_dict(self):
        return {
            'words': self.words,
            'pad': self.PAD,
            'unk': self.UNK,
            'eos': self.EOS,
            'sos': self.SOS
        }

    @staticmethod
    def load(state_dict):
        return Vocab(**state_dict)


class AlbertVocab(Vocab):
    def __init__(self, tokenizer_name, cache_dir=None):
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        words = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
        super(AlbertVocab, self).__init__(words, tokenizer.pad_token, tokenizer.unk_token,
                                          sos=tokenizer.bos_token, eos=tokenizer.eos_token)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        assert self.pad_idx == tokenizer.pad_token_id
        assert self.unk_idx == tokenizer.unk_token_id
        assert self[tokenizer.bos_token] == tokenizer.bos_token_id
        assert self[tokenizer.eos_token] == tokenizer.eos_token_id

    def __getitem__(self, item):
        return self.tokenizer.convert_tokens_to_ids(item)

    def __len__(self):
        return self.tokenizer.vocab_size

    def __str__(self):
        return "vocab size: {}".format(len(self))

    def index(self, sents):
        if type(sents[0]) == list:
            return [self.tokenizer.convert_tokens_to_ids(s) for s in sents]
        elif type(sents) == list:
            return self.tokenizer.convert_tokens_to_ids(sents)
        else:
            return self[sents]

    def convert_tokens_to_string(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def state_dict(self):
        return {
            'tokenizer_name': self.tokenizer_name,
            'cache_dir': self.cache_dir
        }

    @staticmethod
    def load(state_dict):
        return AlbertVocab(**state_dict)

if __name__ == '__main__':
    vocab = AlbertVocab('albert-base-v2', '/Users/feyoe/nlp_data/albert-base-v2')
    sents = ['He', 'is', 'a', 'good', 'boy', '<pad>', '<unk>']
    print(vocab.index(sents))
    state_dict = vocab.state_dict()
    new_vocab = AlbertVocab.load(state_dict)