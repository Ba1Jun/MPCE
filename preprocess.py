import pathlib
import pickle
from typing import List
from collections import Counter
from tqdm import tqdm
from transformers import AlbertTokenizer
import json
from instance import SquadInstance, PAD, UNK, SOS, EOS
from data_load_utils import load_config
from log_utils import init_logger, logger


def collect_vocab(sents: List[List[str]], save_path, vocab_size=None):
    counter = Counter()
    for sent in sents:
        counter.update(sent)
    with open(save_path, 'w') as f:
        for word, _ in counter.most_common(vocab_size):
            f.write(word + '\n')

def __extract_answer_from_src_and_bio(src_tokens, bio_tokens):
    ans_tokens = []
    for idx, bio_token in enumerate(bio_tokens):
        if bio_token is not 'O':
            ans_tokens.append(src_tokens[idx])
    return ans_tokens

def collect_instances(source, target, pos, ner, bio, case, max_src_len, albert_tokenizer:AlbertTokenizer=None):
    file_paths = [source, target, pos, ner, bio, case]
    files = [open(x, 'r') for x in file_paths]
    lines = [x.readlines() for x in files]
    instances = []
    for src, tgt, pos, ner, bio, case in tqdm(zip(*lines), total=len(lines[0])):
        src_tokens = src.strip().split(' ')
        tgt_tokens = tgt.strip().split(' ')
        pos_tokens = pos.strip().split(' ')
        ner_tokens = ner.strip().split(' ')
        bio_tokens = bio.strip().split(' ')
        case_tokens = case.strip().split(' ')
        if albert_tokenizer:
            final_tokens = {'src': [], 'tgt': [], 'bio': [], 'case': [], 'ner': [], 'pos': []}
            for idx, src_token in enumerate(src_tokens):
                new_src_token_list = albert_tokenizer.tokenize(src_token)
                if len(new_src_token_list) > 0:
                    final_tokens['src'].extend(new_src_token_list)
                    final_tokens['bio'].extend([bio_tokens[idx]] * len(new_src_token_list))
                    final_tokens['case'].extend([case_tokens[idx]] * len(new_src_token_list))
                    final_tokens['ner'].extend([ner_tokens[idx]] * len(new_src_token_list))
                    final_tokens['pos'].extend([pos_tokens[idx]] * len(new_src_token_list))
                else:
                    print("zero: {} {}".format(src_token, new_src_token_list))
            for tgt_token in tgt_tokens:
                final_tokens['tgt'].extend(albert_tokenizer.tokenize(tgt_token))
        else:
            final_tokens = {'src': src_tokens, 'tgt': tgt_tokens, 'bio': bio_tokens,
                            'case': case_tokens, 'ner': ner_tokens, 'pos': pos_tokens}
        final_tokens['ans'] = __extract_answer_from_src_and_bio(final_tokens['src'], final_tokens['bio'])
        if len(final_tokens['src']) > max_src_len:
            logger.info("trimmed seq length {} to {}".format(len(final_tokens['src']), max_src_len))
        final_tokens['src'] = final_tokens['src'][:max_src_len]
        final_tokens['tgt'] = final_tokens['tgt'][:max_src_len]
        final_tokens['bio'] = final_tokens['bio'][:max_src_len]
        final_tokens['ner'] = final_tokens['ner'][:max_src_len]
        final_tokens['case'] = final_tokens['case'][:max_src_len]
        final_tokens['pos'] = final_tokens['pos'][:max_src_len]
        final_tokens['ans'] = final_tokens['ans'][:max_src_len]
        instance = SquadInstance(**final_tokens)
        instances.append(instance)
    return instances


if __name__ == '__main__':
    init_logger(level='debug')
    config = load_config()
    data_dir = config['data_dir']
    output_dir = config['save_dir']
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    albert_tokenizer = None
    if config['albert']:
        albert_tokenizer = AlbertTokenizer.from_pretrained(config['albert_model_name'], cache_dir=config['albert_cache_dir'])
    source_fmt = data_dir + "/{}.source.txt"
    target_fmt = data_dir + "/{}.target.txt"
    feat_fmt = data_dir + "/{}.{}"
    train_instances = collect_instances(source=source_fmt.format("train.txt"),
                                        target=target_fmt.format("train.txt"),
                                        pos=feat_fmt.format('train.txt', 'pos'),
                                        ner=feat_fmt.format('train.txt', 'ner'),
                                        bio=feat_fmt.format('train.txt', 'bio'),
                                        case=feat_fmt.format('train.txt', 'case'),
                                        max_src_len=config['max_seq_len'],
                                        albert_tokenizer=albert_tokenizer)
    pickle.dump(train_instances, open(output_dir + '/train.ins', 'wb'))
    word_vocab_save_path = output_dir + '/train.txt.vocab.word'
    collect_vocab([x.src + x.tgt for x in train_instances], word_vocab_save_path, config['vocab_size'])
    bio_vocab_save_path = output_dir + '/train.txt.vocab.bio'
    collect_vocab([x.bio for x in train_instances], bio_vocab_save_path)
    feat_vocab_save_path = output_dir + '/train.txt.vocab.feat'
    collect_vocab([x.ner + x.case + x.pos for x in train_instances], feat_vocab_save_path)
    dev_instances = collect_instances(source=source_fmt.format("dev.txt.shuffle.dev"),
                                      target=target_fmt.format("dev.txt.shuffle.dev"),
                                      pos=feat_fmt.format('dev.txt.shuffle.dev', 'pos'),
                                      ner=feat_fmt.format('dev.txt.shuffle.dev', 'ner'),
                                      bio=feat_fmt.format('dev.txt.shuffle.dev', 'bio'),
                                      case=feat_fmt.format('dev.txt.shuffle.dev', 'case'),
                                      max_src_len=config['max_seq_len'],
                                      albert_tokenizer=albert_tokenizer)
    pickle.dump(dev_instances, open(output_dir + '/dev.ins', 'wb'))
    test_instances = collect_instances(source=source_fmt.format("dev.txt.shuffle.test"),
                                       target=target_fmt.format("dev.txt.shuffle.test"),
                                       pos=feat_fmt.format('dev.txt.shuffle.test', 'pos'),
                                       ner=feat_fmt.format('dev.txt.shuffle.test', 'ner'),
                                       bio=feat_fmt.format('dev.txt.shuffle.test', 'bio'),
                                       case=feat_fmt.format('dev.txt.shuffle.test', 'case'),
                                       max_src_len=config['max_seq_len'],
                                       albert_tokenizer=albert_tokenizer
                                       )
    pickle.dump(test_instances, open(output_dir + '/test.ins', 'wb'))
