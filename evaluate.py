import json
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from typing import List
from nltk.translate import bleu_score
from data_load_utils import batch_iter, load_instances, load_config
from instance import SquadBatch, SquadInstance
from layers.model import QGModel
from nmt_model import NMT
from log_utils import logger, init_logger


def evaluate_ppl(model, instances, batch_size=32, **vocabs):
    """ Evaluate perplexity on dev sentences
    @Revised: *
    :param model : pytorch Model, must return loss directly
    :param dev_examples : list of squad examples
    :param batch_size (batch size)
    :returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()
    loss_crit = nn.CrossEntropyLoss(ignore_index=vocabs['word_vocab'].pad_idx, reduction='sum')
    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        batch: SquadBatch
        for batch in tqdm(batch_iter(instances, batch_size, **vocabs),
                          total=math.floor(len(instances) / batch_size)):
            gen_output = model(batch)
            gen_output = gen_output.transpose(0, 1).contiguous()  #(B, tgt_len-1, vocab)
            gold = torch.tensor([x[1:] for x in batch.tgt_index],
                                dtype=torch.long, device=model.device)  # (B, tgt_len-1)
            loss = loss_crit(gen_output.view(-1, gen_output.size(-1)), gold.view(-1))
            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(batch.tgt_len) - gen_output.size(0)
            cum_tgt_words += tgt_word_num_to_predict

        ppl = math.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def evaluate_bleu(model, instances:List[SquadInstance], config, word_vocab, predict_save_path=None,
                  predict_atten_engy_path=None):
    hypothesis = translate(model, instances, config, word_vocab, predict_save_path, predict_atten_engy_path)
    references = [[instance.tgt] for instance in instances]
    bleu1 = bleu_score.corpus_bleu(references, hypothesis, weights=[1, 0, 0, 0]) * 100
    bleu2 = bleu_score.corpus_bleu(references, hypothesis, weights=[0, 1, 0, 0]) * 100
    bleu3 = bleu_score.corpus_bleu(references, hypothesis, weights=[0, 0, 1, 0]) * 100
    bleu4 = bleu_score.corpus_bleu(references, hypothesis, weights=[0, 0, 0, 1]) * 100
    bleu = bleu_score.corpus_bleu(references, hypothesis) * 100
    return (bleu1, bleu2, bleu3, bleu4, bleu)

def translate(model, instances, config, word_vocab, predict_save_path=None,
              predict_atten_engy_path=None):
    """

    :param model:
    :param instances:
    :param beam_size:
    :param max_decode_step:
    :param vocabs:
    :return: List[List[str]], the translated result for each instance
    """
    was_training = model.training
    model.eval()

    vocabs = {
        'word_vocab': model.word_vocab,
        'bio_vocab': model.bio_vocab,
        'feat_vocab': model.feat_vocab
    }
    max_decode_step = config['max_decode_step']
    dec_method = config['dec_method']
    beam_size = config['beam_size']
    nucleus_p = config['nucleus_p']
    logger.info("translate using method {}".format(dec_method))
    copy_hypothesis = []
    no_copy_hypothesis = []
    atten_engy = []
    total_completed = 0
    with torch.no_grad():
        for batch in tqdm(batch_iter(instances, 1, shuffle=False, **vocabs), total=len(instances)):
            if dec_method == 'beam_search':
                instance_hypothesis, has_completed = model.beam_search(batch, beam_size, max_decode_step)
            elif dec_method == 'nucleus_sampling':
                instance_hypothesis, has_completed = model.nucleus_sampling(batch, max_decode_step, nucleus_p=nucleus_p)
            else:
                raise Exception("decoding method {} is not supported".format(dec_method))
            total_completed += int(has_completed)
            copy_hypothesis.append(instance_hypothesis[0][0])
            no_copy_hypothesis.append(instance_hypothesis[0][1])
            atten_engy.append(instance_hypothesis[0][2])
    if was_training:
        model.train(was_training)
    if predict_save_path:
        obj = []
        for idx, instance in enumerate(instances):
            obj.append({
                'idx': idx,
                'context': " ".join(instance.src),
                'ans': " ".join(instance.ans),
                'gold': " ".join(instance.tgt),
                'no_copy_predict': " ".join(no_copy_hypothesis[idx]),
                'predict': " ".join(copy_hypothesis[idx])
            })
        json.dump(obj, open(predict_save_path, 'w'), indent=2)
    if predict_atten_engy_path:
        obj = []
        for idx, (engy, instance, hypothesis) in enumerate(zip(atten_engy, instances, copy_hypothesis)):
            obj.append({
                'idx': idx,
                'decode_engy': str(engy),
                'src_tokens': ' '.join(instance.src),
                'output_tokens': ' '.join(hypothesis)
            })
        json.dump(obj, open(predict_atten_engy_path, 'w'), indent=2)
    logger.info("{} of {} is completed hypothesis".format(total_completed, len(instances)))
    return copy_hypothesis

if __name__ == '__main__':
    config = load_config()
    init_logger(log_file='evaluate.log')
    device = torch.device('cpu') if config['gpu'] < 0 else torch.device('cuda:{}'.format(config['gpu']))
    if config['model'] == 'nmt':
        model = NMT.load(config['model_save_path'])
        model.to(device)
    else:
        model = QGModel.load(config['model_save_path'], device)

    test_instances = load_instances(config['save_dir'] + '/test.ins')
    bleus = evaluate_bleu(model, test_instances, config, model.word_vocab, config['predict_save_path'])
    logger.info('\nBLEU_1: {}\nBLEU_2: {}\nBLEU_3: {}\nBLEU_4: {}\nBLEU :{}'.format(*bleus))

