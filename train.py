import torch.nn as nn
import math
import random
import numpy as np
import torch
import time
from layers.model import QGModel
from nmt_model import NMT
from optimizer import Optim
from instance import AlbertVocab
from log_utils import logger, init_logger, user_friendly_time_since
from data_load_utils import batch_iter, load_feat_vocab, load_bio_vocab, load_word_vocab, load_instances, load_config
from criterion import lloss, ulloss, ulloss_seq
from evaluate import evaluate_ppl, evaluate_bleu


def train(config, model, optim:Optim, train_instances, dev_instances, word_vocab, bio_vocab, feat_vocab):
    model.train()
    start_time = time.time()
    batch_num = 0
    num_trial = 0
    report_start_time = start_time
    report_loss, report_words_num = 0, 0
    for epoch in range(config['epoch']):
        for batch in batch_iter(train_instances, config['batch_size'],
                                                     word_vocab=word_vocab,
                                                     bio_vocab=bio_vocab,
                                                     feat_vocab=feat_vocab):
            logger.debug("src_tokens\n{}".format(batch.src))
            logger.debug("ans_tokens\n{}".format(batch.ans))
            batch_num += 1
            model.zero_grad()
            gen_output = model(batch)  # (tgt_len-1, B, vocab)
            gen_output = gen_output.transpose(0, 1).contiguous() # (B, tgt_len-1, vocab)
            lprobs = torch.log_softmax(gen_output, dim=-1)
            batch_size = gen_output.size(0)
            if config['max_out_cpy']:
                gold = torch.tensor([x[1:] for x in batch.tgt_extended_index], dtype=torch.long, device=model.device)
            else:
                gold = torch.tensor([x[1:] for x in batch.tgt_index], dtype=torch.long, device=model.device)
                # (B, tgt_len-1)
            batch_loss = lloss(lprobs, gold, ignore_index=word_vocab.pad_idx)
            if config['ulloss']:
                batch_loss += config['ulloss_weight'] * ulloss(lprobs, gold, ignore_index=word_vocab.pad_idx)
            if config['seq_ulloss'] and torch.rand(1).item() < config['seq_ulloss_rate']:
                batch_loss += ulloss_seq(lprobs, config['seq_ulloss_ngram'], config['seq_ulloss_seq_type'],
                                         mask_p=config['seq_ulloss_mask_p'])
            report_loss += batch_loss.item()
            report_words_num += sum(batch.tgt_len) - batch_size
            batch_loss.backward()
            optim.step()

            if batch_num % config['log_per_batches'] == 0:
                logger.info('epoch {}|batch {}|avg.loss {:.4f}|ppl {:.3f}|lr {}|t {}|total t {}'.format(
                    epoch,
                    batch_num,
                    report_loss/report_words_num,
                    math.exp(report_loss / report_words_num),
                    optim.lr,
                    user_friendly_time_since(report_start_time),
                    user_friendly_time_since(start_time)
                ))
                report_loss = report_words_num = 0
                report_start_time = time.time()

            if batch_num > config['start_validate_after_batches'] and batch_num % config['validate_per_batches'] == 0:
                ppl = evaluate_ppl(model, dev_instances,
                                           word_vocab=word_vocab,
                                           bio_vocab=bio_vocab,
                                           feat_vocab=feat_vocab)
                if optim.is_better(ppl):
                    model.save(config['model_save_path'])
                    logger.info("model saved!")
                hit_trial = optim.update_lr(ppl)
                optim.metric_history.append(ppl)
                logger.info('eval ppl {}|patience {}|current lr {}|best metric {}'.format(
                    ppl, optim.patience, optim.lr, optim.best_metric))
                if hit_trial:
                    num_trial += 1
                    logger.info("hit trial: [{}]".format(num_trial))
                    if num_trial >= config['max_num_trial']:
                        logger.info("early stop")
                        exit(0)
                    logger.info('restoring parameters')
                    state = torch.load(config['model_save_path'])
                    model.load_state_dict(state['model_state'])
                    model.to(device)
                import random
                test_instances = random.sample(train_instances, 100)
                bleus = evaluate_bleu(model, test_instances, config, word_vocab)
                logger.info("BLEU_1 {} BLEU_2 {} BLEU_3 {} BLEU_4 {} BLEU {}".format(*bleus))


def initialize_model(model):
    for pr_name, p in model.named_parameters():
        if 'albert_embeddings' in pr_name:
            p.requires_grad = False
        # p.data.uniform_(-opt.param_init, opt.param_init)
        elif 'rezero_alpha' in pr_name:
            logger.info('{} is rezero param'.format(pr_name))
            nn.init.zeros_(p)
        else:
            if p.dim() == 1:
                # p.data.zero_()
                p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
            else:
                nn.init.xavier_normal_(p, math.sqrt(3))
        logger.info("{}: requires_grad {}".format(pr_name, p.requires_grad))


if __name__ == '__main__':
    init_logger(level='info', log_file='train.log')
    config = load_config()
    device = torch.device('cpu') if config['gpu'] < 0 else torch.device('cuda:{}'.format(config['gpu']))
    logger.info("training with param:\n{}".format(config))
    logger.info("training with device: {}".format(device))
    if config['albert']:
        word_vocab = AlbertVocab(config['albert_model_name'], cache_dir=config['albert_cache_dir'])
    else:
        word_vocab = load_word_vocab('squad_out/train.txt.vocab.word', config['vocab_size'])
    logger.info(word_vocab)
    bio_vocab = load_bio_vocab('squad_out/train.txt.vocab.bio')
    logger.info(bio_vocab)
    feat_vocab = load_feat_vocab('squad_out/train.txt.vocab.feat')
    logger.info(feat_vocab)
    train_instances = load_instances('squad_out/train.ins')
    dev_instances = load_instances('squad_out/dev.ins')
    if config['model'] == 'nmt':
        model = NMT(word_vocab, bio_vocab, feat_vocab, config['word_embed_size'],
                    config['bio_embed_size'], config['feat_embed_size'],
                    config['hidden_size'],config['enc_bidir'], config['dropout'])
    else:
        model = QGModel(word_vocab, bio_vocab, feat_vocab, config['albert'], config['word_embed_size'],
                    config['bio_embed_size'], config['feat_embed_size'], config['hidden_size'],
                    config['dropout'], config['enc_bidir'], config['n_head'], config['max_out_cpy'],
                        albert_model_name=config['albert_model_name'], albert_cache_dir=config['albert_cache_dir'],
                        albert_word_embed_size=config['albert_word_embed_size'])
    model.to(device)
    initialize_model(model)
    optim = Optim(config['optim'], config['lr'], config['max_grad_norm'], config['max_weight_value'],
                  config['lr_decay'], config['lr_decay_patience'])
    optim.set_parameters(model.parameters())
    train(config, model, optim, train_instances, dev_instances,
          word_vocab, bio_vocab, feat_vocab)

