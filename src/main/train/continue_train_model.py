import datetime
import os
import sys
from shutil import copyfile, copytree
import argparse

import torch
import torch.optim as optim


from main.objects.Config import Config
from main.objects.Batcher import Batcher
from main.objects.Evaluator import Evaluator
from main.objects.Scorer import Scorer

from main.eval.test_model import test_model


from main.utils.util import make_directory, save_dict_to_json, make_exp_dir
from main.utils.model_helper import get_tokenizer, get_vocab, get_model




def continue_train_model(config, exp_dir):
    """ Continue training based on the given config, model / dataset
    
    param config: config object
    param dataset_name: name of dataset
    param model_name: name of model
    """

    tokenizer, max_len_token = get_tokenizer(config)
    vocab = get_vocab(config, tokenizer, max_len_token)
    model = torch.load(os.path.join(exp_dir, "best_model"))
    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                           weight_decay=config.l2penalty)   

    train_batcher = Batcher(config, 'train', tokenizer)
    dev_evaluator = Evaluator(config, vocab, tokenizer, 'dev', exp_dir, list_k=[5])

    best_score = 0
    best_score_iter = 0

    for train_num_batches, (qry_tk, pos_tk, neg_tk) in enumerate(train_batcher.get_train_batches()):
        optimizer.zero_grad()
        model.train()
        loss = model.compute_loss(qry_tk, pos_tk, neg_tk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

        if train_num_batches > 0 and train_num_batches % config.eval_every_minibatch == 0:
            model.eval()
            stop_train = dev_evaluator.evaluate(model, train_num_batches, float(loss))

            if stop_train:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_dir", required=True)
    args = parser.parse_args()

    config_file = os.path.join(args.config_dir, "config.json")
    config = Config(config_file)

    continue_train_model(config, args.config_dir)