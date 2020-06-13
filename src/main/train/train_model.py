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
from main.objects.DataExtractor import DataExtractor
from main.objects.Tokenizer import Tokenizer

from main.utils.util import make_directory, save_dict_to_json, make_exp_dir
from main.utils.model_helper import get_model

from main.utils.util import device


def train_model(config, exp_dir):
    """ Train based on the given config, model / dataset
    
    param config: config object
    param dataset_name: name of dataset
    param model_name: name of model
    """


    data_extractor = DataExtractor(config)
    tokenizer = Tokenizer(config)
    model = get_model(config, data_extractor, tokenizer)
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                           weight_decay=config.l2penalty)
    train_batcher = Batcher(config, 'train')
    dev_evaluator = Evaluator(config, 'dev', exp_dir, list_k=[1, 10, 50, 100])


    for train_num_batches, batch_ids in enumerate(train_batcher.get_train_batches()):
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(batch_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

        if train_num_batches % 100 == 0:
            print("Num Batches: %d" % train_num_batches, end='\r')
            sys.stdout.flush()
        # Save model checkpoint
        if train_num_batches > 0 and train_num_batches % config.eval_every_minibatch == 0:
            model.eval()
            dev_evaluator.evaluate(model, train_num_batches, float(loss))
            # torch.save(model, os.path.join(exp_dir, "model_%d_batches" % exp_dir))

        if train_num_batches > config.num_batches:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", required=True)
    parser.add_argument("-g", "--grid_search", default="False")
    args = parser.parse_args()

    # Set up the config
    config = Config(args.config_file)
    config.update_dataset()

    # For non grid search, must set up exp dir
    if args.grid_search == "False":
        exp_dir = make_exp_dir(config)
        copytree(os.path.join(os.environ['SED_ROOT'], 'src'), os.path.join(exp_dir, 'src'))  
        config.save_config(exp_dir)
    else:
        exp_dir = os.path.split(args.config_file)[0]

    train_model(config, exp_dir)
