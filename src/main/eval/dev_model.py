import torch
import os
import argparse

from main.objects.Config import Config
from main.objects.Evaluator import Evaluator


def dev_model(config, exp_dir, dev_label_file=None, dev_output_file=None):
    dev_evaluator = Evaluator(config, 'dev', exp_dir, list_k=[1, 10, 50, 100], \
                            labeled_file=dev_label_file, output_file=dev_output_file, is_write=True)
    model = torch.load(os.path.join(exp_dir, "best_model"))

    model.eval()
    dev_evaluator.evaluate(model, train_num_batches=-1, train_loss=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_dir", required=True)
    parser.add_argument("-p", "--is_parallel", required=True)
    parser.add_argument("-s", "--shard")
    args = parser.parse_args()

    config_file = os.path.join(args.config_dir, "config.json")
    config = Config(config_file)

    data_dir = os.path.split(config.test_file)[0]

    if args.is_parallel == "True":
        dev_output_dir = os.path.join(args.config_dir, "dev_shards")
        if not os.path.exists(dev_output_dir):
            os.makedirs(dev_output_dir)
        dev_output_file = os.path.join(dev_output_dir, "shard_" + args.shard + ".pred")
        dev_label_file = os.path.join(data_dir, "dev_shards", "shard_" + args.shard)

    else:
        dev_output_file = os.path.join(args.config_dir, "dev.pred")
        dev_label_file = config.dev_file

    dev_model(config, args.config_dir, dev_label_file, dev_output_file)
