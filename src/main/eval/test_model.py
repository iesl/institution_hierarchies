import torch
import os
import argparse


from main.objects.Config import Config
from main.objects.Evaluator import Evaluator



def test_model(config, exp_dir, test_label_file=None, test_output_file=None):
    test_evaluator = Evaluator(config, 'test', exp_dir, list_k=[1, 10, 50, 100], \
        labeled_file=test_label_file, output_file=test_output_file)
    model = torch.load(os.path.join(exp_dir, "best_model"))

    model.eval()
    test_evaluator.evaluate(model, train_num_batches=-1, train_loss=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_dir", required=True)
    parser.add_argument("-p", "--is_parallel", required=True)
    parser.add_argument("-s", "--shard")
    args = parser.parse_args()

    config_file = os.path.join(args.config_dir, "config.json")
    config = Config(config_file)

    if args.is_parallel == "True":
        data_dir = os.path.split(config.test_file)[0]
        test_output_dir = os.path.join(args.config_dir, "test_shards")
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)        
        test_output_file = os.path.join(test_output_dir, "shard_" + args.shard + ".pred")
        test_label_file = os.path.join(data_dir, "test_shards", "shard_" + args.shard)

    else:
        test_output_file = os.path.join(args.config_dir, "test.pred")
        test_label_file = config.test_file

    test_model(config, args.config_dir, test_label_file, test_output_file)