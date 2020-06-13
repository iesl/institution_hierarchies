import argparse
import os


from main.objects.Config import Config
from main.objects.Evaluator import Evaluator

from main.objects.DataExtractor import DataExtractor
from main.objects.Tokenizer import Tokenizer

from main.utils.model_helper import get_baseline
from main.utils.util import make_exp_dir


def dev_model(config, exp_dir, dev_label_file=None, dev_output_file=None):
    test_evaluator = Evaluator(config, 'dev', exp_dir, list_k=[1, 10, 50, 100], \
        labeled_file=dev_label_file, output_file=dev_output_file, is_write=True)

    data_extractor = DataExtractor(config)
    tokenizer = Tokenizer(config)
    model = get_baseline(config, data_extractor, tokenizer)

    test_evaluator.evaluate(model, train_num_batches=-1, train_loss=-1, is_write=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", required=True)
    args = parser.parse_args()

    config = Config(args.config_file)
    config.update_dataset()
    exp_dir = make_exp_dir(config)

    dev_output_file = os.path.join(exp_dir, "dev.pred")
    dev_label_file = config.dev_file

    dev_model(config, exp_dir, dev_label_file, dev_output_file)
