import argparse
import os


from main.objects.Config import Config
from main.objects.Evaluator import Evaluator

from main.objects.DataExtractor import DataExtractor
from main.objects.Tokenizer import Tokenizer

from main.utils.model_helper import get_baseline
from main.utils.util import make_exp_dir


def test_model(config, exp_dir, test_label_file=None, test_output_file=None):
    test_evaluator = Evaluator(config, 'test', exp_dir, list_k=[1, 10, 50, 100], \
        labeled_file=test_label_file, output_file=test_output_file)

    data_extractor = DataExtractor(config)
    tokenizer = Tokenizer(config)
    model = get_baseline(config, data_extractor, tokenizer)

    test_evaluator.evaluate(model, train_num_batches=-1, train_loss=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", required=True)
    parser.add_argument("-f", "--fold")
    args = parser.parse_args()

    config = Config(args.config_file)
    exp_dir = os.path.dirname(args.config_file)

    test_output_file = os.path.join(exp_dir, "test_fold_%d.pred" % int(args.fold))
    test_label_file = os.path.join("data", "cross_validation", "fold_%d" % int(args.fold), "test.data")

    test_model(config, exp_dir, test_label_file, test_output_file)
