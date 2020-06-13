import torch
import os
import argparse

def get_weights(config_dir):
    '''
    Get linear weights of model

    :param config_dir: config dir for model
    :return:
    '''
    model = torch.load(os.path.join(config_dir, "best_model"))
    print(model.lin_comb.weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_dir", required=True)
    args = parser.parse_args()

    get_weights(args.config_dir)