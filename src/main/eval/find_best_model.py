import sys
import os
import torch
import argparse


from main.objects.Vocab import Vocab
from main.objects.Config import Config
from main.utils.util import setup_map_counter2gpu

import json



'''
Setsup a bash script to evaluate the best model in parallel with multiple GPU's on a slurm manager
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", required=True)
    args = parser.parse_args()


    best_score = 0
    best_config_idx = -1
    best_batch = -1

    # Loop through all configs for grid search
    for config_dirname in os.listdir(args.exp_dir):
        if "config_" in config_dirname:
            underscore_idx = config_dirname.rfind('_')
            config_idx = config_dirname[underscore_idx+1:]
            model_dir = os.path.join(args.exp_dir, config_dirname, "models")

            dev_pred_dir = os.path.join(args.exp_dir, config_dirname, "dev_pred")
            if not os.path.exists(dev_pred_dir):
                os.makedirs(dev_pred_dir)

            # Loop through every checkpoint model saved
            for model_filename in os.listdir(model_dir):
                # Get directory for current model
                underscore_idx = model_filename.rfind('_')
                num_batches = model_filename[underscore_idx+1:]

                score_file = os.path.join(args.exp_dir, config_dirname, "models", model_filename)

                with open(score_file, '') as f:
                    all_lines = f.readlines()
                    for line in all_lines:
                        print(line)
                        score_json = json.load(line)
                        score = score_json["map"]

                        if (float(score) > best_score):
                            best_score = float(score)
                            best_config = float(config_idx)
                            best_batch - float(num_batches)
                            best_line = line

    print(best_config, best_batch)


    with open(os.path.join(args.exp_dir, "best_model.txt")) as f:
        f.write("Config: %d Batch: %d" % (best_config, best_batch))