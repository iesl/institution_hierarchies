import sys 
import json
import os

import argparse

def cross_validation_eval(exp_dir, gpu_type):
    '''
    Loops through configuration in experiment directory and finds the configuration with the highest dev score 
    '''
    bash_script = os.path.join(exp_dir, "eval_random_seeds.sh")

    with open(bash_script, 'w') as write_file:
        write_file.write("#!/usr/bin/env bash \n")

        num_folds = 5
        num_config = 8

        for i in range(num_folds):
            best_score = 0
            best_config_dir = 0

            for j in range(num_config):
                config_idx = i * num_config + j

                config_dir = os.path.join(exp_dir, "config_" + str(config_idx))
                if os.path.exists(config_dir):
                    dev_scores_json = os.path.join(config_dir, "dev_scores.json")
                    if (os.path.exists(dev_scores_json)):
                        with open(dev_scores_json) as dev_file:

                            all_lines = dev_file.readlines()
                            for line in all_lines:
                                score_json = json.loads(line)
                                score = score_json["map"]

                                if (float(score) > best_score):
                                    best_score = float(score)
                                    best_config_dir = config_dir


            error_file = os.path.join(best_config_dir, "test_error")
            output_file = os.path.join(best_config_dir, "test_output")

            command = "sbatch --partition={} --gres=gpu:1 --mem=30G --error {} --output {} bin/run/test_model.sh {} \n".format(\
                gpu_type, error_file, output_file, best_config_dir)
            write_file.write(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", required=True)
    parser.add_argument("-g", "--gpu_type", required=True)
    args = parser.parse_args()

    cross_validation_eval(args.exp_dir, args.gpu_type)