import sys
import json
import os

import argparse


def eval_random_seeds(exp_dir, gpu_type):
    '''
    Loops through configuration in experiment directory and finds the configuration with the highest dev score
    '''
    config_counter = 0

    bash_script = os.path.join(exp_dir, "eval_random_seeds.sh")

    with open(bash_script, 'w') as f:
        f.write("#!/usr/bin/env bash \n")

        while True:
            config_dir = os.path.join(exp_dir, "config_" + str(config_counter))
            if (os.path.exists(config_dir)):

                error_file = os.path.join(config_dir, "test_error")
                output_file = os.path.join(config_dir, "test_output")

                command = "sbatch --partition={} --gres=gpu:1 --mem=30G --error {} --output {} bin/run/test_model.sh {} \n".format(\
                    gpu_type, error_file, output_file, config_dir)

                f.write(command)
            else:
                break

            config_counter += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", required=True)
    parser.add_argument("-g", "--gpu_type", required=True)
    args = parser.parse_args()

    eval_random_seeds(args.exp_dir, args.gpu_type)