import os
import sys
from shutil import copyfile, copytree
import argparse

from main.objects.GridSearchConfig import GridSearchConfig
from main.utils.util import make_exp_dir


def setup_grid_search(exp_dir, grid_search_config, gpu_type_one, num_gpu_type_one, gpu_type_two):
    '''
    Sets up grid search to train different configurations of the model in parallel 
    
    param exp_dir: experiment directory for all the configurations 
    param grid_search_config: grid search configuration with all the parameters used in grid search 
    param gpu_type_one: gpu one to train on
    param num_gpu_type_one: num of gpu one to train on
    param gpu_type_two: gpu two to train on
    param mem: amount of mem to train with
    '''
    bash_script = os.path.join(exp_dir, "grid_search_config.sh")

    with open(bash_script, 'w') as f:
        f.write("#!/usr/bin/env bash \n")

        ctr = 0
        gpu_type = gpu_type_one

        for config_counter, config in enumerate(grid_search_config.configs_iter()):
            config_dir = os.path.join(exp_dir, "config_" + str(config_counter))
            os.makedirs(config_dir)

            config_param = os.path.join(config_dir, "config.json")
            config.save_config(config_dir)

            error_file = os.path.join(config_dir, "error")
            output_file = os.path.join(config_dir, "output") 

            command = "sbatch --partition={} --gres=gpu:1 --mem=30G --error {} --output {} bin/run/train_model.sh {} {} \n".format(\
                gpu_type, error_file, output_file, config_param, "True")

            if num_gpu_type_one is not None:
                ctr += 1
                if ctr >= int(num_gpu_type_one):
                    gpu_type = gpu_type_two

            f.write(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", required=True)
    parser.add_argument("-g1", "--gpu_type_one", required=True)
    parser.add_argument("-n1", "--num_gpu_type_one", default=None)
    parser.add_argument("-g2", "--gpu_type_two", default=None)
    args = parser.parse_args()

    grid_search_config = GridSearchConfig(args.config_file)

    # Sets up experiment directory for all the configuration possibilties
    exp_dir = make_exp_dir(grid_search_config)
    copytree(os.path.join(os.environ['SED_ROOT'], 'src'), os.path.join(exp_dir, 'src'))  
    grid_search_config.save_config(exp_dir)

    setup_grid_search(exp_dir, grid_search_config, args.gpu_type_one, args.num_gpu_type_one, args.gpu_type_two)