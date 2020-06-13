import sys
import os
import torch
import argparse


from main.objects.Vocab import Vocab
from main.objects.Config import Config
from main.utils.util import setup_map_counter2gpu





'''
Setsup a bash script to evaluate the best model in parallel with multiple GPU's on a slurm manager
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", required=True)
    parser.add_argument("-n", "--num_shards", required=True)
    args = parser.parse_args()

    # Get mapping of counter to available gpu
    map_counter2gpu = setup_map_counter2gpu()
    num_gpus = len(map_counter2gpu)

    # Set up error directory for all scripts
    error_dir = os.path.join(args.exp_dir, "error_dir")
    if (not os.path.exists(error_dir)):
        os.makedirs(error_dir)

    # Set up output directory for all scripts
    output_dir = os.path.join(args.exp_dir, "output_dir")
    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    # Set up bash script to run dev in parallel

    bash_script_idx = 0
    bash_script = os.path.join(args.exp_dir, "parallel_dev_%d.sh" % bash_script_idx)
    f = open(bash_script, 'w+')

    dev_pred_dir = os.path.join(args.exp_dir, "dev_pred")
    if not os.path.exists(dev_pred_dir):
        os.makedirs(dev_pred_dir)


    # Loop through the number of shards
    for shard_idx in range(int(args.num_shards)):
        error_file = os.path.join(error_dir, "error_shard_{}".format(shard_idx))
        output_file = os.path.join(output_dir, "output_shard_{}".format(shard_idx))


        command = "sbatch --partition=gpu --exclude=gpu-0-0 --gres=gpu:1 --error {} --output {} --mem=15GB bin/run/dev_model.sh \
                    {} True {} \n".format(error_file, output_file, args.exp_dir, shard_idx)
        f.write(command + '\n')

    # counter = 0
    #
    # # Loop through all configs for grid search
    # for config_dirname in os.listdir(args.exp_dir):
    #     if "config_" in config_dirname:
    #         underscore_idx = config_dirname.rfind('_')
    #         config_idx = config_dirname[underscore_idx+1:]
    #         model_dir = os.path.join(args.exp_dir, config_dirname, "models")
    #
    #         dev_pred_dir = os.path.join(args.exp_dir, config_dirname, "dev_pred")
    #         if not os.path.exists(dev_pred_dir):
    #             os.makedirs(dev_pred_dir)
    #
    #         # Loop through every checkpoint model saved
    #         for model_filename in os.listdir(model_dir):
    #             # Get directory for current model
    #             underscore_idx = model_filename.rfind('_')
    #             num_batches = model_filename[underscore_idx+1:]
    #             dev_pred_batch_dir = os.path.join(dev_pred_dir, ("batch_%s" % num_batches))
    #             if not os.path.exists(dev_pred_batch_dir):
    #                 os.makedirs(dev_pred_batch_dir)
    #
    #             # Loop through the number of shards
    #             for shard_idx in range(int(args.num_shards)):
    #                 error_file = os.path.join(error_dir, "error_config_{}_batch_{}_shard_{}".format(config_idx, num_batches, shard_idx))
    #                 output_file = os.path.join(output_dir, "output_config_{}_batch_{}_shard_{}".format(config_idx, num_batches, shard_idx))
    #
    #                 gpu = map_counter2gpu[counter]
    #
    #                 command = "sbatch --partition={} --gres=gpu:1 --error {} --output {} --mem=15GB bin/run/dev_model.sh \
    #                             {} {} {} {} \n".format(gpu, error_file, output_file, args.exp_dir, config_idx, num_batches, shard_idx)
    #                 f.write(command + '\n')
    #                 counter += 1
    #
    #                 if counter >= num_gpus:
    #                     counter = 0
    #                     bash_script_idx += 1
    #                     bash_script = os.path.join(args.exp_dir, "parallel_dev_%d.sh" % bash_script_idx)
    #                     f = open(bash_script, 'w+')