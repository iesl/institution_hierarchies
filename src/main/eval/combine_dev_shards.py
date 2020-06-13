import sys
import os
import torch
import argparse


'''
Combines all the test shards predictions into one file to score later 
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_dir", required=True)
    parser.add_argument("-n", "--num_partitions", required=True)
    args = parser.parse_args()

    partition_folder = os.path.join(args.config_dir, "dev_shards")
    test_prediction_filename = os.path.join(args.config_dir, "dev.predictions")


    with open(test_prediction_filename, 'w+') as f_out:
        total_lines = 0
        for i in range(int(args.num_partitions)):
            parititon_prediction_filename = os.path.join(partition_folder, "shard_{}.pred".format(str(i)))
            if(os.path.exists(parititon_prediction_filename)):
                with open(parititon_prediction_filename, 'r') as f_in:
                    all_lines =  f_in.readlines()
                total_lines += len(all_lines)
                for line in all_lines:
                    f_out.write(line)
        print(total_lines)