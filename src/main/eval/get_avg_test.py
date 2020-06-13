import sys
import json
import os
import numpy as np

from main.utils.util import save_dict_to_json

import argparse


def eval_random_seeds_baseline(exp_dir):
    '''
    Loops through configuration in experiment directory and finds the configuration with the highest dev score
    '''

    list_map = []
    list_hits_at_1 = []
    list_hits_at_10 = []
    list_hits_at_50 = []
    list_hits_at_100 = []

    config_counter = 0

    while True:
        test_file = os.path.join(exp_dir, "config_" + str(config_counter), "test_scores.json")
        if (os.path.exists(test_file)):
            with open(test_file) as f:

                all_lines = f.readlines()
                for line in all_lines:
                    score_json = json.loads(line)
                    map = score_json["map"]
                    hits_at_1 = score_json["hits_at_1"]
                    hits_at_10 = score_json["hits_at_10"]
                    hits_at_50 = score_json["hits_at_50"]
                    hits_at_100 = score_json["hits_at_100"]
                    list_map.append(map)
                    list_hits_at_1.append(hits_at_1)
                    list_hits_at_10.append(hits_at_10)
                    list_hits_at_50.append(hits_at_50)
                    list_hits_at_100.append(hits_at_100)
        else:
            break

        config_counter += 1

    dict_scores = {}

    map = np.asarray(list_map)
    dict_scores["avg_map"] = "%.3f" % np.mean(map)
    dict_scores["std_map"] = "%.3f" % np.std(map)

    hits_at_1 = np.asarray(list_hits_at_1)
    dict_scores["avg_hits_at_1"] = "%.3f" % np.mean(hits_at_1)
    dict_scores["std_hits_at_1"] = "%.3f" % np.std(hits_at_1)

    hits_at_10 = np.asarray(list_hits_at_10)
    dict_scores["avg_hits_at_10"] = "%.3f" % np.mean(hits_at_10)
    dict_scores["std_hits_at_10"] = "%.3f" % np.std(hits_at_10)

    hits_at_50 = np.asarray(list_hits_at_50)
    dict_scores["avg_hits_at_50"] = "%.3f" % np.mean(hits_at_50)
    dict_scores["std_hits_at_50"] = "%.3f" % np.std(hits_at_50)

    hits_at_100 = np.asarray(list_hits_at_100)
    dict_scores["avg_hits_at_100"] = "%.3f" % np.mean(hits_at_100)
    dict_scores["std_hits_at_100"] = "%.3f" % np.std(hits_at_100)

    test_file = os.path.join(exp_dir, "test_scores.json")
    save_dict_to_json(dict_scores, test_file)


def eval_random_seeds(exp_dir):
    '''
    Loops through configuration in experiment directory and finds the configuration with the highest dev score
    '''

    list_map = []
    list_hits_at_1 = []
    list_hits_at_10 = []
    list_hits_at_50 = []
    list_hits_at_100 = []

    for i in range(40):

        test_file = os.path.join(exp_dir, "config_" + str(i), "test_scores.json")
        if (os.path.exists(test_file)):
            with open(test_file) as f:

                all_lines = f.readlines()
                for line in all_lines:
                    score_json = json.loads(line)
                    map = score_json["map"]
                    hits_at_1 = score_json["hits_at_1"]
                    hits_at_10 = score_json["hits_at_10"]
                    hits_at_50 = score_json["hits_at_50"]
                    hits_at_100 = score_json["hits_at_100"]
                    list_map.append(map)
                    list_hits_at_1.append(hits_at_1)
                    list_hits_at_10.append(hits_at_10)
                    list_hits_at_50.append(hits_at_50)
                    list_hits_at_100.append(hits_at_100)
                    print(score_json)

    dict_scores = {}

    map = np.asarray(list_map)
    dict_scores["avg_map"] = "%.3f" % np.mean(map)
    dict_scores["std_map"] = "%.3f" % np.std(map)

    hits_at_1 = np.asarray(list_hits_at_1)
    dict_scores["avg_hits_at_1"] = "%.3f" % np.mean(hits_at_1)
    dict_scores["std_hits_at_1"] = "%.3f" % np.std(hits_at_1)

    hits_at_10 = np.asarray(list_hits_at_10)
    dict_scores["avg_hits_at_10"] = "%.3f" % np.mean(hits_at_10)
    dict_scores["std_hits_at_10"] = "%.3f" % np.std(hits_at_10)

    hits_at_50 = np.asarray(list_hits_at_50)
    dict_scores["avg_hits_at_50"] = "%.3f" % np.mean(hits_at_50)
    dict_scores["std_hits_at_50"] = "%.3f" % np.std(hits_at_50)

    hits_at_100 = np.asarray(list_hits_at_100)
    dict_scores["avg_hits_at_100"] = "%.3f" % np.mean(hits_at_100)
    dict_scores["std_hits_at_100"] = "%.3f" % np.std(hits_at_100)

    test_file = os.path.join(exp_dir, "test_scores.json")
    save_dict_to_json(dict_scores, test_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--baseline", default="False")
    parser.add_argument("-e", "--exp_dir", required=True)
    args = parser.parse_args()

    if args.baseline == "False":
        eval_random_seeds(args.exp_dir)
    else:
        eval_random_seeds_baseline(args.exp_dir)