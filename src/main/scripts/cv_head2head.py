import sys
import json
import os

import argparse


def get_scores(exp_dir):
    '''
    Get the scores for each fold

    :param exp_dir:
    :return:
    '''

    num_folds = 5
    num_config = 8

    map_folds = []
    hits_at_1_folds = []
    hits_at_10_folds = []
    hits_at_50_folds = []
    hits_at_100_folds = []

    for i in range(num_folds):
        for j in range(num_config):
            config_idx = i * num_config + j

            config_dir = os.path.join(exp_dir, "config_" + str(config_idx))
            if os.path.exists(config_dir):
                test_scores_json = os.path.join(config_dir, "test_scores.json")
                if (os.path.exists(test_scores_json)):
                    with open(test_scores_json) as f:
                        all_lines = f.readlines()
                        score_json = json.loads(all_lines[-1])
                        map_folds.append(score_json["map"])
                        hits_at_1_folds.append(score_json["hits_at_1"])
                        hits_at_10_folds.append(score_json["hits_at_10"])
                        hits_at_50_folds.append(score_json["hits_at_50"])
                        hits_at_100_folds.append(score_json["hits_at_100"])



    assert(len(map_folds) == num_folds)

    return map_folds, hits_at_1_folds, hits_at_10_folds, hits_at_50_folds, hits_at_100_folds


def h2h(exp_one_dir, exp_two_dir):
    '''
    Get head to head comparisons

    :param exp_one_dir:
    :param exp_two_dir:
    :return:
    '''
    exp_one_results = get_scores(exp_one_dir)
    exp_two_results = get_scores(exp_two_dir)


    list_results_exp_one_wins = []
    list_results_exp_two_wins = []


    for (exp_one_score, exp_two_score, score) in zip(exp_one_results, exp_two_results, ["map", "hits at 1", "hits at 10", "hits at 50", "hits at 100"]):
        exp_one_wins = 0
        exp_two_wins = 0

        for (fold_one_score, fold_two_score) in zip(exp_one_score, exp_two_score):
            if fold_one_score > fold_two_score:
                exp_one_wins +=1
            if fold_two_score > fold_one_score:
                exp_two_wins +=1

        print(score, exp_one_wins, exp_two_wins)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e1", "--exp_one_dir", required=True)
    parser.add_argument("-e2", "--exp_two_dir", required=True)
    args = parser.parse_args()


    h2h(args.exp_one_dir, args.exp_two_dir)