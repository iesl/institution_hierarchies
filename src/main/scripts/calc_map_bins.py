
import pickle
import os
import argparse

from main.utils.util import read_pred_file
from main.objects.Scorer import Scorer
from main.utils.util import save_dict_to_json


inst_name_dict = pickle.load(open(os.path.join("data", "ancestors", "grid2inst_name.dict"), 'rb'))
city_dict = pickle.load(open(os.path.join("data", "ancestors", "grid2city.dict"), 'rb'))
state_dict = pickle.load(open(os.path.join("data", "ancestors", "grid2state.dict"), 'rb'))
country_dict = pickle.load(open(os.path.join("data", "ancestors", "grid2country.dict"), 'rb'))
type_dict = pickle.load(open(os.path.join("data", "ancestors", "grid2type.dict"), 'rb'))


LIST_BINS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

def get_name(grid_id):
    '''
    Returns the name of a grid_id

    :param grid_id:
    :return:
    '''
    list_fields = [inst_name_dict[grid_id], city_dict[grid_id], state_dict[grid_id], country_dict[grid_id]]

    return ','.join(list_fields)


def calc_percent_match(qry_name, cnd_name):
    '''
    Calculate % of tokens in candidate also in query

    :param qry_name:
    :param cnd_name:
    :return:
    '''

    qry_tokens = set(qry_name.split(' '))
    cnd_tokens = set(cnd_name.split(' '))

    intsct_tokens = qry_tokens.intersection(cnd_tokens)

    return float(len(intsct_tokens)) / len(cnd_tokens)

def calc_map_bins(pred_file, output_file):
    '''
    Calculate map for each bin

    :param pred_file:
    :param output_file:
    :return:
    '''
    dict_qry2tuple = read_pred_file(pred_file)

    dict_percent2scorer = {}
    for i in LIST_BINS:
        dict_percent2scorer[i] = Scorer([], "", 0, 0)


    for qry, tuple in dict_qry2tuple.items():
        qry_name = get_name(qry)

        dict_percent2tuples = {}
        for i in LIST_BINS:
            dict_percent2tuples[i] = []

        for (cnd, lbl, score) in tuple:
            cnd_name = get_name(cnd)
            percent_sim = round(calc_percent_match(qry_name, cnd_name), 1)
            dict_percent2tuples[percent_sim].append((int(lbl), float(score)))

        for i in LIST_BINS:

            list_tuples = dict_percent2tuples[i]

            if len(list_tuples) >  0:
                (lbl, score) = zip(*list_tuples)
                dict_percent2scorer[i].add_batch_pred_scores(score, lbl, True)

    scores_obj = {}

    for (k, s) in dict_percent2scorer.items():
        score = s.calc_scores(is_write=False)
        scores_obj[k] = float(str("%.3f") % score)

    save_dict_to_json(scores_obj, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred_file", required=True)
    parser.add_argument("-o", "--output_file")

    args = parser.parse_args()
    calc_map_bins(args.pred_file, args.output_file)