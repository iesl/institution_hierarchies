
import pickle
import os
import argparse

from main.utils.util import read_pred_file



TOP_K = 1

inst_name_dict = pickle.load(open(os.path.join("data", "ancestors", "grid2inst_name.dict"), 'rb'))
city_dict = pickle.load(open(os.path.join("data", "ancestors", "grid2city.dict"), 'rb'))
state_dict = pickle.load(open(os.path.join("data", "ancestors", "grid2state.dict"), 'rb'))
country_dict = pickle.load(open(os.path.join("data", "ancestors", "grid2country.dict"), 'rb'))
type_dict = pickle.load(open(os.path.join("data", "ancestors", "grid2type.dict"), 'rb'))


def get_name(grid_id):
    '''
    Returns the name of a grid_id

    :param grid_id:
    :return:
    '''
    list_fields = [inst_name_dict[grid_id], city_dict[grid_id], state_dict[grid_id], country_dict[grid_id]]

    return ','.join(list_fields)

def check(pred_file):
    '''
    Finds prediction that file one got correct and file two got wrong

    :param pred_one:
    :param pred_two:
    :param output_file:
    :return:
    '''

    dict_pred_qry2tuple = read_pred_file(pred_file)
    all_qry = dict_pred_qry2tuple.keys()

    dict_pair2lbl = {}

    count = 0

    dict_wrong_pair = {}

    for grid_id in all_qry:
        for idx, (grid_id_one, lbl, _) in enumerate(dict_pred_qry2tuple[grid_id]):
            pair = (grid_id, grid_id_one)

            if pair in dict_pair2lbl:
                if dict_pair2lbl[pair] != lbl:
                    count += 1
                    dict_wrong_pair[pair] = True
            else:
                dict_pair2lbl[pair] = lbl

    print("Count: ", count)

    return dict_wrong_pair



def compare_pred_old(pred_one_file, pred_two_file, output_file):
    '''
    Finds prediction that file one got correct and file two got wrong

    :param pred_one:
    :param pred_two:
    :param output_file:
    :return:
    '''

    dict_pred_one_qry2tuple = read_pred_file(pred_one_file)
    dict_pred_two_qry2tuple = read_pred_file(pred_two_file)

    output = open(output_file, 'w+')

    dict_wrong_pair =  check(pred_one_file)

    all_qry = dict_pred_one_qry2tuple.keys()


    for grid_id in all_qry:
        pred_one_tuple = dict_pred_one_qry2tuple[grid_id]
        pred_two_tuple = dict_pred_two_qry2tuple[grid_id]

        pred_one_tuple.sort(key=lambda x: x[2], reverse=True)
        pred_two_tuple.sort(key=lambda x: x[2])

        # Stores true positives that are scored the highest
        pred_one_top_k_correct = []
        pred_one_top_k_grid_id = {}
        for idx, (grid_id_one, lbl_one, score_one) in enumerate(pred_one_tuple):
            if int(lbl_one) == 1:
                pred_one_top_k_correct.append((grid_id_one, lbl_one, score_one))
                pred_one_top_k_grid_id[grid_id_one] = 1
            else:
                break

        seen_lbl_1 = False
        # Stores false negatives that are predicted higher than a true positive
        pred_two_top_k_incorrect = {}
        # Stores false postive
        pred_two_false_positive = {}
        for (grid_id_two, lbl_two, score_two) in pred_two_tuple:
            if int(lbl_two) == 1:
                seen_lbl_1 = True
                pred_two_false_positive[grid_id_two] = (grid_id_two, lbl_two, score_two)
                if grid_id_two in pred_one_top_k_grid_id:
                    pred_one_top_k_grid_id[grid_id] = 0
            if seen_lbl_1 and int(lbl_two) == 0:
                pred_two_top_k_incorrect[grid_id_two] = (grid_id_two, lbl_two, score_two)


        for (grid_id_one, lbl_one, score_one) in pred_one_top_k_correct:
            skip_write = False

            if grid_id_one in pred_one_top_k_grid_id and pred_one_top_k_grid_id[grid_id_one] == 1:
                list_lines = ['\t'.join(["TP", get_name(grid_id), get_name(grid_id_one), lbl_one, score_one]) + '\n']

                for (grid_id_two, lbl_two, score_two) in list(pred_two_false_positive.values()):
                    if (grid_id, grid_id_two) not in dict_wrong_pair:
                        if grid_id_two not in pred_two_top_k_incorrect:
                            list_lines.append('\t'.join(["FN: ", get_name(grid_id), get_name(grid_id_two), lbl_two, score_two]) + '\n')

                for (grid_id_two, lbl_two, score_two) in list(pred_two_top_k_incorrect.values()):
                    if grid_id_two not in pred_one_top_k_grid_id and (grid_id, grid_id_two) not in dict_wrong_pair:
                        list_lines.append('\t'.join(["FP:", get_name(grid_id), get_name(grid_id_two), lbl_two, score_two]) + '\n')
                    else:
                        skip_write = True

            if not skip_write:
                for line in list_lines:
                    output.write(line)


def compare_pred(pred_one_file, pred_two_file, output_file):
    '''
    Finds prediction that file one got correct and file two got wrong

    :param pred_one:
    :param pred_two:
    :param output_file:
    :return:
    '''

    dict_pred_one_qry2tuple = read_pred_file(pred_one_file)
    dict_pred_two_qry2tuple = read_pred_file(pred_two_file)
    all_qry = dict_pred_one_qry2tuple.keys()


    output = open(output_file, 'w+')
    dict_wrong_pair = check(pred_one_file)

    for grid_id in all_qry:
        pred_one_tuple = dict_pred_one_qry2tuple[grid_id]
        pred_two_tuple = dict_pred_two_qry2tuple[grid_id]

        pred_one_dict = {}
        for (grid_id_one, lbl_one, score_one) in pred_one_tuple:
            pred_one_dict[grid_id_one] = (score_one, lbl_one)

        pred_two_dict = {}
        for (grid_id_two, lbl_two, score_two) in pred_two_tuple:
            pred_two_dict[grid_id_two] = (lbl_two, score_two)

        pred_one_tuple.sort(key=lambda x: x[2], reverse=True)
        pred_two_tuple.sort(key=lambda x: x[2], reverse=True)

        pred_one_top_k = pred_one_tuple[:TOP_K]
        pred_two_top_k = pred_two_tuple[:TOP_K]

        pred_one_top_k_correct = list(filter(lambda x: int(x[1]) == 1, pred_one_top_k))

        pred_one_top_k_grid_id = list(map(lambda x: x[0], pred_one_top_k))
        pred_two_top_k_grid_id = list(map(lambda x: x[0], pred_two_top_k))


        list_lines = []
        for (grid_id_one, lbl_one, score_one) in pred_one_top_k_correct:
            if grid_id_one not in pred_two_top_k_grid_id:
                (score_two, lbl_two) = pred_two_dict[grid_id_one]
                print(get_name(grid_id), get_name(grid_id_one))
                if lbl_two == lbl_one:
                    list_lines.append('\t'.join(["P1TP: ", get_name(grid_id), get_name(grid_id_one), lbl_one, score_one]) + '\n')
                    list_lines.append('\t'.join(["P2FP: ", get_name(grid_id), get_name(grid_id_one), lbl_two, score_two]) + '\n')

        for (grid_id_two, lbl_two, score_two) in pred_two_top_k:
            if grid_id_two not in pred_one_top_k_grid_id:
                (score_one, lbl_one) = pred_one_dict[grid_id_two]
                if lbl_two == lbl_one:
                    list_lines.append('\t'.join(["P2FN: ", get_name(grid_id), get_name(grid_id_two), lbl_two, score_two]) + '\n')
                    list_lines.append('\t'.join(["P1TN: ", get_name(grid_id), get_name(grid_id_two), lbl_one, score_one]) + '\n')


        for line in list_lines:
            output.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--pred_one_file", required=True)
    parser.add_argument("-f2", "--pred_two_file")
    parser.add_argument("-o", "--output_file")

    args = parser.parse_args()
    # compare_pred(args.pred_one_file, args.pred_two_file, args.output_file)
    check(args.pred_one_file)