import datetime
import torch
import codecs
import subprocess
import json
import os

global device; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def __filter_json(the_dict):
    '''
    Filters the dictionary so that only the string, floats, ints, and lists are stored 

    param the_dict: dictionary to filter 
    return: filtered dictionary 
    '''
    res = {}
    for k in the_dict.keys():
        if type(the_dict[k]) is str or type(the_dict[k]) is float or type(the_dict[k]) is int or type(the_dict[k]) is list:
            res[k] = the_dict[k]
        elif type(the_dict[k]) is dict:
            res[k] = __filter_json(the_dict[k])
    return res

def save_dict_to_json(the_dict,the_file):
    '''
    Saves the dictionary to file 

    param the_dict: dictionary to save 
    param the_file: file to save dictionary 
    '''
    with open(the_file, 'a+') as fout:
        fout.write(json.dumps(__filter_json(the_dict)))
        fout.write("\n")

def make_directory(dir_name):
    '''
    Makes directory if it doesn't exist 
    
    param dir_name: directory name 
    '''
    if (not os.path.exists(dir_name)):
        os.makedirs(dir_name)


def make_exp_dir(config):
    '''
    Makes experiment directory which includes timestamp to ensure distinct 
    
    param config: config
    return: experiment directory name
    '''

    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second)

    field_name = "inst"
    if config.include_city:
        field_name += "_city"
    if config.include_state:
        field_name += "_state"
    if config.include_country:
        field_name += "_country"
    if config.include_type:
        field_name += "_type"

    exp_dir = os.path.join("exp_out", config.model_name, field_name, ts)
    make_directory(exp_dir)

    return exp_dir

def setup_map_counter2gpu():
    '''
    Returns mapping of counter to GPU given how many available GPUS there are

    :return:
    '''
    NUM_1080_SHORT = 44
    NUM_1080_SHORT = 0
    NUM_1080_LONG = 20
    NUM_TITANX_SHORT = 40
    NUM_TITANX_LONG = 20
    NUM_M40_SHORT = 10
    NUM_M40_LONG = 6
    NUM_2080_SHORT = 10
    NUM_2080_LONG = 6

    list_num_gpus = [NUM_1080_SHORT, NUM_1080_LONG, NUM_TITANX_SHORT, NUM_TITANX_LONG, NUM_M40_SHORT, NUM_M40_LONG, NUM_2080_SHORT, NUM_2080_LONG]
    list_gpus_names = ["1080ti-short", "1080ti-long", "titanx-short", "titanx-long", "m40-short", "m40-long", "2080ti-short", "2080ti-long"]
    map_counter2gpu = {}

    counter = 0
    for idx, num_gpus in enumerate(list_num_gpus):
        for i in range(num_gpus):
            map_counter2gpu[counter] = list_gpus_names[idx]
            counter += 1

    return map_counter2gpu


def read_pred_file(filename):
    '''
    Reads predictions from filename

    :param filename:
    :return: dictionary of qry to list of tuples (cnd, lbl, pred_score)
    '''

    dict_qry2pred = {}

    with open(filename, 'r') as f:
        for line in f.readlines():
            split_tab = line.strip('\n').split('\t')
            qry = split_tab[0]
            tuple = (split_tab[1], split_tab[2], split_tab[3])

            if split_tab[0] in dict_qry2pred:
                dict_qry2pred[qry].append(tuple)
            else:
                dict_qry2pred[qry] = [tuple]

    return dict_qry2pred
