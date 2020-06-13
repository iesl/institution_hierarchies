import torch
import os
import argparse
import pickle

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

def get_weights(config_dir):
    '''
    Get linear weights of model

    :param config_dir: config dir for model
    :return:
    '''

    test_pred = os.path.join(config_dir, "test.pred")
    test_pred_inst_name = os.path.join(config_dir, "test_inst_name.pred")


    f_in = open(test_pred, 'r')

    with open(test_pred_inst_name, 'w+') as f_out:
        for line in f_in.readlines():
            tab_split = line.strip('\n').split('\t')

            if len(tab_split) != 4:
                print("Error: %s has less than 4 tabs" % line)

            qry_grid_id = tab_split[0]
            cnd_grid_id = tab_split[1]

            qry_inst_name = get_name(qry_grid_id)
            cnd_inst_name = get_name(cnd_grid_id)

            new_line = '\t'.join([qry_inst_name, cnd_inst_name, tab_split[2], tab_split[3]]) + '\n'
            f_out.write(new_line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_dir", required=True)
    args = parser.parse_args()

    get_weights(args.config_dir)