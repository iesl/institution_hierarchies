from main.utils.graph import get_connected_components
from main.utils.read_csv import read_relationship_csv, read_grid_csv
import main.utils.data_pathnames as data_pathnames

import random
import os
import pickle


def find_non_ancestor_same_comp_negatives(dict_grid2comp, dict_comp2grid, dict_comp2ancestors):
    '''
    Finds negatives which are non ancestors in the same component for each grid id

    :param dict_grid2comp: dictionary of grid_id to comp_idx
    :param dict_comp2grid: dictionary of comp_idx to grid_id
    :param dict_comp2ancestors: dictionary of comp idx to (num_grid, dictionary grid id to ancestors )
    :return: dictionary of grid_id to negatives
    '''

    dict_grid2negatives = {}

    for (grid_id, comp_idx) in dict_grid2comp.items():
        # Get the ancestors of the current grid_id
        ancestors = set(dict_comp2ancestors[comp_idx][1][grid_id])
        # Get all possible negatives, which are every grid_id in the same component
        possible_negatives = set(dict_comp2grid[comp_idx])

        # Remove the grid_id itself as a negative
        possible_negatives = possible_negatives.difference(ancestors)
        possible_negatives = possible_negatives.difference(set([grid_id]))

        negatives = list(possible_negatives)

        if len(negatives) > 1000:
            negatives = random.sample(negatives, 1000)
        dict_grid2negatives[grid_id] = negatives

    return dict_grid2negatives

def find_ancestor(parent_of, grid_id):
    '''
    Finds all the ancestors of a node

    :param parent_of: dictionary of parent of
    :param grid_id: grid_id to get ancestors of
    :param list_ancestors: list of ancestors to return
    :return:
    '''
    list_ancestors = []

    if grid_id in parent_of:
        list_parent_grid_id = parent_of[grid_id]

        for parent_grid_id in list_parent_grid_id:
            list_ancestors.append(parent_grid_id)
            old_parent_grid_id = str(parent_grid_id)
            ancestors = find_ancestor(parent_of, parent_grid_id)
            list_ancestors.extend(ancestors)

    return list_ancestors

def get_all_ancestors(parent_of, dict_comp2grid):
    '''
    Gets all the ancestors for every node

    :param parent_of: dictionary of parents to children
    :param dict_comp2grid: dictionary of component to grid_id
    :return: dictionary mapping comp_idx to dictionary of grid_id to its ancestors
    :return: comp_idx of the largest component
    '''

    dict_comp2ancestors = {}
    list_large_comp_idx = []
    for (comp_idx, list_grid_id) in dict_comp2grid.items():

        if len(list_grid_id) > 1:
            # Keep track of number of ancestors in each component
            num_anc = 0
            # dictionary of grid_id to its ancestors
            dict_id2ancestors = {}

            for grid_id in list_grid_id:
                ancestors = find_ancestor(parent_of, grid_id)
                num_anc += len(ancestors)
                dict_id2ancestors[grid_id] = ancestors
            dict_comp2ancestors[comp_idx] = (num_anc, dict_id2ancestors)

            # Save the large comp idx
            if num_anc > 1000:
                list_large_comp_idx.append(comp_idx)

        else:
            dict_comp2ancestors[comp_idx] = (0, {list_grid_id[0]: []})

    return dict_comp2ancestors, list_large_comp_idx

def init_data_dict():
    '''
    Init data dict that will be reused for all folds
    '''

    dict_id2name = read_grid_csv()
    parent_of, child_of = read_relationship_csv()

    comp_dict = os.path.join("data", "dict", "comp_dict")
    if not os.path.exists(comp_dict):
        os.makedirs(comp_dict)

    dict_grid2comp, dict_comp2grid = get_connected_components(dict_id2name, parent_of, child_of)
    dict_comp2ancestors, large_comp_idx = get_all_ancestors(parent_of, dict_comp2grid)
    dict_grid2non_ancestor_negative_same_comp = find_non_ancestor_same_comp_negatives(dict_grid2comp, dict_comp2grid, dict_comp2ancestors)

    pickle.dump(dict_grid2comp, open(data_pathnames.DICT_GRID2COMP, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_comp2grid, open(data_pathnames.DICT_COMP2GRID, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_comp2ancestors, open(data_pathnames.DICT_COMP2ANCESTORS, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(large_comp_idx, open(data_pathnames.LARGE_COMP_IDX, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict_grid2non_ancestor_negative_same_comp, open(data_pathnames.DICT_GRID2NONANCESTOR_SAME_COMP_NEGATIVE, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    data_pathnames.init()
    init_data_dict()