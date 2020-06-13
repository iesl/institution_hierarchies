from main.scripts.graph import get_connected_components, get_root_nodes
from main.scripts.read_csv import read_relationship_csv, read_grid_csv
import main.scripts.data_path as data_path

import itertools
import argparse
import random
import os
import pickle
import numpy as np

class DatasetCreator(object):

    def __init__(self, dataset_seed):
        '''
        :param seed:
        '''
        random.seed(0)
        self.seed = dataset_seed
        self.dataset_dir = os.path.join("data", "cross_validation", "fold_%d" % self.seed)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        self.train_data = None
        self.dev_data = None
        self.test_data = None

    def create_dict(self):
        '''
        Create dictionaries to create datasets
        '''
        data_path.init()

        self.dict_id2name = read_grid_csv()
        self.parent_of, self.child_of = read_relationship_csv()

        dict_grid2comp_file = os.path.join(data_path.DICT_DIR, "grid2comp.dict")
        dict_comp2grid_file = os.path.join(data_path.DICT_DIR, "comp2grid.dict")
        dict_comp2ancestors_file = os.path.join(data_path.DICT_DIR, "comp2ancestors.dict")
        dict_grid2non_ancestor_negative_file = os.path.join(data_path.DICT_DIR, "grid2nonancestor_negative.dict")
        large_comp_idx_file = os.path.join(data_path.DICT_DIR, "large_comp_idx.dict")

        self.dict_grid2comp, self.dict_comp2grid = get_connected_components(self.dict_id2name, self.parent_of, self.child_of)
        self.dict_comp2ancestors, self.large_comp_idx = self.get_all_ancestors(self.parent_of, self.dict_comp2grid)
        self.dict_comp2split, self.dict_split2comp = self.map_comp_2_split(self.dict_comp2ancestors, self.large_comp_idx)
        self.dict_split2grid, self.dict_split2non_grid = self.get_split2grid(self.dict_split2comp, self.dict_comp2grid)

        self.dict_grid2non_ancestor_negative = self.find_non_ancestor_negatives(self.dict_grid2comp, self.dict_comp2grid, self.dict_comp2ancestors)

        pickle.dump(self.dict_grid2comp, open(dict_grid2comp_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.dict_comp2grid, open(dict_comp2grid_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.dict_comp2ancestors, open(dict_comp2ancestors_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.large_comp_idx, open(large_comp_idx_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.dict_grid2non_ancestor_negative, open(dict_grid2non_ancestor_negative_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        dict_comp2split_file = os.path.join(self.dataset_dir, "comp2split.dict")
        dict_split2comp_file = os.path.join(self.dataset_dir, "split2comp.dict")
        dict_grid2negatives_outside_component_file = os.path.join(self.dataset_dir, "grid2negatives_outside_component.dict")
        dict_grid2random_negatives_file = os.path.join(self.dataset_dir, "grid2random_negatives.dict")
        dict_grid2negatives_file = os.path.join(self.dataset_dir, "grid2negatives.dict")

        self.dict_grid2negatives_outside_component = self.find_negatives_outside_component(self.dict_grid2comp, self.dict_comp2grid, self.dict_comp2split, self.dict_split2grid, self.dict_split2non_grid)
        self.dict_grid2random_negatives = self.randomly_sample_negatives(self.dict_grid2comp, self.dict_comp2grid, self.dict_split2comp, self.dict_comp2split)
        self.dict_grid2negatives = self.merge_negatives(self.dict_grid2non_ancestor_negative, self.dict_grid2negatives_outside_component, self.dict_grid2random_negatives)

        pickle.dump(self.dict_comp2split, open(dict_comp2split_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.dict_split2comp, open(dict_split2comp_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.dict_grid2negatives_outside_component, open(dict_grid2negatives_outside_component_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.dict_grid2random_negatives, open(dict_grid2random_negatives_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.dict_grid2negatives, open(dict_grid2negatives_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


    def load_dict(self):
        '''
        Load dictionary
        '''
        self.dict_id2name = read_grid_csv()
        self.parent_of, self.child_of = read_relationship_csv()

        dict_grid2comp_file = os.path.join(data_path.DICT_DIR, "grid2comp.dict")
        dict_comp2grid_file = os.path.join(data_path.DICT_DIR, "comp2grid.dict")
        dict_comp2ancestors_file = os.path.join(data_path.DICT_DIR, "comp2ancestors.dict")
        dict_grid2non_ancestor_negative_file = os.path.join(data_path.DICT_DIR, "grid2nonancestor_negative.dict")
        large_comp_idx_file = os.path.join(data_path.DICT_DIR, "large_comp_idx.dict")

        self.dict_grid2comp = pickle.load(open(dict_grid2comp_file, 'rb'))
        self.dict_comp2grid = pickle.load(open(dict_comp2grid_file, 'rb'))
        self.dict_comp2ancestors = pickle.load(open(dict_comp2ancestors_file, 'rb'))
        self.dict_grid2non_ancestor_negative = pickle.load(open(dict_grid2non_ancestor_negative_file, 'rb'))
        self.large_comp_idx = pickle.load(open(large_comp_idx_file, 'rb'))

        dict_comp2split_file = os.path.join(self.dataset_dir, "comp2split.dict")
        dict_split2comp_file = os.path.join(self.dataset_dir, "split2comp.dict")
        dict_grid2negatives_outside_component_file = os.path.join(self.dataset_dir, "grid2negatives_outside_component.dict")
        dict_grid2random_negatives_file = os.path.join(self.dataset_dir, "grid2random_negatives.dict")
        dict_grid2negatives_file = os.path.join(self.dataset_dir, "grid2negatives.dict")

        self.dict_comp2split = pickle.load(open(dict_comp2split_file, 'rb'))
        self.dict_split2comp = pickle.load(open(dict_split2comp_file, 'rb'))
        self.dict_grid2negatives_outside_component = pickle.load(open(dict_grid2negatives_outside_component_file, 'rb'))
        self.dict_grid2random_negatives = pickle.load(open(dict_grid2random_negatives_file, 'rb'))
        self.dict_grid2negatives = pickle.load(open(dict_grid2negatives_file, 'rb'))


    def get_all_ancestors(self, parent_of, dict_comp2grid):
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
                    ancestors = self.find_ancestor(parent_of, grid_id)
                    num_anc += len(ancestors)
                    dict_id2ancestors[grid_id] = ancestors
                dict_comp2ancestors[comp_idx] = (num_anc, dict_id2ancestors)

                # Save the large comp idx
                if num_anc > 1000:
                    list_large_comp_idx.append(comp_idx)

            else:
                dict_comp2ancestors[comp_idx] = (0, {list_grid_id[0]: []})

        return dict_comp2ancestors, list_large_comp_idx


    def map_comp_2_split(self, dict_comp2ancestors, list_large_comp_idx):
        '''
        Creates a mapping from component to train/dev/test split

        :param dict_comp2ancestors: dictionary mapping comp_idx to dictionary of grid_id to its ancestors
        :param large_comp_idx: comp_idx of the largest component
        :return:
        '''
        # Note that in all split 0 = train, 1 = dev, 2 = test
        # Keeps mapping of component to split
        dict_comp2split = {}

        #
        # for large_comp_idx in list_large_comp_idx:
        #     dict_comp2split[large_comp_idx] = 0

        dict_split2comp = {}

        filtered_dict_comp2ancestor = list(filter(lambda x: dict_comp2ancestors[x][0] > 1, dict_comp2ancestors.keys()))
        num_keys = len(filtered_dict_comp2ancestor)

        if self.seed == 0:
            test_idx = range(int(.2*num_keys))
            dev_idx = range(int(.2*num_keys), int(.4*num_keys))
            train_idx = range(int(.4*num_keys), int(num_keys+1))
        elif self.seed == 1:
            train_idx = list(range(int(.2*num_keys)))
            test_idx = range(int(.2*num_keys), int(.4*num_keys))
            dev_idx = range(int(.4*num_keys), int(.6*num_keys))
            train_idx.extend(range(int(.6 * num_keys), int(num_keys+1)))
        elif self.seed == 2:
            train_idx = list(range(int(.4*num_keys)))
            test_idx = range(int(.4*num_keys), int(.6*num_keys))
            dev_idx = range(int(.6*num_keys), int(.8*num_keys))
            train_idx.extend(range(int(.8*num_keys), int(num_keys+1)))
        elif self.seed == 3:
            train_idx = range(int(.6*num_keys))
            test_idx = range(int(.6*num_keys), int(.8*num_keys))
            dev_idx = range(int(.8*num_keys), int(num_keys+1))
        elif self.seed == 4:
            train_idx = range(int(.2*num_keys), int(.8*num_keys))
            test_idx = range(int(.8*num_keys), int(num_keys+1))
            dev_idx = range(int(.2*num_keys))

        for comp_ctr, comp_idx in enumerate(filtered_dict_comp2ancestor):
            # Ensure that comp_idx has already been counted

            if comp_idx not in dict_comp2split and comp_idx not in self.large_comp_idx:
                if comp_ctr in train_idx:
                    cur_idx = 0
                elif comp_ctr in dev_idx:
                    cur_idx = 1
                elif comp_ctr in test_idx:
                    cur_idx = 2
                else:
                    raise ValueError("Invalid Comp Idx")

                dict_comp2split[comp_idx] = cur_idx

                if cur_idx in dict_split2comp:
                    dict_split2comp[cur_idx].append(comp_idx)
                else:
                    dict_split2comp[cur_idx] = [comp_idx]

        return dict_comp2split, dict_split2comp

    def inverse_dict(self, dict):
        '''
        Finds the inverse of a dictionary

        :param dict:
        :return:
        '''
        inv_dict = {}

        for (k, v) in dict.items():
            # Remove null city, state, and countries
            if len(v) > 0:
                if v in inv_dict:
                    inv_dict[v].append(k)
                else:
                    inv_dict[v] = [k]

        return inv_dict

    def find_non_ancestor_negatives(self, dict_grid2comp, dict_comp2grid, dict_comp2ancestors):
        '''
        Finds negatives which are non ancestors for each grid id

        :param dict_grid2comp: dictionary of grid_id to comp_idx
        :param dict_comp2grid: dictionary of comp_idx to grid_id
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

    def get_split2grid(self, dict_split2comp, dict_comp2grid):
        '''
        Get dictionary of split to list of grid_id

        :return:
        '''

        # Get dictionary of split to the grid_id in that split
        dict_split2grid = {}
        for (split_idx, list_comp_id) in dict_split2comp.items():
            set_grid_id = set()

            for comp_id in list_comp_id:
                set_grid_id.update(dict_comp2grid[comp_id])

            dict_split2grid[split_idx] = set_grid_id

        # Get dictionary of split to the grid_id non in that split
        dict_split2non_grid = {}
        all_splits = list(dict_split2grid.keys())
        for split_id in all_splits:
            non_splits = all_splits.copy()
            non_splits.remove(split_id)

            non_grid = set()
            for split in non_splits:
                non_grid.update(dict_split2grid[split])
            dict_split2non_grid[split_id] = non_grid

        return dict_split2grid, dict_split2non_grid


    def find_negatives_outside_component(self, dict_grid2comp, dict_comp2grid, dict_comp2split, dict_split2grid, dict_split2non_grid):
        '''
        Finds negatives outside the component for each grid with the same city, state, or country

        :param dict_grid2comp: dictionary of grid_id to comp_idx
        :param dict_comp2grid: dictionary of comp_idx to grid_id
        :param dict_comp2split: dictionary of comp_idx to split_idx
        :param dict_split2grid:
        :param dict_split2non_grid
        :return:
        '''
        dict_grid2negatives = {}


        dict_grid2city = read_grid_csv("city")
        dict_city2grid = self.inverse_dict(dict_grid2city)
        dict_grid2state = read_grid_csv("state")
        dict_state2grid = self.inverse_dict(dict_grid2state)
        dict_grid2country = read_grid_csv("country")
        dict_country2grid = self.inverse_dict(dict_grid2country)

        for (grid_id, comp_idx) in dict_grid2comp.items():

            # Only find negatives for grid_id in dict_comp2split, which means they are train/dev/test
            if comp_idx in dict_comp2split:
                set_negative_grid = set()

                # Add grid_ids with the same city
                city = dict_grid2city[grid_id]
                if city in dict_city2grid:
                    same_city_grid = dict_city2grid[city]
                    set_negative_grid.update(same_city_grid)

                # Add grid_ids with the same state
                state = dict_grid2state[grid_id]
                if state in dict_state2grid:
                    same_state_grid = dict_state2grid[state]
                    set_negative_grid.update(same_state_grid)

                # Add grid_ids with the same country
                country = dict_grid2country[grid_id]
                if country in dict_country2grid:
                    same_country_grid = dict_country2grid[country]
                    set_negative_grid.update(same_country_grid)

                # Remove grid_id in the same component
                set_negative_grid = set_negative_grid.difference(set(dict_comp2grid[comp_idx]))\
                # Remove grid_id itself as a negative
                set_negative_grid = set_negative_grid.difference(set([grid_id]))
                # # Ensure grid_ids are in the same split by removing the grid_id not in that split
                # split_id = dict_comp2split[comp_idx]
                # set_negative_grid = set_negative_grid.difference(dict_split2non_grid[split_id])
                # set_negative_grid = set_negative_grid.intersection(dict_split2grid[split_id])

                list_negative_grid = list(set_negative_grid)
                if len(list_negative_grid) > 1000:
                    list_negative_grid = random.sample(list_negative_grid, 1000)

                dict_grid2negatives[grid_id] = list_negative_grid



        return dict_grid2negatives


    def find_ancestor(self, parent_of, grid_id):
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
                ancestors = self.find_ancestor(parent_of, parent_grid_id)
                list_ancestors.extend(ancestors)

        return list_ancestors

    def iterate_train_data(self, query_grid_id, positive_grid_id, negative_grid_id, dictid2name):
        '''
        Iterate through cross product of pos and neg to generate train data

        :param query_grid_id: query grid_id
        :param positive_grid_id: ancestors of query grid_id
        :param negative_grid_id: non ancestors of query grid_id
        :param dictid2name: dictionary of grid_id to name

        :return: list of data
        '''
        data = []

        # Writes training examples as (qry, pos, neg)
        for p in positive_grid_id:
            if p != query_grid_id:
                for n in negative_grid_id:
                    triplet = [query_grid_id, p, n]
                    triplet.extend(list(map(lambda x: dictid2name[x], triplet)))
                    data.append(triplet)

        return data

    def iterate_dev_test_data(self, query_grid_id, positive_grid_id, negative_grid_id, dictid2name):
        '''
        Generate data

        :param query_grid_id: query grid_id
        :param positive_grid_id: ancestors of query grid_id
        :param negative_grid_id: non ancestors of query grid_id
        :param dictid2name: dictionary of grid_id to name

        :return: list of data
        '''
        data = []

        # Writes dev/test examples as (qry, pos/neg, 1/0)
        for p in positive_grid_id:
            if p != query_grid_id:
                double = [query_grid_id, p]
                double.append(1)
                double.extend(list(map(lambda x: dictid2name[x], double[:-1])))
                data.append((double))

        for n in negative_grid_id:
            double = [query_grid_id, n]
            double.append(0)
            double.extend(list(map(lambda x: dictid2name[x], double[:-1])))
            data.append(double)

        return data


    def randomly_sample_negatives(self, dict_grid2comp, dict_comp2grid, dict_split2comp, dict_comp2split):
        '''
        Randomly sample grid_ids in the same component as negatives

        :param dict_grid2comp: dictionary of grid_id to comp_id
        :param dict_comp2grid: dictionary of comp_id to list of grid_id
        :param dict_split2comp: dictionary of split_id to list of comp_id
        :param dict_comp2split: dictionary of comp_id to split_id
        :return:
        '''
        dict_grid2negatives = {}

        for (grid_id, comp_id) in dict_grid2comp.items():
            if comp_id in dict_comp2split:

                # Get all components in the same split
                split_id = dict_comp2split[comp_id]
                list_comp_id = dict_split2comp[split_id]

                # Add all grid_ids in the same component
                set_grid_id = set()
                for comp_id in list_comp_id:
                    set_grid_id.update(dict_comp2grid[comp_id])

                # Remove grid_id itelf as a negative
                set_grid_id = set_grid_id.difference(set([grid_id]))

                list_grid_id = list(set_grid_id)

                if len(list_grid_id) > 1000:
                    list_grid_id = random.sample(list_grid_id, 1000)

                dict_grid2negatives[grid_id] = list_grid_id

        return dict_grid2negatives

    def merge_negatives(self, dict_grid2non_ancestor_negative, dict_grid2negatives_outside_component, dict_grid2random_negatives):
        '''
        Merges the negatives from both dictionary giving priority to non_descendant negatives

        :param dict_grid2non_descendant_negative:
        :param dict_grid2negatives_outside_component:
        :param dict_grid2random_negatives
        :return: dictionary of grid_id to exactly 10 negatives
        '''
        dict_grid2negatives = {}

        num_negatives = []

        for (grid_id, list_non_ancestor_negatives) in dict_grid2non_ancestor_negative.items():
            # Only create negatives for grid_id which are in component
            if grid_id in dict_grid2negatives_outside_component and grid_id in dict_grid2random_negatives:
                list_negatives_outside_component = dict_grid2negatives_outside_component[grid_id]

                random_negatives = dict_grid2random_negatives[grid_id]
                all_negatives = list_negatives_outside_component + random_negatives + list_non_ancestor_negatives

                all_negatives = list(set(all_negatives).difference(set([grid_id])))

                if grid_id in all_negatives:
                    print("BUG: GRID ID found negatives")

                num_negatives.append(len(all_negatives))
                dict_grid2negatives[grid_id] = all_negatives

        return dict_grid2negatives

    def gen_data(self, list_comp_idx, dict_comp2ancestors, dict_grid2negatives, dict_id2name, is_train=True):
        '''
        Generate the data

        :param list_comp_idx: list of comp_idx in train
        :param dict_comp2ancestors: dict of grid_id 2 positives
        :param dict_grid2negatives: dict of grid_id 2 negatives
        :param dict_id2name:
        :return: data (qry_id, pos_id, neg_id, qry_name, pos_name, neg_name)
        :return: data (qry_id, cnd_id, lbl, qry, cnd)
        '''
        data = []

        for comp_idx in list_comp_idx:
            for (grid_id, positives) in dict_comp2ancestors[comp_idx][1].items():
                if grid_id in dict_grid2negatives and len(positives) > 0:
                    negatives = dict_grid2negatives[grid_id]
                    negatives = list(set(negatives).difference(set(positives)))

                    if len(negatives) > 0:
                        if is_train:
                            temp_data = self.iterate_train_data(grid_id, positives, negatives, dict_id2name)
                        else:
                            temp_data = self.iterate_dev_test_data(grid_id, positives, negatives, dict_id2name)
                        data.extend(temp_data)

        return data

    def write_dataset(self, data, file):
        '''
        Write the data to a file

        :param data:
        :param file:
        :return:
        '''

        with open(file, 'w+') as f:
            for line in data:
                str_line = map(lambda x: str(x), line)
                f.write('\t'.join(str_line) + '\n')

    def create_datasets(self):
        '''
        Create all the datasets

        :return:
        '''
        self.create_dict()

        train_data = self.gen_data(self.dict_split2comp[0], self.dict_comp2ancestors,
                                        self.dict_grid2negatives, self.dict_id2name)
        pickle.dump(train_data, open(os.path.join(self.dataset_dir, "train.list"), 'wb'))
        self.write_dataset(train_data, os.path.join(self.dataset_dir, "train.data"))

        dev_data = self.gen_data(self.dict_split2comp[1], self.dict_comp2ancestors,
                                        self.dict_grid2negatives, self.dict_id2name, is_train=False)
        pickle.dump(dev_data, open(os.path.join(self.dataset_dir, "dev.list"), 'wb'))
        self.write_dataset(dev_data, os.path.join(self.dataset_dir, "dev.data"))

        test_data = self.gen_data(self.dict_split2comp[2], self.dict_comp2ancestors, self.dict_grid2negatives, self.dict_id2name, is_train=False)
        pickle.dump(test_data, open(os.path.join(self.dataset_dir, "test.list"), 'wb'))
        self.write_dataset(test_data, os.path.join(self.dataset_dir, "test.data"))

    def shard_dev_test(self):
        '''
        Shard all the datasets

        :return:
        '''
        # self.create_dict()
        self.load_dict()

        dev_shards_dir = os.path.join(self.dataset_dir, "dev_shards")
        if not os.path.exists(dev_shards_dir):
            os.makedirs(dev_shards_dir)
        dev_data = pickle.load(open(os.path.join(self.dataset_dir, "dev.list"), 'rb'))

        test_shards_dir = os.path.join(self.dataset_dir, "test_shards")
        if not os.path.exists(test_shards_dir):
            os.makedirs(test_shards_dir)
        test_data = pickle.load(open(os.path.join(self.dataset_dir, "test.list"), 'rb'))

        self.shard(5, dev_data, dev_shards_dir)
        self.shard(5, test_data, test_shards_dir)


    def shard(self, num_shard, dataset, shard_dir):
        '''
        Shard

        :param num_shard:
        :param data:
        :param shard_dir:
        :return:
        '''

        dict_shard_data = [[]] * num_shard

        new_qry_ctr = 0
        cur_qry = None

        for data in dataset:
            if data[0] != cur_qry:
                new_qry_ctr += 1
                cur_qry = data[0]
            dict_shard_data[new_qry_ctr % num_shard].append(data)

        for shard_idx, shard_data in enumerate(dict_shard_data):
            pickle.dump(shard_data, open(os.path.join(shard_dir, "shard_%d.list" % shard_idx), 'wb'))
            self.write_dataset(shard_data, os.path.join(shard_dir, "shard_%d.data" % shard_idx))

    def check_dataset_fold(self):
        '''
        Check the dataset are cross validation

        :return:
        '''

        self.load_dict()

        fixed_grid = 0
        for comp_idx in self.large_comp_idx:
            fixed_grid += len(self.dict_comp2grid[comp_idx])
        print("Num Fixed Grid: ", fixed_grid)
        print(len(self.large_comp_idx))
        split2comp_dict = pickle.load(open(os.path.join("data/cross_validation/fold_%d" % self.seed, "split2comp.dict"), 'rb'))

        for (prtn, list_comp_idx) in split2comp_dict.items():
            num_grid_idx = 0
            num_pos_idx = 0
            for comp_idx in list_comp_idx:
                num_grid_idx += len(self.dict_comp2grid[comp_idx])

                for grid_id in self.dict_comp2grid[comp_idx]:
                    if grid_id in self.parent_of:
                        num_pos_idx += len(self.parent_of[grid_id])

            list_sizes = []

            for comp_idx in list_comp_idx:
                list_sizes.append(len(self.dict_comp2grid[comp_idx]))


            print("Comp Size - Mean: ", np.mean(list_sizes), " Std: ", np.std(list_sizes))

            print("Prtn: ", prtn, " Num Grid Idx: ", num_grid_idx, " Num Comp Idx: ", len(list_comp_idx), " Num Pos Idx: ", num_pos_idx)



    def get_qry(self, filename, is_train=True):

        set_qry = set()

        with open(filename, 'r') as f:
            for line in f.readlines():
                tab_split = line.strip('\n').split('\t')
                set_qry.add(tab_split[0])
                if is_train:
                    set_qry.add(tab_split[1])
                else:
                    if int(tab_split[2]) == 1:
                        set_qry.add(tab_split[1])


        return set_qry


    def check_orig(self):

        self.load_dict()

        train_grid = self.get_qry(os.path.join("data", "ancestors", "train.txt"))
        dev_grid = self.get_qry(os.path.join("data", "ancestors", "dev.txt"), is_train=False)
        test_grid = self.get_qry(os.path.join("data", "ancestors", "test.txt"), is_train=False)


        prtn_grid = [train_grid, dev_grid, test_grid]

        for prtn_idx, list_grid in enumerate(prtn_grid):
            print("Partition: ", prtn_idx)
            print("Num Grid: ", len(list_grid))


            set_comp_idx = set()
            num_pos_idx = 0

            for grid_id in list_grid:
                comp_idx = self.dict_grid2comp[grid_id]
                set_comp_idx.add(comp_idx)

                if grid_id in self.parent_of:
                    num_pos_idx += len(self.parent_of[grid_id])

            print("Num Comp: ", len(set_comp_idx))
            print("Num Pos: ", num_pos_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fold")
    args = parser.parse_args()

    # dc = DatasetCreator(int(args.fold))
    # dc.create_datasets()




    for i in range(5):
        print("Fold: ", i)
        dc = DatasetCreator(i)
        dc.check_dataset_fold()

    # dc = DatasetCreator(0)
    # dc.check_orig()