from main.utils.read_csv import read_relationship_csv, read_grid_csv
import main.utils.data_pathnames as data_pathnames

from main.objects.Tokenizer import Char
from main.objects.Tokenizer import Unigram

import argparse
import random
import os
import pickle


class CrossValidationFold(object):

    def __init__(self, dataset_seed):
        '''
        :param seed:
        '''
        random.seed(0)
        self.seed = dataset_seed
        self.fold_dir = os.path.join("data", "cross_validation", "fold_%d" % self.seed)
        if not os.path.exists(self.fold_dir):
            os.makedirs(self.fold_dir)

        self.train_data = None
        self.dev_data = None
        self.test_data = None

    def load_comp_dict(self):
        '''
        Load component dictionary
        '''

        self.dict_id2name = read_grid_csv()
        self.parent_of, self.child_of = read_relationship_csv()

        self.dict_grid2comp = pickle.load(open(data_pathnames.DICT_GRID2COMP, 'rb'))
        self.dict_comp2grid = pickle.load(open(data_pathnames.DICT_COMP2GRID, 'rb'))
        self.dict_comp2ancestors = pickle.load(open(data_pathnames.DICT_COMP2ANCESTORS, 'rb'))
        self.large_comp_idx = pickle.load(open(data_pathnames.LARGE_COMP_IDX, 'rb'))
        self.dict_grid2non_ancestor_negative_same_comp = pickle.load(open(data_pathnames.DICT_GRID2NONANCESTOR_SAME_COMP_NEGATIVE, 'rb'))

    def init_fold_dict(self):
        '''
        Create dictionaries to create datasets
        '''

        self.dict_comp2split, self.dict_split2comp = self.map_comp_2_split(self.dict_comp2ancestors)
        self.dict_split2grid, self.dict_split2non_grid = self.get_split2grid(self.dict_split2comp, self.dict_comp2grid)

        self.dict_grid2negatives_same_loc = self.find_negatives_same_loc(self.dict_grid2comp, self.dict_comp2grid, self.dict_comp2split)
        self.dict_grid2random_negatives = self.randomly_sample_negatives(self.dict_grid2comp, self.dict_comp2grid, self.dict_split2comp, self.dict_comp2split)
        self.dict_grid2negatives = self.merge_negatives(self.dict_grid2non_ancestor_negative_same_comp, self.dict_grid2negatives_same_loc, self.dict_grid2random_negatives)


    def map_comp_2_split(self, dict_comp2ancestors):
        '''
        Creates a mapping from component to train/dev/test split

        :param dict_comp2ancestors: dictionary mapping comp_idx to dictionary of grid_id to its ancestors
        :param large_comp_idx: comp_idx of the largest component
        :return:
        '''
        # Note that in all split 0 = train, 1 = dev, 2 = test
        dict_comp2split = {}
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


    def find_negatives_same_loc(self, dict_grid2comp, dict_comp2grid, dict_comp2split):
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

                list_negative_grid = list(set_negative_grid)
                if len(list_negative_grid) > 1000:
                    list_negative_grid = random.sample(list_negative_grid, 1000)

                dict_grid2negatives[grid_id] = list_negative_grid


        return dict_grid2negatives

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

    def make_vocab(self, train_file, write_file, tokenizer_name):
        '''
        Counts the number of each token in the file and then writes the tokens that occur more than a threshold to form the vocab

        param read_file: file to read tokens
        param write_file: file to write vocab
        param tokenizer_name: tokenizer to use
        '''
        tokenizer = None
        if (tokenizer_name == "Char"):
            tokenizer = Char()
        elif (tokenizer_name == "Unigram"):
            tokenizer = Unigram()
        else:
            raise ValueError("Invalid Tokenizer")

        token_dict = {}

        with open(train_file, 'r+') as rf:
            for line in rf:
                splt = line.strip().split("\t")
                # First 3 tab splits are strings
                for s in splt[:3]:
                    s_tokens = tokenizer.tokenize(s)
                    for token in s_tokens:
                        if token not in token_dict:
                            token_dict[token] = 1
                        else:
                            token_dict[token] += 1

        with open(write_file, "w+", "UTF-8") as wf:
            token_id = 2
            for token in token_dict.keys():
                if token_dict[token] >= 10:
                    wf.write("{}\t{}\n".format(token, token_id))
                    token_id += 1
            wf.flush()
            wf.close()

    def create_datasets(self):
        '''
        Create all the datasets

        :return:
        '''
        self.load_comp_dict()
        self.init_fold_dict()

        train_data = self.gen_data(self.dict_split2comp[0], self.dict_comp2ancestors,
                                        self.dict_grid2negatives, self.dict_id2name)
        pickle.dump(train_data, open(os.path.join(self.fold_dir, "train.list"), 'wb'))
        self.write_dataset(train_data, os.path.join(self.fold_dir, "train.data"))
        self.make_vocab(os.path.join(self.fold_dir, "train.data"), os.path.join(self.fold_dir, "vocab_char"), "Char")
        self.make_vocab(os.path.join(self.fold_dir, "train.data"), os.path.join(self.fold_dir, "vocab_unigram"), "Unigram")

        dev_data = self.gen_data(self.dict_split2comp[1], self.dict_comp2ancestors,
                                        self.dict_grid2negatives, self.dict_id2name, is_train=False)
        pickle.dump(dev_data, open(os.path.join(self.fold_dir, "dev.list"), 'wb'))
        self.write_dataset(dev_data, os.path.join(self.fold_dir, "dev.data"))

        test_data = self.gen_data(self.dict_split2comp[2], self.dict_comp2ancestors, self.dict_grid2negatives, self.dict_id2name, is_train=False)
        pickle.dump(test_data, open(os.path.join(self.fold_dir, "test.list"), 'wb'))
        self.write_dataset(test_data, os.path.join(self.fold_dir, "test.data"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    for i in range(5):
        dc = CrossValidationFold(i)
        dc.create_datasets()