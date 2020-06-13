import random
import os
import sys
import shutil
import argparse


def split_set(filepath, num_shards):
    '''
    Shards the data in file to multiple files, ensuring that all the lines with the same query are grouped together

    :param filepath: filepath of file to shard
    :param num_shards: number of shards
    '''
    map_query_2_lines = get_map_query_2_lines(filepath)

    # Sharding test file
    if "test" in filepath:
        data_shard_folder = os.path.join(os.path.split(filepath)[0], "test_shards")
    # Sharding dev file
    else:
        data_shard_folder = os.path.join(os.path.split(filepath)[0], "dev_shards")

    # Remove previous shards
    if(os.path.exists(data_shard_folder)):
        shutil.rmtree(data_shard_folder)
    os.makedirs(data_shard_folder)

    for counter, (query, list_of_entities) in enumerate(map_query_2_lines.items()):
        shard = counter % int(num_shards)

        shard_filename = os.path.join(data_shard_folder, "shard" + "_" + str(shard))

        with open(shard_filename, 'a+') as f:
            for entities in list_of_entities:
                f.write('\t'.join(entities) + '\n')


def get_map_query_2_lines(filepath):
    '''
    Gets a dictionary of query to all the lines that include the query

    param filepath: filepath to data file 
    return: dictionary of query to lines 
    '''
    map_query_2_lines = {}

    with open(filepath, 'r') as file:        
        for line in file.readlines():
            entities = line.strip('\n').split('\t')
            query = entities[0]
            if query in map_query_2_lines.keys():
                map_query_2_lines[query].append(entities)
            else:
                map_query_2_lines[query] = [entities]

    return map_query_2_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-s", "--num_shards", required=True)
    args = parser.parse_args()

    split_set(args.file, args.num_shards)
