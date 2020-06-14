import main.utils.data_pathnames as data_pathnames
data_pathnames.init()


def read_relationship_csv(check=False):
    '''
    Reads the parent child csv

    :param check: whether to check if the parent child dictionaries are symmetric
    :return parent_of: dict of parent to its children
    :return child_of: dict of child to its parents
    '''
    parent_of = {}
    child_of = {}

    with open(data_pathnames.RELATIONSHIP_CSV, 'r+') as f:
        # Skip first line
        f.readline()

        for line in f.readlines():
            #(id, relationship, related_id)
            split = line.strip('\n').split(',')

            # is child means is the same as parent of
            if(split[1] == "Parent"):
                if(split[0] in parent_of):
                    parent_of[split[0]].append(split[2])
                else:
                    parent_of[split[0]] = [split[2]]
            # is parent is the same as child of
            elif(split[1] == "Child"):
                if(split[0] in child_of):
                    child_of[split[0]].append(split[2])
                else:
                    child_of[split[0]] = [split[2]]

    # Checks that the child parent relationships are symetric
    if check:
        for (child, parents) in parent_of.items():
            for parent in parents:
                if(not child in child_of[parent]):
                    raise ValueError('Error: Parent Child Relationships not symmetric')

    return parent_of, child_of

def read_grid_csv(part_to_retrieve=None):
    '''
    Read the grid csv to get dict of id to name

    :param: part_to_retrieve
    :return dict_id2name: dictionary of id to the institution name
    '''
    dict_id2name = {}

    retrieve_idx = -1
    if part_to_retrieve == "institution_name":
        retrieve_idx = 1
    elif part_to_retrieve == "city":
        retrieve_idx = 2
    elif part_to_retrieve == "state":
        retrieve_idx = 3
    elif part_to_retrieve == "country":
        retrieve_idx = 4

    with open(data_pathnames.GRID_CSV, 'r+') as f:
        # Skip first line
        f.readline()

        for line in f.readlines():
            #(grid_id, institution_name, city, state, country)
            split = line.strip('\n').split(',')

            if retrieve_idx == -1:
                dict_id2name[split[0]] = ','.join(split[1:])
            else:
                dict_id2name[split[0]] = split[retrieve_idx]


    return dict_id2name

def read_type_csv():
    '''
    Reads the type csv

    :return: dict_id2alias_name: dictionary of grid_id to list of alias name
    '''

    dict_id2type = {}

    with open(data_pathnames.TYPES_CSV, 'r+') as f:
        # SKip first line
        f.readline()

        for line in f.readlines():
            # (grid_id, institution_name)
            split = line.strip('\n').split(',')

            if split[0] in dict_id2type:
                dict_id2type[split[0]].append(split[1])
            else:
                dict_id2type[split[0]] = [split[1]]
    return dict_id2type
