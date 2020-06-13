from collections import defaultdict

def get_connected_components(dict_id2name, parent_of, child_of):
    '''
    Gets the connected components such that child and parents are in the same component

    :param dict_id2name: dictionary of grid_id to name
    :param parent_of: dictionary of parent to children
    :param child_of: dictionary of children to parent
    :return dict_grid2comp: dictionary of grid_id to component_id
    :return dict_comp2grid: dictionary of component_id to list of grid_id
    '''
    dict_grid2comp = defaultdict(lambda: -1)
    comp_id = 0

    for grid_id in dict_id2name.keys():
        # Ensure vertex not already visited
        if(dict_grid2comp[grid_id] == -1):
            list_neighbors = get_connected_vertices(grid_id, parent_of, child_of, dict_grid2comp, comp_id)
            for n in list_neighbors:
                dict_grid2comp[n] = comp_id
        # break
        comp_id += 1
    dict_comp2grid = {}
    for (grid_id, comp_id) in dict_grid2comp.items():
        if(comp_id in dict_comp2grid):
            dict_comp2grid[comp_id].append(grid_id)
        else:
            dict_comp2grid[comp_id] = [grid_id]

    #Check each grid_id in exactly 1 connected component
    grid_hit = {}

    for (comp_id, list_grid_id) in dict_comp2grid.items():
        for grid_id in list_grid_id:
            if(grid_id in grid_hit):
                raise ValueError('Error: grid id in more than 1 conencted component')
            grid_hit[grid_id] = True

    # Check parent of and child of relationships satisfy being in the same component
    for (grid_id, list_parents) in parent_of.items():
        for parent_id in list_parents:
            if(dict_grid2comp[grid_id] != dict_grid2comp[parent_id]):
                raise ValueError('Error: grid id different for is child relationship')
    for (grid_id, list_children) in child_of.items():
        for child_id in list_children:
            if(dict_grid2comp[grid_id] != dict_grid2comp[child_id]):
                raise ValueError('Error: grid id different for is parent relationship')

    return dict(dict_grid2comp), dict_comp2grid

def get_connected_vertices(grid_id, parent_of, child_of, dict_grid2comp, comp_id):
    '''
    Gets the vertices connected to a grid id

    :param grid_id: grid id to get neighbors of
    :param parent_of: dictionary of parents to children
    :param child_of: dictionary of children to parents
    :param dict_grid2comp: dictionary of grid_id to component_id
    :param comp_id: component_id to assign the grid_id
    :return: list of neighbors of grid_id to give the same component_id
    '''

    list_neighbors = []
    dict_grid2comp[grid_id] = comp_id

    if grid_id in parent_of:
        for neighbor in parent_of[grid_id]:
            list_neighbors.append(neighbor)
            if(dict_grid2comp[neighbor] == -1):
                list_neighbors.extend(get_connected_vertices(neighbor, parent_of, child_of, dict_grid2comp, comp_id))

    if grid_id in child_of:
        for neighbor in child_of[grid_id]:
            list_neighbors.append(neighbor)
            if(dict_grid2comp[neighbor] == -1):
                list_neighbors.extend(get_connected_vertices(neighbor, parent_of, child_of, dict_grid2comp, comp_id))

    return list_neighbors

def get_root_nodes(list_grid_id, child_of):
    '''
    Gets the root nodes from nodes

    :param list_grid_id: list of nodes to get root from
    :param child_of: dictionary of node to which nodes its a child of
    :return: the root nodes
    '''
    root_nodes = set()

    for grid_id in list_grid_id:
        if grid_id in child_of:
            root_nodes.add(grid_id)

    return root_nodes
