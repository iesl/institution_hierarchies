import os

def init():

    global CSV_DIR; CSV_DIR = os.path.join("data", "csv")
    global GRID_DICT_DIR; GRID_DICT_DIR = os.path.join("data", "dict", "grid_dict")

    global GRID_CSV; GRID_CSV = os.path.join(CSV_DIR, "grid.csv")
    global RELATIONSHIP_CSV; RELATIONSHIP_CSV = os.path.join(CSV_DIR, "relationships.csv")
    global TYPES_CSV; TYPES_CSV = os.path.join(CSV_DIR, "types.csv")

    global GRID2INST_NAME;  GRID2INST_NAME = os.path.join(GRID_DICT_DIR, "grid2inst_name.dict")
    global GRID2CITY; GRID2CITY = os.path.join(GRID_DICT_DIR, "grid2city.dict")
    global GRID2STATE; GRID2STATE = os.path.join(GRID_DICT_DIR, "grid2state.dict")
    global GRID2COUNTRY; GRID2COUNTRY = os.path.join(GRID_DICT_DIR, "grid2country.dict")
    global GRID2TYPE; GRID2TYPE = os.path.join(GRID_DICT_DIR, "grid2type.dict")


    global COMP_DICT_DIR; COMP_DICT_DIR = os.path.join("data", "dict", "comp_dict")

    global DICT_GRID2COMP; DICT_GRID2COMP = os.path.join(COMP_DICT_DIR, "grid2comp.dict")
    global DICT_COMP2GRID; DICT_COMP2GRID = os.path.join(COMP_DICT_DIR, "comp2grid.dict")
    global DICT_COMP2ANCESTORS; DICT_COMP2ANCESTORS = os.path.join(COMP_DICT_DIR, "comp2ancestors.dict")
    global DICT_GRID2NONANCESTOR_SAME_COMP_NEGATIVE; DICT_GRID2NONANCESTOR_SAME_COMP_NEGATIVE = os.path.join(COMP_DICT_DIR, "grid2nonancestor_same_comp_negative.dict")
    global LARGE_COMP_IDX; LARGE_COMP_IDX = os.path.join(COMP_DICT_DIR, "large_comp_idx.dict")