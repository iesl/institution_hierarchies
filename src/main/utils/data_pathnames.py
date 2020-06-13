import os

def init():

    global CSV_DIR; CSV_DIR = os.path.join("data", "csv")
    global DICT_DIR; DICT_DIR = os.path.join("data", "dict")

    global GRID_CSV; GRID_CSV = os.path.join(CSV_DIR, "grid.csv")
    global RELATIONSHIP_CSV; RELATIONSHIP_CSV = os.path.join(CSV_DIR, "relationships.csv")
    global TYPES_CSV; TYPES_CSV = os.path.join(CSV_DIR, "types.csv")

    global GRID2INST_NAME;  GRID2INST_NAME = os.path.join(DICT_DIR, "grid2inst_name.dict")
    global GRID2CITY; GRID2CITY = os.path.join(DICT_DIR, "grid2city.dict")
    global GRID2STATE; GRID2STATE = os.path.join(DICT_DIR, "grid2state.dict")
    global GRID2COUNTRY; GRID2COUNTRY = os.path.join(DICT_DIR, "grid2country.dict")
    global GRID2TYPE; GRID2TYPE = os.path.join(DICT_DIR, "grid2type.dict")