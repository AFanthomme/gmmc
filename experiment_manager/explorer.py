import os
import numpy as np
import pandas as pd
import json



def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def make_index():
    # Explore the raw folder and build a pandas frame with relevant run informations
    hashes = get_immediate_subdirectories('out/raw')
    all_keys = set()
    dict_of_dicts = {}

    for hash in hashes:
        with open('out/raw/{}/params'.format(hash), 'r') as outfile:
            params = json.load(outfile)
        dict_of_dicts[hash] = params

    database = pd.DataFrame.from_dict(dict_of_dicts)
    database.to_pickle('out/raw/parameters_database.pkl')
    database.to_string(open('out/raw/parameters_database_human_readable.txt', mode='w+'))
    database.to_csv(open('out/raw/parameters_database.csv', mode='w+'))
    print('Current index table : {}'.format(database))

def get_siblings(ref_hash, traversal_key):
    # Take the hash of a reference experiment and return list of hashes such that only 'traversal_key' differs
    db = pd.read_pickle('out/raw/parameters_database.pkl')
    all_keys = list(db.keys())
    hashes = db.index.values.tolist()
    siblings = []
    values = []

    with open('out/raw/{}/params'.format(ref_hash), 'r') as outfile:
        ref_params = json.load(outfile)

    for hash in hashes:
        is_sibling = True
        with open('out/raw/{}/params'.format(hash), 'r') as outfile:
            params = json.load(outfile)
        for key in all_keys:
            if params[key] != ref_params[key] and key != traversal_key:
                is_sibling = False
                break
        if is_sibling:
            siblings.append(hash)
            values.append(params[traversal_key])

    print('To vary parameter {} in {}, visit {}'.format(traversal_key, values, siblings))


    return siblings
