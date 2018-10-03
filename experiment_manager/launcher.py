import tqdm
import hashlib
import json
import os
import numpy as np
from multiprocessing import Pool as ThreadPool
from itertools import product
from copy import deepcopy


def run_multi_threaded(function, params):
    '''
    Run given function with different seeds in an appropriate directory

    :param function: the function to execute with each new set of params; signature (dir:str, params:dict, seed:int) -> None.
                        First step of that function should be to set the seed.
    :param params: the parameters dict to run in multi-threaded (exploration on this done through "explore_params")
    :return: None
    '''

    def get_id_for_dict(in_dict):
        # Transform a parameter dict into a 16 digits hash for easier storage
        # Forget n_seeds and n_threads, if there is no param named like that it will have no effect
        dict_filtered = {key: in_dict[key] for key in in_dict.keys() if key not in ['n_threads', 'n_seeds']}
        return hashlib.sha256(json.dumps(dict_filtered, sort_keys=True).encode('utf-8')).hexdigest()[:16]

    # Determine the hash for that particular experiment
    hash = get_id_for_dict(params)
    # Print a message for debugging
    print('Exp with id {} and params {}'.format(hash, params))
    out_dir = 'out/raw/{}'.format(hash)

    # Just in case two params gave exactly the same hash
    # If more than 2 exps with same hash, will fail (but should never happen)
    try:
        os.makedirs(out_dir)
        out_dir += '/'
    except FileExistsError:
        try:
            out_dir += '_dup'
            os.makedirs(out_dir)
            out_dir += '/'
        except FileExistsError:
            return

    with open(out_dir + 'params', 'w') as outfile:
        json.dump(params, outfile)

    pool = ThreadPool(params['n_threads'])

    print([out_dir for _ in range(params['n_seeds'])])
    print(range(int(params['n_seeds'])))

    _ = pool.starmap(function, zip(
            [out_dir for _ in range(params['n_seeds'])],
            [params for _ in range(params['n_seeds'])],
            range(int(params['n_seeds'])))
            )

    with open(out_dir + 'exited_naturally', 'w') as outfile:
        outfile.write('True')


def explore_params(function, base_params, search_grid):
    '''
    Generate all parameter combinations and run each using run_multi_threaded

    :param base_params: config from which we want to start exploring
    :param search_grid: dict {key: list of values} of values to test for each specified param
    :return:
    '''

    print('Using base configuration {}'.format(base_params))

    params_to_vary = list(search_grid.keys())
    n_variables = len(params_to_vary)
    n_values_per_param = [len(search_grid[p]) for p in params_to_vary]
    print('Total number of experiments : {}, make sure it is reasonable...'.format(np.prod(n_values_per_param)))
    all_values = list(product(*[search_grid[key] for key in params_to_vary]))

    for param_tuple in tqdm.tqdm(all_values):
        tmp = deepcopy(base_params)
        for i in range(n_variables):
            tmp[params_to_vary[i]] = param_tuple[i]
        print('Using variable parameters {}'.format(tmp))
        # Now, call multi-threaded simulation for these params (not optimal, we could start new threads as soon as 1
        # is done, but should be reasonable if n_threads divides n_seeds (if not, might have to wait for one thread
        # to do full simulation before starting the next batch of 12...)
        run_multi_threaded(function, tmp)

