import os
import numpy as np
import pickle
import glob
from tqdm import tqdm


def read_DJ01_ctrls(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [list(map(float,l.strip(' \n').split(' '))) for l in lines]
    return np.array(lines).squeeze()

def load_rigs_to_cache(root, n_rig, version_old=None):
    n_folders = len(os.listdir(root))    
    pkl_path = root + f'{n_folders}.pkl'
    print('loading rig pkl:', pkl_path)
    rigs = {}

    load_rig = read_DJ01_ctrls
    print(root)

    if version_old:
        pkl_path_old = root + f'{version_old}.pkl'
        with open(pkl_path_old, 'rb') as f:
            rigs = pickle.load(f)
    
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            rigs = pickle.load(f)
    else:
        print('=> loading rigs to cache..')
        # folders = ['linjie_expr_test3']
        # files_name = []
        # for fold in folders:
        #     files_name += [y for x in os.walk(os.path.join(root, fold)) for y in glob.glob(os.path.join(x[0], '*.txt'))]

        files_name = [y for x in os.walk(root) for y in glob.glob(os.path.join(x[0], '*.txt'))]
        files_name = [fn.replace(root, '').strip('/') for fn in files_name]
        files_name = [fn for fn in files_name if fn not in rigs.keys()]
        # rigs = {}
        for fname in tqdm(files_name, total=len(files_name)):
            try:
                rig = load_rig(os.path.join(root, fname))
            except:
                print(fname)
                exit()

            rigs[fname] = rig
        with open(pkl_path, 'wb' ) as f:
            pickle.dump(rigs, f)
    return rigs