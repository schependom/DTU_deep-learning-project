import numpy as np
import json

for split in ['train', 'test']:
    path = f'src/data/hanoi_action/{split}/'
    meta = json.load(open(path + 'dataset.json'))
    print(f'\n{split.upper()}:')
    print(f'  seq_len: {meta["seq_len"]}')
    print(f'  vocab_size: {meta["vocab_size"]}')
    print(f'  total_groups: {meta["total_groups"]}')
    print(f'  total_puzzles: {meta["total_puzzles"]}')
    
    inputs = np.load(path + 'all__inputs.npy')
    print(f'  inputs shape: {inputs.shape}')