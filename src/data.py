import numpy as np
import torch
import json, os
from dotmap import DotMap

def load_data(args):
    datas = DotMap()

    with open(os.path.join(args.data_dir, args.dataset, 'statistics.json'), 'r') as f:
        statistics = json.load(f)
        datas.update(statistics)
    
    datas.n_nodes = datas.n_users + datas.n_items
    datas.beh2idx = {k: v for v, k in enumerate(datas.behaviors)}
    
    datas.A = load_behavior_adjmat(args, datas)
    datas.train, datas.test = load_train_test(args)
    return datas
    
def load_behavior_adjmat(args, datas):
    A = dict()
    for behavior in datas.behaviors:
        beh_file = os.path.join(args.data_dir, args.dataset, f'{behavior}.txt')
        edges = torch.from_numpy(np.loadtxt(beh_file, dtype=np.int32)).to(args.device)
        adj = edge_to_sparse_adj(edges, datas.n_users, datas.n_items)
        A[behavior] = adj
    return A

def edge_to_sparse_adj(edge, n_users, n_items):
    index, val = edge[:, :2].T, torch.ones(edge.shape[0], device=edge.device)
    return torch.sparse_coo_tensor(index, val, (n_users, n_items), dtype=torch.float32)        

def load_train_test(args):
    train_file = os.path.join(args.data_dir, args.dataset, 'train.txt')
    test_file = os.path.join(args.data_dir, args.dataset, 'test.txt')
    
    train = np.loadtxt(train_file, dtype=np.int32)
    test = np.loadtxt(test_file, dtype=np.int32)
    
    return train, test