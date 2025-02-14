import os
import torch
from scipy import sparse as sp
from loguru import logger
    
def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
def setup_directory(args):
    valid = args.valid
    
    args.data_dir = os.path.join('data', args.dataset, valid.split('_')[0])
    args.exp_dir = os.path.join('experiments', args.exp_name, args.dataset, valid)
    args.emb_dir = os.path.join(args.exp_dir, 'emb')
    make_directory(args.emb_dir)
    
    args.result_path = os.path.join(args.exp_dir, f'{args.algorithm}.csv')
    
    
def slice_sparse_tensor_columns(sparse_tensor, start_col, end_col):
    if not sparse_tensor.is_sparse:
        raise ValueError("Input tensor must be a sparse tensor.")
    
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    size = sparse_tensor.size()

    mask = (indices[1] >= start_col) & (indices[1] < end_col)
    
    new_indices = indices[:, mask]
    new_indices[1] -= start_col
    new_values = values[mask]
    new_size = (size[0], end_col - start_col)

    return torch.sparse_coo_tensor(new_indices, new_values, new_size)

def sparse_diag(input, non_zero=True):
    N = input.shape[0]
    values = input
    if input.is_sparse:
        values = torch.zeros(N, device=input.device)
        ind, val = input.indices(), input._values()
        values[ind] = val
    if non_zero:
        values[values==0] += 1e-4
    arr = torch.arange(N, device=values.device)
    indices = torch.stack([arr, arr])
    return torch.sparse_coo_tensor(indices, values, (N, N))

def create_batches(N, B):
    tensor = torch.arange(N)
    
    if N < B:
        return [tensor.tolist()]
    batches = tensor.split(B)
    
    return [batch.tolist() for batch in batches]

def to_scipy_sparse_from_torch_sparse(sparse_tensor):
    indices = sparse_tensor._indices().cpu().numpy()
    values = sparse_tensor._values().cpu().numpy()
    shape = sparse_tensor.shape
    return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)

def log_param(args):
    logger.info('Arguments:')
    for k, v in vars(args).items():
        logger.info(f'{k}: {v}')
    logger.info('\n')