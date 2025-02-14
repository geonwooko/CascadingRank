import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['taobao', 'tenrec', 'tmall'], help='dataset')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--alpha', type=float, default=0.0, help='alpha')
    parser.add_argument('--beta', type=float, default=0.9, help='beta')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='tolerance')
    parser.add_argument('--max_iter', type=int, default=100, help='max iteration')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--ks', type=list, default=[10, 30, 50, 100, 200], help='ks')
    return parser.parse_args()