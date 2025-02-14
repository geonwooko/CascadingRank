from loguru import logger
from utils import log_param
from parser import parse_args
from data import load_data
from evaluate import evaluate
from cascadingrank import CascadingRank

def main(args):
    log_param(args)
    
    logger.info('Loading data...')
    data = load_data(args)
    
    logger.info('Running CascadingRank')
    model = CascadingRank(args, data)
    ranking_scores, converged_iterations = model.run()
    
    logger.info('Evaluating...')
    hr, ndcg = evaluate(ranking_scores, data, args)
    
    for k in args.ks:
        logger.info(f'HR@{k}: {hr[k-1]:.4f}, NDCG@{k}: {ndcg[k-1]:.4f}')
    logger.info(f'Converged iterations: {converged_iterations}')
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)