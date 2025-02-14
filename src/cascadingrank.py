import torch        
from utils import sparse_diag, create_batches, slice_sparse_tensor_columns
from tqdm.autonotebook import tqdm
import numpy as np

class CascadingRank:
    def __init__(self, args, datas):
        self.datas = datas
        self.n_users, self.n_items = datas['n_users'], datas['n_items']
        self.alpha, self.beta = args.alpha, args.beta
        self.tolerance, self.max_iter = args.tolerance, args.max_iter
        self.batch_size = args.batch_size
        self.device = args.device
        self.setup_query()
        
    @torch.no_grad()
    def setup_query(self):
        I = sparse_diag(torch.ones(self.n_users, device=self.device)) 
        
        self.qu = []
        self.qi = []
        self.A = []
        
        for k, beh in enumerate(self.datas['behaviors']):
            Ab = self.datas['A'][beh] # |U| X |I|
                        
            Du = sparse_diag(Ab.sum(axis=1)) 
            Di = sparse_diag(Ab.T.sum(axis=1))
            
            qub = I.coalesce()
            # Column-normalize matrix (item query)
            qib = (Ab.T @ (torch.pow(Du, -1))).coalesce() 
            
            # Symmetric-normalized matirx (smoothing)
            Ab_tilde = torch.pow(Du, -0.5) @ Ab @ (torch.pow(Di, -0.5))
            
            self.qu.append(qub)
            self.qi.append(qib)
            self.A.append(Ab_tilde) 
            
    @torch.no_grad()
    def run(self):
        n_users, n_items = self.n_users, self.n_items
        batch_size = self.batch_size
            
        ranking_scores = []        
        converged_iterations_behs = []
        # Batch
        for i, batch in enumerate(tqdm(create_batches(n_users, batch_size), desc=f'Batch')):
            start_idx, end_idx = batch[0], batch[-1]+1
            converged_iterations_batch = []
            # Behavior Cascading
            for j, beh in enumerate(self.datas['behaviors']):
                qub = slice_sparse_tensor_columns(self.qu[j], start_idx, end_idx).to_dense() # |U| X B
                qib = slice_sparse_tensor_columns(self.qi[j], start_idx, end_idx).to_dense() # |I| X B
                
                if j == 0:
                    rub_casc = qub.clone()
                    rib_casc = qib.clone()
                
                rub, rib = qub.clone(), qib.clone()
                pbar =tqdm(range(self.max_iter), desc=f'CascadingRank {beh} Iteration', leave=False)
                
                # Power Iteration
                for k in pbar:
                    rub_cur = (1-self.alpha-self.beta) * (self.A[j] @ rib) + self.alpha * qub + self.beta * rub_casc
                    rib_cur = (1-self.alpha-self.beta) * (self.A[j].T @ rub) + self.alpha * qib + self.beta * rib_casc
                
                    residual = (torch.absolute(rub_cur - rub).sum() + torch.absolute(rib_cur - rib).sum()) / batch_size
                    pbar.set_postfix({'residual': residual.item()})
                    
                    if residual <= self.tolerance or (k+1) == self.max_iter:
                        if beh == 'buy':
                            ranking_scores.append(rib_cur.cpu())
                        else:
                            rub_casc = rub_cur
                            rib_casc = rib_cur
                        converged_iterations_batch.append(k+1)
                        break
                        
                    rub = rub_cur
                    rib = rib_cur
                converged_iterations_behs.append(converged_iterations_batch)
                
        ranking_scores = torch.concat(ranking_scores, dim=1).T # |U| X |I|
        converged_iterations = np.mean(converged_iterations_behs, axis=0).round().astype(int)
        return ranking_scores, converged_iterations

            