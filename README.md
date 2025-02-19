# CascadingRank
This is the official code for **CascadingRank** (Personalized Ranking on Cascading Behavior Graphs for
Accurate Multi-Behavior Recommendation), submitted to Information Sciences 2025.  
You can find the ArXiv version of the paper here: [https://arxiv.org/abs/2502.11335](https://arxiv.org/abs/2502.11335).
## Prerequisties
You can install the required packages with a conda environment by typing the following command in your terminal:
```bash
conda create -n CascadingRank python=3.9
conda activate CascadingRank
pip install -r requirements.txt
```


## Datasets
The statistics of datasets used in CascadingRank are summarized as follows.   
| Dataset | Users  | Items  | Views       | Collects        | Carts         | Buys   |
|---------|--------:|--------:|-------------:|-----------------:|---------------:|--------:|
| Taobao  | 15,449 | 11,953 | 873,954 | -               | 195,476 | 92,180 |
| Tenrec   | 27,948 | 15,440 | 1,489,997 | 13,947  | 1,914   | 1,307|
| Tmall   | 41,738 | 11,953 | 1,813,498 | 221,514 | 1,996 | 255,586|

<!--<img src="./assets/data_statistics.png" width="500px" height="200px" title="data statistics"/>-->

We gathered Taobao dataset from [MBCGCN](https://github.com/SS-00-SS/MBCGCN), Tenrec datasets from [tenrec](https://github.com/yuangh-x/2022-NIPS-Tenrec), and Tmall datasets from [CRGCN](https://github.com/MingshiYan/CRGCN).

## Usage

### Overview
This code computes the ranking scores on items for all users, where the ranking score matrix has a dimension of |U| $\times$ |I|\, with each row representing a user's score on all items.

### Run CascadingRank
You can run our model with the best hyperparameters for HR@10 on each dataset by typing the following command in your terminal:

#### Run CascadingRank in the `Taobao` dataset
```python
python ./src/main.py --dataset taobao \
                     --alpha 0.0 \
                     --beta 0.9 \
                     --tolerance 1e-4 \
                     --batch_size 1024
```

#### Run CascadingRank in the `Tenrec` dataset
```python
python ./src/main.py --dataset tenrec \
                     --alpha 0.3 \
                     --beta 0.6 \
                     --tolerance 1e-4 \
                     --batch_size 1024
```

#### Run CascadingRank in the `Tmall` dataset
```python
python ./src/main.py --dataset tmall \
                     --alpha 0.7 \
                     --beta 0.2 \
                     --tolerance 1e-4 \
                     --batch_size 1024
```


## Result of CascadingRank
The test performance of CascadingRank for each dataset is as follows:
|**Dataset**|**HR@10**|**NDCG@10**|
|:-:|:-:|:-:|
|**Taobao**|0.3324|0.1626|
|**Tenrec**|0.4747|0.2723|
|**Tmall**|0.3751|0.1871|

All experiments are conducted on RTX 4090 (24GB) with cuda version 11.8

### Validated hyperparameters of CascadingRank
We provide the validated hyperparameters of CascadingRank for each dataset to ensure reproducibility.

<table border="1" cellspacing="0" cellpadding="5">
    <thead>
        <tr>
            <th rowspan="2">Metric</th>
            <th colspan="3">HR@10</th>
            <th colspan="3">NDCG@10</th>
        </tr>
        <tr>
            <th>α</th>
            <th>β</th>
            <th>γ</th>
            <th>α</th>
            <th>β</th>
            <th>γ</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Taobao</td>
            <td>0</td>
            <td>0.9</td>
            <td>0.1</td>
            <td>0</td>
            <td>0.9</td>
            <td>0.1</td>
        </tr>
        <tr>
            <td>Tenrec</td>
            <td>0.3</td>
            <td>0.6</td>
            <td>0.1</td>
            <td>0</td>
            <td>0.6</td>
            <td>0.4</td>
        </tr>
        <tr>
            <td>Tmall</td>
            <td>0.7</td>
            <td>0.2</td>
            <td>0.1</td>
            <td>0.7</td>
            <td>0.2</td>
            <td>0.1</td>
        </tr>
    </tbody>
</table>


**Description of each hyperparameter**
* $\alpha$: strength of query fitting (`--alpha`)
* $\beta$: strength of cascading alignment (`--beta`)
* $\gamma$: strength of ranking score smoothing, where $\gamma = 1-\alpha-\beta$


## Detailed Options
You can run CascadingRank with custom hyperparameters as follows:
|**Option**|**Description**|**Default**|**Options**|
|:-:|:-:|:-:|:-:|
|`dataset`|Dataset name|taobao|[taobao, tmall, tenrec]|
|`device`|Computation device|cuda|[cpu, cuda]|
|`data_dir`| Data directory path|./data||
|`alpha`|Strength of query fitting|0.0|range in [0, 1]|
|`beta`|Strength of cascading alignment|0.9|range in [0, 1]|
|`tolerance`|Convergence threshold ($\epsilon$) for residual|1e-4||
|`max_iter`|Maximum number of iterations|100||
|`batch_size`|Batch size| 1024||
|`ks`|Parameter for top-k recommendation|[10, 30, 50, 100, 200]||

Note that `alpha` + `beta` should not be greater than one.


Please cite the paper if you use this code in your own work:
```
@misc{ko2025personalizedrankingcascadingbehavior,
      title={Personalized Ranking on Cascading Behavior Graphs for Accurate Multi-Behavior Recommendation}, 
      author={Geonwoo Ko and Minseo Jeon and Jinhong Jung},
      year={2025},
      eprint={2502.11335},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2502.11335}, 
}
```
