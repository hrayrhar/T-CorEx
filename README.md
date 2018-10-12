# T-CorEx

Time-Varying version of [Linear CorEx](https://arxiv.org/abs/1706.03353).

## Requiremenets
* Python 2 or Python 3 (Python 2 is well tested)
* numpy, scipy, sklearn, PyTorch
* Linear CorEx
    * `pip install linearcorex`
* (Optional) Install nose for tests

## Description

The main method is the class 'pytorch_tcorex.TCorex'. It has the following parameters:


| parameter | description |
|:---------|:---|
|`nt`| number of time steps|
| `nv`| number of variables of time series|
| `n_hidden` | number of latent factors|
| `max_iter` | how many iterations to train for each annealing step. 500 is usually enough.|
| `tol`| value used in checking convergence. The default value works well.|
| `anneal`| `True` or `False`. Whether to do annealing while training. The default value is `True`, which always works better than `False`.|
| `missing_values`| what value to impute for missing values. It is better to do the imputation outside the method and put `None` for this parameter.|
| `discourage_overlap` | The default is `True`. It always should be `True`.|
| `gaussianize` | How to pre-process the data. Keep the default value.|
| `gpu` | `True` or `False`. The default value is `False`.|
| `y_scale`  | The default value is 1. No need to change.|
| `update_iter` | Print some information after each `update_iter` iteration. Works when `verbose`=True. The default value is 10.|
| `pretrained_weights` | This argument can be used to use pre-trained weights for each linear CorEx. The default value is `None`.|
| `verbose` | `True` or `False`. The default value is `False`. In case of `True` the algorithm will print some information about training.|
| `seed` | The seed of random number generators. The default is `None`.|
| `l1` | A non-negative number specifying the weight of L1 temporal regularization.|
| `l2` | A non-negative number specifying the weight of L2 temporal regularization. Use either `l1` or `l2`.|
| `reg_type` | What to regularize. The default value is 'W', which works the best. It means to regularize  (W_{t+1} - W_t). |
| `init` | `True` or `False`. The default is `True`. Whether to initialize the weights with the weights of a linear CorEx learnt on whole data. Setting this `True` is almost always helpful.|
| `gamma` | Real number in [0,1]. This argument controls the weighs of different samples used for estimating some entities at time step t. The formula of weights is: weight_t(t') = gamma ^ (\| t' - t\|). If the time series are very dynamic the value of gamma should be higher.|
| `weighted_obj` | `True` or `False`. Whether the objective for time step t uses the samples of other timesteps. The default is `False`.|


## Usage

Run the following command for a sample run of TCorex. 
```bash
python -m sample_run
```

The code is shown below:
``` python 
from experiments import generate_data
from experiments import baselines
from experiments import utils
from pytorch_tcorex import TCorex


def main():
    nv = 32  # number of observed variables
    m = 4  # number of hidden variables
    nt = 10  # number of time periods
    train_cnt = 8  # number of training samples for each window
    val_cnt = 4  # number of validation samples for each window

    # generate some data
    data, sigma = generate_data.generate_nglf(nv=nv, m=m, ns=nt * (train_cnt + val_cnt))

    # divide it into buckets
    bucketed_data, index_to_bucket = utils.make_buckets(data, window=train_cnt + val_cnt, stride='full')

    # split it into train and validation
    train_data = [X[:train_cnt] for X in bucketed_data]
    val_data = [X[train_cnt:] for X in bucketed_data]

    # the core method we is the TCorex class of pytorch_tcorex
    tc = TCorex(nt=nt,
                nv=nv,
                n_hidden=m,
                max_iter=500,
                torch_device='cpu',  # for GPU set 'cuda',
                l1=0.001,  # coefficient of temporal regularization term
                gamma=0.8,  # parameter that controls sample weights
                verbose=1,  # 0, 1, 2
                )

    tc.fit(train_data)
    covs = tc.get_covariance()  # returns a list of cov. matrices
    clusters = tc.clusters()  # returns clustering of variables for each time step

    # if you want the method to select its hyperparameters using grid search
    # NOTE: this can take time !
    baseline, grid = (baselines.TCorex(tcorex=TCorex, name='T-Corex'), {
        'nv': nv,
        'n_hidden': m,
        'max_iter': 500,
        'torch_device': 'cpu',
        'l1': [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        'gamma': [1e-6, 0.3, 0.6, 0.8, 0.9],
    })

    best_score, best_params, best_covs, best_method, all_results = baseline.select(train_data, val_data, grid)
    tc = best_method  # this is the model that performed the best on the validation data, you can use it as above


if __name__ == '__main__':
    main()

```