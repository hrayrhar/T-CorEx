from tcorex.experiments.data import generate_nglf, make_buckets
from tcorex.experiments import baselines
from tcorex import base
from tcorex import TCorex


def main():
    nv = 32  # number of observed variables
    m = 4  # number of hidden variables
    nt = 10  # number of time periods
    train_cnt = 8  # number of training samples for each window
    val_cnt = 4  # number of validation samples for each window

    # generate some data
    data, sigma = generate_nglf(nv=nv, m=m, ns=nt * (train_cnt + val_cnt))

    # divide it into buckets
    bucketed_data, index_to_bucket = make_buckets(data, window=train_cnt + val_cnt, stride='full')

    # split it into train and validation
    train_data = [X[:train_cnt] for X in bucketed_data]
    val_data = [X[train_cnt:] for X in bucketed_data]

    # the core method we is the TCorex class of pytorch_tcorex
    tc = TCorex(nt=nt,
                nv=nv,
                n_hidden=m,
                max_iter=500,
                device='cpu',  # for GPU set 'cuda',
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
        'device': 'cpu',
        'l1': [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        'gamma': [1e-6, 0.3, 0.6, 0.8, 0.9],
    })

    best_score, best_params, best_covs, best_method, all_results = baseline.select(train_data, val_data, grid)
    tc = best_method  # this is the model that performed the best on the validation data, you can use it as above
    base.save(tc, 'best_method.pkl')


if __name__ == '__main__':
    main()
