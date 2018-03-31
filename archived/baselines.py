"""
The following class is an implementation for T-CorEx baseline class. It selects the best value of
hyper-parameters then RE-TRAINS a tcorex using these parameters and tests on the test data.
Sometimes this retraining is unstable, and we get far validation and test performances.
"""
class TCorex(Baseline):
    def __init__(self, tcorex, **kwargs):
        self.tcorex = tcorex
        super(TCorex, self).__init__(**kwargs)

    def select(self, train_data, val_data, params):
        print("Selecting the best parameter values for {} ...".format(self.name))
        best_score = 1e18
        best_params = None

        const_params = dict()
        search_params = []
        for k, v in params.items():
            if k in ['l1', 'l2']:
                continue
            if isinstance(v, list):
                arr = [(k, x) for x in v]
                search_params.append(arr)
            else:
                const_params[k] = v

        # add L1/L2 regularization parameters
        reg_params = []
        if 'l1' in params:
            reg_params += [('l1', x) for x in params['l1']]
        if 'l2' in params:
            reg_params += [('l2', x) for x in params['l2']]
        if len(reg_params) > 0:
            search_params.append(reg_params)

        # add a dummy variable if the grid is empty
        if len(search_params) == 0:
            search_params = [[('dummy', None)]]

        grid = list(itertools.product(*search_params))

        for index, cur_params in enumerate(grid):
            print("done {} / {}".format(index, len(grid)), end='')
            print(" | running with ", end='')
            for k, v in cur_params:
                print('{}: {}\t'.format(k, v), end='')
            print('')

            cur_params = dict(cur_params)
            for k, v in const_params.items():
                cur_params[k] = v
            cur_params['nt'] = len(train_data)

            if best_params is None:
                best_params = cur_params  # just to select one valid set of parameters
            cur_score = self.evaluate(train_data, val_data, cur_params, n_iter=1, verbose=False)['mean']
            if not np.isnan(cur_score) and cur_score < best_score:
                best_score = cur_score
                best_params = cur_params
        print('\n')
        return (best_params, best_score)

    def evaluate(self, train_data, test_data, params, n_iter, verbose=True):
        if verbose:
            print("Evaluating {} for {} iterations ...".format(self.name, n_iter))
        start_time = time.time()
        scores = []
        for iteration in range(n_iter):
            c = self.tcorex(**params)
            c.fit(train_data)
            covs = c.get_covariance()
            cur_nll = metric_utils.calculate_nll_score(data=test_data, covs=covs)
            scores.append(cur_nll)
        finish_time = time.time()
        print("\tElapsed time {:.1f}s, mean score = {}".format(finish_time - start_time, np.mean(scores)))
        return self.report_scores(scores, n_iter)

    def timeit(self, train_data, params):
        c = self.tcorex(**params)
        start_time = time.time()
        c.fit(train_data)
        covs = c.get_covariance()
        finish_time = time.time()
        return finish_time - start_time
