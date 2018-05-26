# T-CorEx

Time-Varying version of [Linear CorEx](https://arxiv.org/abs/1706.03353).

## Installation
* Install miniconda 2 or miniconda 3 depending on python version.
* Install numpy, scipy, sklearn, Theano, Lasagne.


## Description

The main method is the `theano_time_corex.TCorexWeights` class. 
It takes a temporal data and learns one linear CorEx for each time step.

Here are the parameters of `TCorexWeights`:

* `nt` - number of time steps
* `nv` - number of variables of time series
* `n_hidden` - number of latent factors
* `max_iter` - how many iterations to train for each annealing step. 500 is usually enough.
* `tol` - value used in checking convergence. The default value works well.
* `anneal` - `True` of `False`. Whether to do annealing while training. The default value is `True`, which always works better than `False`.
* `missing_values` - what value to impute for missing values. It is better to do the imputation outside the method and put `None` for this parameter.
* `discourage_overlap` - The default is `True`. It always should be `True`.
* `gaussianize` - How to pre-process the data. Keep the default value.
* `gpu` - `True` or `False`. The default value is `False`.
* `y_scale`  - The default value is 1. No need to change.
* `update_iter` - Print some information after each `update_iter` iteration. Works when `verbose`=True. The default value is 10.
* `pretrained_weights` - This argument can be used to use pre-trained weights for each linear CorEx. The default value is `None`.
* `verbose` - `True` or `False`. The default value is `False`. In case of `True` the algorithm will print some information about training.
* `seed` - The seed of random number generators. The default is `None`.
* `l1` - A non-negative number specifying the weight of L1 temporal regularization.
* `l2` - A non-negative number specifying the weight of L2 temporal regularization. Use either `l1` or `l2`.
* `reg_type` - What to regularize. The default value is 'W', which works the best. It means to regularize  (W_{t+1} - W_t). 
* `init` - `True` or `False`. The default is `True`. Whether to initialize the weights with the weights of a linear CorEx learnt on whole data. Setting this `True` is almost always helpful.
* `gamma` - A positive number >= 1. This argument controls the weighs of different samples used for estimating some entities at time step t. The formula of weights is: weight_t(t') = 1 / gamma ^ (| t' - t|). If the time series are very dynamic the value of gamma should be higher.

## Usage

```python
from theano_time_corex import TCorexWeights
import numpy as np


train_data = np.array([
        [[  9.05826664e-01,   2.95392900e+00,  -2.30730664e-01,
          -1.57738814e-01,   3.53468166e-01,   2.12777208e-01],
        [  1.10947049e-01,   2.38739815e+00,  -1.89155010e+00,
          -4.76880536e+00,   1.35902630e+00,   7.70381891e-01],
        [ -3.63468472e-01,   3.28970321e-01,   2.02399240e+00,
           4.72883558e+00,  -1.71090140e+00,  -1.62805686e+00],
        [ -5.78837120e-01,   1.78409018e+00,   1.08130259e+00,
           8.39528123e-01,  -3.17508644e-01,   1.92113325e-01],
        [ -7.40604519e-01,  -1.50226271e+00,   5.38100057e-01,
           4.51820096e+00,  -1.19668930e+00,  -9.01719197e-01],
        [ -1.87416211e+00,  -3.07618874e+00,   3.70077140e+00,
          -2.38143535e+00,   3.27999128e-01,   7.98266446e-01],
        [ -3.35496629e-01,   2.88104807e+00,  -4.27584420e-01,
           3.01953735e-01,  -7.35743418e-01,  -8.18070940e-02],
        [ -3.12118913e-01,   4.47871842e-02,  -7.18409020e-01,
          -2.56706440e+00,   2.34620922e-01,   9.76063275e-01]],

       [[  7.34920201e-01,   1.56441263e+00,   3.72964275e+00,
           5.82552115e+00,  -1.31706505e+00,  -1.46299008e+00],
        [  1.09905958e+00,  -2.06443278e+00,  -4.45342962e+00,
           2.38501410e-03,  -1.54738769e+00,  -9.17159122e-01],
        [  5.71888107e-02,  -3.29898105e+00,   3.08475023e-01,
          -1.85228995e+00,   8.97107657e-01,   9.17863447e-01],
        [ -6.59398125e-01,   1.49821771e+00,   1.89703963e+00,
           1.58343419e+00,   3.74944782e-01,  -1.03138944e+00],
        [ -2.04530141e-01,   1.31673408e+00,   2.84818804e-02,
           3.88736844e-01,   1.57481463e+00,   2.59237189e-01],
        [  5.58675132e-01,  -2.12643817e+00,  -1.10454938e+00,
          -2.00813563e-01,  -2.09306048e+00,  -6.38894288e-01],
        [  1.55625578e-01,   7.66031988e-01,  -5.96081991e-01,
           4.47048523e+00,  -1.52810322e+00,   4.51219753e-01],
        [ -1.87906782e+00,  -9.18937191e-02,   6.83064367e-01,
          -2.14151202e-01,  -1.37378818e+00,   9.81216638e-03]],

       [[  2.15576970e+00,   2.63810764e+00,   2.18272764e+00,
          -2.78631365e+00,  -9.00679996e-01,   1.83278966e+00],
        [  6.65359725e-01,  -3.04761949e+00,   1.69390366e+00,
          -2.60503611e-02,   2.34909395e+00,   3.29751612e-01],
        [ -2.43639950e+00,  -3.41004910e+00,  -1.02133299e+00,
           1.93481931e+00,   1.09188368e+00,  -2.71362758e-01],
        [ -2.05364212e+00,   2.85565590e+00,   2.86266866e+00,
          -2.72481055e-01,   2.44116024e+00,   4.50517095e-01],
        [  2.11758494e+00,   6.32919929e-01,  -1.97941411e+00,
           2.47013424e+00,  -9.77017622e-01,  -6.33382266e-02],
        [ -3.15356213e-01,   7.35066378e-01,  -9.48242357e-01,
           1.30660921e+00,  -4.67149359e-01,  -7.16003983e-01],
        [ -8.46283662e-01,   5.84488807e+00,  -1.67484476e+00,
           2.57205304e+00,   1.24532682e+00,   5.75049328e-01],
        [ -6.16156846e-01,   3.84315594e+00,   1.78420301e+00,
          -1.93740110e+00,  -1.28692301e+00,   1.49258893e+00]],

       [[  3.92852999e-02,  -5.61553320e-01,   1.79961963e+00,
          -2.05658728e+00,   8.83890639e-01,   1.33450876e-02],
        [  4.78749558e-01,   6.25495790e-01,  -1.95834638e+00,
           2.16923900e+00,   1.28821894e+00,   2.19514930e+00],
        [  3.13186712e-01,   1.93569265e+00,   1.84753108e+00,
          -1.38720639e-01,   2.22944486e+00,   3.57921510e-01],
        [  6.95313071e-01,   1.34739007e+00,   8.36872070e-02,
          -3.18134243e+00,   2.97560133e-01,   3.64739410e-01],
        [  1.23907354e+00,   8.12564333e-01,   2.87281186e+00,
           7.50304338e-01,   1.91198746e-02,   1.82975390e-01],
        [ -1.70019187e+00,   4.15764083e-01,   2.56796920e+00,
           1.12249488e+00,  -9.78340985e-02,   9.24382322e-01],
        [ -9.41114136e-01,  -3.73469238e+00,   2.45143548e+00,
           9.88228862e-01,  -2.22102188e-01,   3.09796809e+00],
        [  5.99409622e-01,   1.40489129e+00,  -2.73448787e+00,
           1.77108072e+00,  -1.05145832e+00,   8.30633386e-01]],

       [[  2.49522121e+00,  -2.70625842e+00,   9.72605407e-01,
           3.30998479e-01,  -7.02397208e-01,  -9.81801570e-01],
        [  6.63185577e-01,   1.12913018e+00,  -1.64064263e+00,
           5.26992954e-02,  -4.13849841e+00,  -7.11024115e-02],
        [ -2.41691811e-01,  -2.39169497e+00,  -2.01307575e-01,
          -7.73492015e-01,   1.26749372e+00,  -7.53838460e-01],
        [  1.13224898e+00,   3.14900703e+00,  -6.27163058e+00,
          -8.89032094e-01,   1.31018035e+00,   1.44370543e-01],
        [ -2.16399048e+00,  -1.13481828e+00,   1.47982402e+00,
          -1.00613826e+00,  -7.66382970e-01,  -3.22793232e-01],
        [ -2.34366454e+00,   8.49011486e-01,   3.04753553e+00,
          -1.05161868e-01,  -1.54544730e+00,   7.62017336e-01],
        [  3.40174030e+00,   3.23553701e+00,  -4.83591538e+00,
          -1.35551035e-01,   2.41374468e+00,  -8.08084505e-01],
        [ -3.51947607e+00,   7.29992994e-01,  -3.07478604e+00,
          -4.40154631e-01,   1.59697173e+00,  -9.56570512e-01]],

       [[  2.83354163e-01,  -3.21678246e-01,  -1.52181814e+00,
           2.32525564e+00,  -2.68334038e+00,   1.11234636e+00],
        [ -2.22607894e+00,  -3.61316411e+00,   1.59997747e+00,
           1.66642654e+00,  -3.60939602e+00,   1.12570885e+00],
        [ -1.80326906e-01,  -4.80360874e-01,   3.35558771e-01,
           8.72967449e-01,  -1.36546307e+00,   6.92690740e-01],
        [ -1.82393538e+00,  -4.57797685e+00,   1.47855884e+00,
           2.56880327e+00,  -2.47265010e+00,   1.34471047e+00],
        [ -7.11085189e-01,  -4.07372696e-01,   4.74030920e-01,
           9.42203962e-01,  -1.38038484e+00,   7.89081650e-01],
        [  1.14598451e-01,  -1.37750074e+00,   2.89016367e-01,
           5.80483508e-01,  -1.82374374e+00,   6.34900756e-01],
        [  1.13493562e+00,   1.35654393e+00,  -1.90536590e+00,
           3.00486107e-01,  -1.60588542e+00,   3.21997925e-01],
        [ -1.67177241e+00,  -3.01544704e+00,   1.86490872e+00,
           7.05662478e-01,  -1.38603490e+00,   8.07605816e-01]]])

print("train_data shape =", train_data.shape)
nt = train_data.shape[0]
nv = train_data.shape[2]

corex = TCorexWeights(nt=nt, nv=nv, n_hidden=2, max_iter=500, l1=0.01, reg_type='W', init=True, gamma=1.5)
corex.fit(train_data)

covs = corex.get_covariance()

print(covs[0])
```

Output:
```text
Install CUDA and cudamat (for python) to enable GPU speedups.
Install CUDA and cudamat (for python) to enable GPU speedups.
Install CUDA and cudamat (for python) to enable GPU speedups.
train_data shape = (6, 8, 6)
Annealing iteration finished, time = 0.4757235050201416
Annealing iteration finished, time = 0.47513866424560547
Annealing iteration finished, time = 0.4742848873138428
Annealing iteration finished, time = 0.47440338134765625
Annealing iteration finished, time = 0.4816720485687256
Annealing iteration finished, time = 0.47649645805358887
Annealing iteration finished, time = 0.4767448902130127
[[ 1.36984698e+00  9.56758135e-01 -7.20350685e-01 -4.58781010e-02
   2.75691751e-02 -1.22990780e-02]
 [ 9.56758135e-01  5.07061427e+00 -1.88246994e+00 -2.38230437e-01
   1.07618233e-01 -4.34434063e-04]
 [-7.20350685e-01 -1.88246994e+00  4.16697957e+00  5.26021399e-01
  -1.85729860e-01 -9.16622574e-02]
 [-4.58781010e-02 -2.38230437e-01  5.26021399e-01  6.29238555e+00
  -1.75550873e+00 -1.53778619e+00]
 [ 2.75691751e-02  1.07618233e-01 -1.85729860e-01 -1.75550873e+00
   1.77046106e+00  4.64011470e-01]
 [-1.22990780e-02 -4.34434063e-04 -9.16622574e-02 -1.53778619e+00
   4.64011470e-01  8.44947222e-01]]
```
