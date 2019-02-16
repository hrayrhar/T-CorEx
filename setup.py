from setuptools import setup

long_description = '''
T-CorEx, Temporal Correlation Explanation, is designed for covariance estimation from temporal data.
In its essence, T-CorEx trains a [linear CorEx](https://arxiv.org/abs/1706.03353) for each time period,
while employing two regularization techniques to enforce temporal consistency of estimates.
T-CorEx has linear time and memory complexity with respect to the number of observed variables and can be applied to
truly high-dimensional datasets. It takes less than an hour on a moderate PC to estimate the covariance structure
for time series with 100K variables. T-CorEx is implemented in PyTorch and can run on both CPUs and GPUs.

This package also contains a PyTorch implementation of linear CorEx and some useful tools for working
with high-dimensional low-rank plus diagonal matrices. The code is compatible with Python 2.7-3.6 and
is distributed under the GNU Affero General Public License v3.0.
'''

setup(name='T-CorEx',
      version='1.0',
      description='Temporal Correlation Explanation',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Hrayr Harutyunyan',
      author_email='harhro@gmail.com',
      url='https://github.com/Harhro94/T-CorEx',
      license='GNU Affero General Public License v3.0',
      install_requires=['numpy>=1.14.2',
                        'scipy>=1.1.0',
                        'torch>=0.4.1'],
      tests_require=['nose>=1.3.7',
                    'tqdm>=4.26'],
      classifiers=[
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Programming Language :: Python :: 3'
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 2'
          'Programming Language :: Python :: 2.7',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Affero General Public License v3'
      ],
      packages=['tcorex', 'tcorex.experiments'])
