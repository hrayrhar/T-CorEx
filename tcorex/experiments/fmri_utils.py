from __future__ import print_function
from __future__ import division

import nibabel as nib
import numpy as np


def compute_variance_of_cluster(clusters, cluster_index, coords):
    filtered = coords[clusters == cluster_index]
    return ((filtered - filtered.mean(axis=0)) ** 2).sum(axis=1).mean(axis=0)


def plot_least_varying(plt, clusters, coords, left, right):
    n_clusters = np.max(clusters) + 1
    variances = [compute_variance_of_cluster(clusters, k, coords) for k in range(n_clusters)]
    order = np.argsort(variances)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    for k in range(left, right):
        print(variances[order[k]])
        index = (clusters == order[k])
        filtered = coords[index]
        ax.scatter(filtered[:, 0], filtered[:, 1], filtered[:, 2], s=5, alpha=0.1)


def plot_most_important(plt, clusters, importance, coords, left, right, mode='absolute'):
    a = np.array(importance).copy()
    if mode == 'relative':
        a = np.array(importance).copy()
        n_clusters = np.max(clusters)
        for j in range(n_clusters):
            cnt = np.sum(clusters == j)
            a[j] /= (cnt + 1e-4)

    order = np.argsort(-a)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    for k in range(left, right):
        print(a[order[k]])
        index = (clusters == order[k])
        filtered = coords[index]
        ax.scatter(filtered[:, 0], filtered[:, 1], filtered[:, 2], s=5, alpha=0.1)


def plot_biggest(plt, clusters, coords, left, right):
    n_clusters = np.max(clusters) + 1
    cnt = [0] * n_clusters
    for j in range(n_clusters):
        cnt[j] = np.sum(clusters == j)
    order = np.argsort(-np.array(cnt))
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    for k in range(left, right):
        print(cnt[order[k]])
        index = (clusters == order[k])
        filtered = coords[index]
        ax.scatter(filtered[:, 0], filtered[:, 1], filtered[:, 2], s=5, alpha=0.1)


def plot_clusters_probabilistic(plotting, prob_clusters, coords, source_img):
    """ Plot probabilistic atlas.
    :param plotting: nilearn.plotting
    :param prob_clusters: (n_clusters, n_voxels)
    :param coords: (n_voxels, 3)
    :return:
    """
    X, Y, Z, T = source_img.shape
    a = np.zeros((X, Y, Z))
    for j in range(prob_clusters.shape[0]):
        for i in range(prob_clusters.shape[1]):
            x = int(coords[i, 0])
            y = int(coords[i, 1])
            z = int(coords[i, 2])
            a[x, y, z, j] = prob_clusters[j, i]
    atlas = nib.Nifti1Image(a, affine=source_img.affine)
    plotting.plot_prob_atlas(atlas, bg_img=False)


def plot_clusters(plotting, clusters, coords, source_img, output_file=None, figure=None):
    """ Plot probabilistic atlas.
    :param plotting: nilearn.plotting
    :param clusters: (n_voxels,)
    :param coords: (n_voxels, 3)
    :param output_file: if given the plot is saved here
    :param figure: figure param to be passed to plotting.plot_roi function
    :return:
    """
    X, Y, Z, T = source_img.shape
    a = np.zeros((X, Y, Z))
    for i in range(clusters.shape[0]):
        x = int(coords[i, 0])
        y = int(coords[i, 1])
        z = int(coords[i, 2])
        a[x, y, z] = clusters[i]
    img = nib.Nifti1Image(a, affine=source_img.affine)
    return plotting.plot_roi(img, output_file=output_file, figure=figure)
