# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..core.trajdataset import TrajDataset
from ..utils.histogram_sampler import histogram_sampler_norm


def path_length(trajectory: pd.DataFrame):
    traj_poss = trajectory[['pos_x', 'pos_y']].diff().dropna()
    distance = np.linalg.norm(traj_poss, axis=1).sum()
    return distance


def path_efficiency(trajectory: pd.DataFrame):
    """
    ratio of distance between the endpoints of a segment over actual length of the trajectory
    """
    actual_length = path_length(trajectory)
    end2end_dist = np.linalg.norm(np.diff(trajectory[['pos_x', 'pos_y']].iloc[[0, -1]], axis=0))
    return end2end_dist / actual_length


def path_efficiency_index(trajlets_np: np.ndarray):
    num = np.linalg.norm(trajlets_np[:, -1, :2] - trajlets_np[:, 0, :2], axis=1)
    denom = np.linalg.norm(np.diff(trajlets_np[:, :, :2], axis=1), axis=2).sum(axis=1)
    return num / denom


def run(trajlets, output_dir, num_samples: int = 500, num_bins: int = 50, 
                            quantile_interval = [0.05, 0.95]):

    dataset_names = list(trajlets.keys())

    print("\nCalculating ...")
    path_eff_values = []
    for ds_name, ds in trajlets.items():
        path_eff_ind = path_efficiency_index(ds) * 100
        path_eff_values.append(path_eff_ind)

    print("\nSampling ...")
    sample_kwargs = dict(max_n_samples=num_samples, n_bins=num_bins, quantile_interval=quantile_interval)
    path_eff_values = histogram_sampler_norm(path_eff_values, **sample_kwargs)
    df_path_eff = pd.concat([pd.DataFrame({'title':   dataset_names[ii],
                                        'path_eff': path_eff_values[ii]}) for ii in range(len(dataset_names))])

    print("\nVisualizing ...")
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(111)
    sns.swarmplot(y='path_eff', x='title', data=df_path_eff, size=1)
    # plt.ylim([90, 100])
    plt.xlabel('')
    # plt.xticks([])
    # ax1.set_yticks([0, 0.5, 1, 1.5, 2.])
    plt.ylabel('Path Efficiency (%)')
    plt.xticks(rotation=-20)
    ax1.yaxis.label.set_size(9)
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)

    plt.savefig(os.path.join(output_dir, 'path_efficiency.png'), dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    from ..loaders.loader_all import get_trajlets

    opentraj_root = "./datasets"
    output_dir = "./cache/benchmark/path_efficiency"
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)

    trajlets = get_trajlets(opentraj_root)
    run(trajlets, output_dir, num_samples=200)
