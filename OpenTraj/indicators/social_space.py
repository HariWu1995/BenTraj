# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances

from ..utils.histogram_sampler import histogram_sampler


def frame_distance(group):
    # get all positions
    X_t = group[["pos_x", "pos_y"]]
    N_t = len(X_t)  # number of agents

    if N_t > 1:
        # compute distance matrix between all pairs of agents
        DD_t = euclidean_distances(X_t)

        # remove the diagonal elements (or self-distances)
        DD_t = DD_t[~np.eye(N_t, dtype=bool)].reshape(N_t, N_t - 1)

        # calculate min distance for each agent
        minD_t = np.min(DD_t, axis=1)
        group["min_dist"] = minD_t
    else:
        group["min_dist"] = 1000

    return group


def social_space(dataset: pd.DataFrame):
    """
    The minimum distance that each of agent see during their lifetime
    """
    dataset = dataset.reset_index(drop=True)
    new_data = dataset.groupby(["scene_id", "frame_id"]).apply(frame_distance)

    trajlet_length = 4.8
    trajlet_overlap = 2.
    traj_lenoverlap = trajlet_length - trajlet_overlap

    threshold = 8
    eps = 1E-2

    sspace = []

    def trajlet_min_social_space(group):
        n_frames = len(group)
        if n_frames < 2: 
            return

        md = group["min_dist"].to_numpy()
        ts = group["timestamp"].to_numpy()
        dt = ts[1] - ts[0]

        f_per_traj = int(np.ceil((trajlet_length - eps) / dt))
        f_step     = int(np.ceil((traj_lenoverlap - eps) / dt))

        for start_f in range(0, n_frames - f_per_traj, f_step):
            sspace.append(min(md[start_f:start_f + f_per_traj]))

        return group

    new_data.reset_index(drop=True).groupby(["scene_id", "agent_id"])\
            .apply(trajlet_min_social_space)

    sspace = np.array(sspace)
    return sspace[sspace < threshold]


def run(datasets, output_dir, num_samples: int = 500, num_bins: int = 50):

    dataset_names = []
    soc_space_values = []
    for ds_name, ds in datasets.items():
        dataset_names.append(ds_name)
        soc_space_values.append(social_space(ds.data))

    print("\nSampling ...")
    soc_space_values = histogram_sampler(soc_space_values, max_n_samples=num_samples, n_bins=num_bins)

    df_social_space = pd.concat([pd.DataFrame({'title':    dataset_names[ii],
                                        'social_space': soc_space_values[ii]}) 
                                                                     for ii in range(len(dataset_names))])

    print("\nVisualizing ...")
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(12, 5))

    fig.add_subplot(111)
    sns.swarmplot(y='social_space', x='title', data=df_social_space, size=1)

    plt.xlabel('')
    # plt.xticks([])
    plt.xticks(rotation=-60)
    plt.savefig(os.path.join(output_dir, 'sspaces.png'), dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    import os
    from ..loaders.loader_all import get_datasets

    opentraj_root = "./datasets"
    output_dir = "./cache/benchmark/social_space"
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)

    datasets = get_datasets(opentraj_root)
    run(datasets, output_dir, num_samples=1000, num_bins=100)

