# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from ..core.trajdataset import TrajDataset
from ..core.trajlet import split_trajectories


def num_scenes(dataset: TrajDataset):
    n_scenes = dataset.data.groupby(["scene_id"]).count()
    return n_scenes


def dataset_duration(dataset: TrajDataset):
    timestamps = dataset.data.groupby(["scene_id"])["timestamp"]
    dur = sum(timestamps.max() - timestamps.min())
    return dur


def num_pedestrians(dataset: TrajDataset):
    return len(dataset.data[dataset.data["label"] == "pedestrian"].groupby(["scene_id", "agent_id"]))


def total_trajectory_duration(dataset: TrajDataset):
    timestamps = dataset.data[dataset.data["label"] == "pedestrian"].groupby(["scene_id", "agent_id"])["timestamp"]
    dur = sum(timestamps.max() - timestamps.min())
    return dur


def num_trajlets(dataset: TrajDataset, length=4.8, overlap=2):
    trajs = dataset.get_trajectories(label="pedestrian")
    trajlets = split_trajectories(trajs, length, overlap, static_filter_thresh=0.)
    non_static_trajlets = split_trajectories(trajs, length, overlap, static_filter_thresh=1.)
    return len(trajlets), len(non_static_trajlets)


def pprint_stats(ds_name, n_agents, full_dur_td, trajs_dur_td, n_trajlets, n_non_static_trajlets):
    print('\n*******************')
    print('Dataset:', ds_name)
    print('___________________')
    print('Duration =', full_dur_td)
    print('Duration|Σtraj =', trajs_dur_td)
    print('# agents =', n_agents)
    print('# trajlets =', n_trajlets)
    print('% non-static =', round(n_non_static_trajlets / n_trajlets * 100, 2))
    print('*******************\n')


def run(datasets, output_dir = None):
    for ds_name, ds in datasets.items():
        ds.data = ds.data.reset_index(drop=True)
        # n_scenes = num_scenes(ds)
        n_agents = num_pedestrians(ds)
        full_dur = dataset_duration(ds)
        trajs_dur = total_trajectory_duration(ds)
        n_trajlets, n_non_static_trajlets = num_trajlets(ds)

        full_dur_td  = timedelta(0, int(round( full_dur)), 0)
        trajs_dur_td = timedelta(0, int(round(trajs_dur)), 0)

        pprint_stats(ds_name, n_agents, full_dur_td, trajs_dur_td, n_trajlets, n_non_static_trajlets)


if __name__ == "__main__":

    from ..loaders.loader_all import get_datasets

    opentraj_root = "./datasets"

    datasets = get_datasets(opentraj_root)
    run(datasets)
