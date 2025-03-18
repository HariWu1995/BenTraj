# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import os

import pandas as pd
import numpy as np

from ..core.trajdataset import TrajDataset


def load_hermes(path, **kwargs):
    traj_dataset = TrajDataset()

    # read from csv => fill traj table
    csv_columns = ["agent_id", "frame_id", "pos_x", "pos_y", "pos_z"]
    raw_dataset = pd.read_csv(path, sep=r"\s+", header=None, names=csv_columns)

    # convert from cm => meter
    raw_dataset["pos_x"] = raw_dataset["pos_x"] / 100.
    raw_dataset["pos_y"] = raw_dataset["pos_y"] / 100.

    traj_dataset.title = kwargs.get('title', "no_title")

    # copy columns
    columns = ["frame_id", "agent_id", "pos_x", "pos_y"]
    traj_dataset.data[columns] = raw_dataset[columns]

    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', 16)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    transform = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])
    traj_dataset.apply_transformation(transform, inplace=True)

    return traj_dataset


if __name__ == '__main__':
    dataroot = "F:/__Datasets__/OpenTraj/HERMES"
    dataset = load_hermes(f"{dataroot}/Corridor-1D/uo-050-180-180.txt")
    print(dataset.get_agent_ids())

