# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
import pandas as pd

from ..core.trajdataset import TrajDataset


DATA_FIELDS = ["frame_id", "agent_id", "pos_x", "pos_z", "pos_y", "vel_x", "vel_z", "vel_y"]
FPS = 2.5


def load_eth(path, **kwargs):

    traj_dataset = TrajDataset()
    traj_dataset.title = kwargs.get('title', "no_title")

    # read from csv => fill traj table
    raw_dataset = pd.read_csv(path, sep=r"\s+", header=None, names=DATA_FIELDS)

    # copy columns
    columns = ["frame_id", "agent_id", "pos_x", "pos_y", "vel_x", "vel_y"]
    traj_dataset.data[columns] = raw_dataset[columns]

    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', -1)
    if fps < 0:
        d_frame = np.diff(pd.unique(raw_dataset["frame_id"]))
        fps = d_frame[0] * FPS

    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)

    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)
    return traj_dataset


if __name__ == '__main__':
    dataroot = "F:/__Datasets__/OpenTraj"
    dataset = load_eth(f"{dataroot}/ETH/seq_eth/obsmat.txt")
    print(dataset.get_agent_ids())

