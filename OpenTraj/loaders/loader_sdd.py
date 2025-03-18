# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import yaml

from tqdm import tqdm
from glob import glob

import pandas as pd

from ..core.trajdataset import TrajDataset


def load_sdd(path, **kwargs):
    sdd_dataset = TrajDataset()
    sdd_dataset.title = kwargs.get("title", "SDD")

    # read from csv => fill traj table
    csv_columns = ["agent_id", "x_min", "y_min", "x_max", "y_max", 
                   "frame_id", "lost", "occluded", "generated", "label"]
    raw_dataset = pd.read_csv(path, sep=" ", header=None, names=csv_columns)
    
    scale = kwargs.get("scale", 1)
    raw_dataset["pos_x"] = scale * (raw_dataset["x_min"] + raw_dataset["x_max"]) / 2
    raw_dataset["pos_y"] = scale * (raw_dataset["y_min"] + raw_dataset["y_max"]) / 2

    drop_lost_frames = kwargs.get('drop_lost_frames', False)
    if drop_lost_frames:
        raw_dataset = raw_dataset.loc[raw_dataset["lost"] != 1]

    # copy columns
    sdd_columns = ["frame_id", "agent_id",
                   "pos_x", "pos_y", # "x_min", "y_min", "x_max", "y_max",
                   "label", "lost", "occluded", "generated"]
    sdd_dataset.data[sdd_columns] = raw_dataset[sdd_columns]
    sdd_dataset.data["scene_id"] = kwargs.get("scene_id", 0)

    # calculate velocities + perform some checks
    fps = kwargs.get('sampling_rate', 30)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)

    sdd_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)
    return sdd_dataset


def load_sdd_dir(path: str, **kwargs):
    search_filter_str = "**/annotations.txt"
    if not path.endswith("/"):
        search_filter_str = "/" + search_filter_str
    files_list = sorted(glob(path + search_filter_str, recursive=True))

    scales_yaml_file = os.path.join(path, 'estimated_scales.yaml')
    with open(scales_yaml_file, 'r') as f:
        scales_yaml_content = yaml.load(f, Loader=yaml.FullLoader)

    partial_datasets = []
    for file in tqdm(files_list):
        dir_names = file.replace('\\', '/').split('/')
        scene_name = dir_names[-3]
        scene_video_id = dir_names[-2]
        scale = scales_yaml_content[scene_name][scene_video_id]['scale']

        partial_dataset = load_sdd(file, scale=scale,
                                        scene_id=scene_name+scene_video_id.replace('video', ''))
        partial_datasets.append(partial_dataset.data)

    traj_dataset = TrajDataset()
    traj_dataset.data = pd.concat(partial_datasets)

    fps = kwargs.get('sampling_rate', 30)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)

    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)
    return traj_dataset


if __name__ == '__main__':

    dataroot = "F:/__Datasets__/OpenTraj/SDD"

    # dataset = load_sdd(f"{dataroot}/bookstore/video0/annotations.txt")
    # print(dataset.get_agent_ids())

    dataset = load_sdd_dir(dataroot)
    print(dataset.get_agent_ids())

