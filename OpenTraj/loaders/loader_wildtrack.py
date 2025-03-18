# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import json

from tqdm import tqdm
from glob import glob

import numpy as np
import pandas as pd

from ..core.trajdataset import TrajDataset


def load_wildtrack(path: str, **kwargs):
    traj_dataset = TrajDataset()

    files_list = sorted(glob(path + "/*.json"))
    raw_data = []
    for file_name in tqdm(files_list):
        frame_id = int(os.path.basename(file_name).replace('.json', ''))

        with open(file_name, 'r') as json_file:
            json_content = json_file.read()
            annots_list = json.loads(json_content)
            for annot in annots_list:
                person_id = annot["personID"]
                position_id = annot["positionID"]

                X = -3.0 + 0.025 * (position_id % 480)
                Y = -9.0 + 0.025 * (position_id / 480)
                raw_data.append([frame_id, person_id, X, Y])

    csv_columns = ["frame_id", "agent_id", "pos_x", "pos_y"]
    raw_dataset = pd.DataFrame(np.array(raw_data), columns=csv_columns)

    traj_dataset.title = kwargs.get('title', "Grand Central")

    # copy columns
    traj_columns = ["frame_id", "agent_id", "pos_x", "pos_y"]
    traj_dataset.data[traj_columns] = raw_dataset[traj_columns]

    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', 10)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)

    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)
    return traj_dataset


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    opentraj_root = "F:/__Datasets__/OpenTraj/Wildtrack"
    wildtrack_root = f'{opentraj_root}/annotations_positions'

    traj_datasets = load_wildtrack(wildtrack_root, title='WildTrack', 
                                            sampling_rate=1, # original_annot_framerate=2
                                                use_kalman=False)  
                                   
    trajs = list(traj_datasets.get_trajectories())
    for traj in trajs:
        plt.plot(traj[1]["pos_x"], traj[1]["pos_y"])
    plt.show()