# Author: Pat
# Email: bingqing.zhang.18@ucl.ac.uk

# LCAS data (2 scenes in the rawdata: minerva & strands2, 
#   but only minerva is included here 
#   b/c strands2 data has some issues with repeated time)

# checked with the data provided by TrajNet++, 
# they actually only used part of the data from minerva. 
# Here I included all data from minerva

import numpy as np
import pandas as pd

from glob import glob

from ..core.trajdataset import TrajDataset


def load_lcas(path, **kwargs):
    traj_dataset = TrajDataset()
    traj_dataset.title = kwargs.get('title', "LCAS")

    minerva_files_list = glob(path + "/minerva/**/data.csv")
    minerva_columns = ['frame_id','person_id','pos_x','pos_y','rot_z','rot_w','scene_id']
   
    # This load data from all files
    minerva_raw_dataset = []
    for file in minerva_files_list:
        data = pd.read_csv(file, sep=",", header=None, names=minerva_columns)
        minerva_raw_dataset.append(data)

    minerva_raw_dataset = pd.concat(minerva_raw_dataset)
    minerva_raw_dataset['scene_id'] = 'minerva'

    minerva_raw_dataset.rename(inplace=True, columns={'person_id': 'agent_id'})
    minerva_raw_dataset.reset_index(inplace=True, drop=True)

    columns = ["frame_id", "agent_id", "pos_x", "pos_y", "scene_id"]
    traj_dataset.data[columns] = minerva_raw_dataset[columns]

    traj_dataset.data["label"] = "pedestrian"

    # post-process
    # For LCAS, raw data do not include velocity
    fps = kwargs.get('fps', 2.5)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)

    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)
    return traj_dataset


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    opentraj_root = "F:/__Datasets__/OpenTraj/L-CAS"
    lcas_root = f'{opentraj_root}/data'

    # FIXME: original_fps = 2.5
    traj_dataset = load_lcas(lcas_root, title="L-CAS", use_kalman=False, sampling_rate=1)
    trajs = list(traj_dataset.get_trajectories())

    for traj in trajs:
        plt.plot(traj[1]["pos_x"], traj[1]["pos_y"])
    plt.title("L-CAS dataset")
    plt.show()
