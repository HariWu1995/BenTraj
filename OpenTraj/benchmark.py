import os
import sys

# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, os.path.abspath(os.path.join(dir_path, '../..')))

from .loaders.loader_all import get_datasets, get_trajlets

from .indicators import general_stats
from .indicators import crowd_density
from .indicators import path_efficiency
from .indicators import collision_energy
from .indicators import motion_properties
from .indicators import traj_deviation
from .indicators import traj_entropy
from .indicators import global_multimodality


if __name__ == "__main__":

    output_dir = "./cache/benchmark"
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)

    opentraj_root = "./datasets"
    all_datasets = get_datasets(opentraj_root)  # map from dataset_name: str -> `TrajDataset` object
    all_trajlets = get_trajlets(opentraj_root)  # map from dataset_name: str -> Trajlets (np array)

    sampling_kwargs = dict(num_samples=250, num_bins=50)

    general_stats.run(all_datasets, f"{output_dir}/stats")
    crowd_density.run(all_datasets, f"{output_dir}/crowd_density", **sampling_kwargs)
    collision_energy.run(all_datasets, f"{output_dir}/collision_energy", **sampling_kwargs)

    global_multimodality.run(all_trajlets, f"{output_dir}/multi_modality", opentraj_root, run_parallel=True)
    motion_properties.run(all_trajlets, f"{output_dir}/motion", **sampling_kwargs)
    path_efficiency.run(all_trajlets, f"{output_dir}/path_efficiency", **sampling_kwargs)
    traj_deviation.run(all_trajlets, f"{output_dir}/trajectory_deviation")
    traj_entropy.run(all_trajlets, f"{output_dir}/trajectory_entropy")
