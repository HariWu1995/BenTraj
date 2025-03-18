# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import sys
import json

from .loader_eth import load_eth
from .loader_ucy import load_ucy
from .loader_gcs import load_gcs


def load_meta(opentraj_root, metafile, **kwargs):

    with open(metafile) as json_file:
        data = json.load(json_file)

    loader_name = data['loader']

    if loader_name == "loader_eth":
        traj_dataset = load_eth(os.path.join(opentraj_root, data['data_path']))

    elif loader_name == "loader_ucy":
        traj_dataset = load_ucy(os.path.join(opentraj_root, data['data_path']), **kwargs)

    elif loader_name == "loader_gc":
        traj_dataset = load_gcs(os.path.join(opentraj_root, data['data_path']))

    else:
        traj_dataset = None

    return traj_dataset


if __name__ == "__main__":
    opentraj_root = "./datasets"
    meta_filepath = os.path.join(opentraj_root, "ETH/ETH-ETH.json")

    traj_dataset = load_meta(opentraj_root, meta_filepath)
    if traj_dataset:
        print(traj_dataset.get_agent_ids())
