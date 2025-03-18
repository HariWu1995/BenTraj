# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import yaml
from glob import glob

import numpy as np
import pandas as pd

from ..core.trajlet import split_trajectories
from ..core.trajdataset import TrajDataset, merge_datasets

from .loader_eth import load_eth
from .loader_ucy import load_ucy
from .loader_gcs import load_gcs
from .loader_ind import load_ind
from .loader_sdd import load_sdd, load_sdd_dir
from .loader_lcas import load_lcas
from .loader_pets import load_pets
from .loader_town import load_town
from .loader_kitti import load_kitti
from .loader_hermes import load_hermes
from .loader_trajnet import load_trajnet
from .loader_edinburgh import load_edinburgh
from .loader_wildtrack import load_wildtrack


all_dataset_names = [
    'ETH-Univ',
    'ETH-Hotel',

    'UCY-Zara',
    # 'UCY-Zara1',
    # 'UCY-Zara2',

    'UCY-Univ',
    # 'UCY-Univ3',

    # 'PETS-S2l1',

    'SDD-coupa',
    'SDD-bookstore',
    'SDD-deathCircle',
    # 'SDD-gates',
    # 'SDD-hyang',
    # 'SDD-little',
    # 'SDD-nexus',
    # 'SDD-quad',

    # 'GC',

    # 'InD-1',  # location_id = 1
    # 'InD-2',  # location_id = 2
    # 'InD-3',  # location_id = 3
    # 'InD-4',  # location_id = 4

    # 'LCas-Minerva',
    'KITTI',

    # 'Edinburgh',
    # 'Edinburgh-01Jul',
    # 'Edinburgh-01Aug',
    # 'Edinburgh-01Sep',

    # Bottleneck (Hermes)
    'BN-1d-w180',
    'BN-2d-w160',

    # 'TownCenter',
    'WildTrack',
]


def get_trajlets(opentraj_root, dataset_names=all_dataset_names, to_numpy=True):
    trajlets = {}

    # Make a temp dir to store and load trajlets (no splitting anymore)
    # trajlet_dir = os.path.join(opentraj_root, 'trajlets__temp_')
    trajlet_dir = './cache/trajlets'
    if not os.path.exists(trajlet_dir): 
        os.makedirs(trajlet_dir)

    for dataset_name in dataset_names:
        trajlet_npy_file = os.path.join(trajlet_dir, dataset_name + '-trl.npy')
        if to_numpy and os.path.exists(trajlet_npy_file):
            trajlets[dataset_name] = np.load(trajlet_npy_file)
            print("loading trajlets from:", trajlet_npy_file)
        else:
            ds = get_datasets(opentraj_root, [dataset_name])[dataset_name]
            trajs = ds.get_trajectories(label="pedestrian")
            trajlets[dataset_name] = split_trajectories(trajs, to_numpy=to_numpy)
            if to_numpy:
                np.save(trajlet_npy_file, trajlets[dataset_name])
            print("writing trajlets ndarray into:", trajlet_npy_file)

    return trajlets


def get_dataset(opentraj_root, dataset_name, use_kalman='default'):

    # ========== ETH ==============
    if 'eth-univ' == dataset_name.lower():
        eth_univ_root = f'{opentraj_root}/ETH/seq_eth/obsmat.txt'
        dataset = load_eth(eth_univ_root, title=dataset_name, scene_id='Univ', use_kalman=True)

    elif 'eth-hotel' == dataset_name.lower():
        eth_hotel_root = f'{opentraj_root}/ETH/seq_hotel/obsmat.txt'
        dataset = load_eth(eth_hotel_root, title=dataset_name, scene_id='Hotel')
    # ******************************

    # ========== UCY ==============
    elif 'ucy-zara' == dataset_name.lower():  
        # all 3 zara sequences
        zara01_dir = f'{opentraj_root}/UCY/zara01'
        zara02_dir = f'{opentraj_root}/UCY/zara02'
        zara03_dir = f'{opentraj_root}/UCY/zara03'
        zara_01_ds = load_ucy(f'{zara01_dir}/annotation.vsp', homog_file=f'{zara01_dir}/H.txt', scene_id='1', use_kalman=True)
        zara_02_ds = load_ucy(f'{zara02_dir}/annotation.vsp', homog_file=f'{zara02_dir}/H.txt', scene_id='2', use_kalman=True)
        zara_03_ds = load_ucy(f'{zara03_dir}/annotation.vsp', homog_file=f'{zara03_dir}/H.txt', scene_id='3', use_kalman=True)
        dataset = merge_datasets([zara_01_ds, zara_02_ds, zara_03_ds], dataset_name)

    elif 'ucy-univ' == dataset_name.lower():  
        # all 3 sequences
        st001_dir = f'{opentraj_root}/UCY/students01'
        st003_dir = f'{opentraj_root}/UCY/students03'
        uni_ex_dir = f'{opentraj_root}/UCY/uni_examples'
        st001_ds  =  load_ucy(f'{st001_dir}/annotation.vsp', homog_file=f'{st003_dir}/H.txt', scene_id='st001', use_kalman=True)
        st003_ds  =  load_ucy(f'{st003_dir}/annotation.vsp', homog_file=f'{st003_dir}/H.txt', scene_id='st003', use_kalman=True)
        uni_ex_ds = load_ucy(f'{uni_ex_dir}/annotation.vsp', homog_file=f'{st003_dir}/H.txt', scene_id='uni-ex', use_kalman=True)
        dataset = merge_datasets([st001_ds, st003_ds, uni_ex_ds], dataset_name)

    elif 'ucy-zara1' == dataset_name.lower():
        zara01_root = f'{opentraj_root}/UCY/zara01/obsmat.txt'
        dataset = load_eth(zara01_root, title=dataset_name)

    elif 'ucy-zara2' == dataset_name.lower():
        zara02_root = f'{opentraj_root}/UCY/zara02/obsmat.txt'
        dataset = load_eth(zara02_root, title=dataset_name)

    elif 'ucy-univ3' == dataset_name.lower():
        st003_root = f'{opentraj_root}/UCY/students03/obsmat.txt'
        dataset = load_eth(st003_root, title=dataset_name)
    # ******************************

    # ========== HERMES ==============
    elif 'bn' in dataset_name.lower().split('-'):
        [_, exp_flow, cor_size] = dataset_name.split('-')
        if exp_flow == '1d' and cor_size == 'w180':   
            # 'Bottleneck-udf-180'
            bottleneck_path = f'{opentraj_root}/HERMES/Corridor-1D/uo-180-180-120.txt'
        elif exp_flow == '2d' and cor_size == 'w160':  
            # 'Bottleneck-bdf-160'
            bottleneck_path = f"{opentraj_root}/HERMES/Corridor-2D/bo-360-160-160.txt"
        else:
            "Unknown Bottleneck dataset!"
            return None
        dataset = load_hermes(bottleneck_path, title=dataset_name, 
                                        sampling_rate=6, use_kalman=True)
    # ******************************

    # ========== PETS ==============
    elif 'pets-s2l1' == dataset_name.lower():
        pets_root = f'{opentraj_root}/PETS-2009/data'
        dataset = load_pets(path=f'{pets_root}/annotations/PETS2009-S2L1.xml',  # Pat:was PETS2009-S2L2
                      calib_path=f'{pets_root}/calibration/View_001.xml',
                           title=dataset_name, sampling_rate=2)
    # ******************************

    # ========== GC ==============
    elif 'gc' == dataset_name.lower():
        gc_root = f'{opentraj_root}/GC/Annotation'
        dataset = load_gcs(gc_root, title=dataset_name, world_coord=True, use_kalman=True)
    # ******************************

    # ========== InD ==============
    elif dataset_name.lower().startswith('ind'):
        data_ind_loc = int(dataset_name.lower()[-1])
        if data_ind_loc == 4:
            file_ids = range(0, 6 + 1)
        elif data_ind_loc == 1:
            file_ids = range(7, 17 + 1)
        elif data_ind_loc == 2:
            file_ids = range(18, 29 + 1)
        elif data_ind_loc == 3:
            file_ids = range(30, 32 + 1)
        
        ind_root = f'{opentraj_root}/InD/inD-dataset-v1.0/data'
        ind_datasets = []
        for id in file_ids:
            fn = '%02d_tracks.csv' % id
            si = '1-%02d' % id
            dataset_i = load_ind(f"{ind_root}/{fn}", scene_id=si, sampling_rate=10, use_kalman=True)
            ind_datasets.append(dataset_i)
        dataset = merge_datasets(ind_datasets, new_title=dataset_name)
    # ******************************

    # ========== KITTI ==============
    elif 'kitti' == dataset_name.lower():
        kitti_root = f'{opentraj_root}/KITTI/data'
        dataset = load_kitti(kitti_root, 
                            title=dataset_name,
                            sampling_rate=4, # FIXME: original_fps = 2.5 
                            use_kalman=True)
    # ******************************

    # ========== L-CAS ==============
    elif 'lcas-minerva' == dataset_name.lower():
        lcas_root = f'{opentraj_root}/L-CAS/data'
        dataset = load_lcas(lcas_root, 
                            title=dataset_name,
                            sampling_rate=1, # FIXME: original_fps = 2.5 
                            use_kalman=True)
    # ******************************

    # ========== Wild-Track ==============
    elif 'wildtrack' == dataset_name.lower():
        wildtrack_root = f'{opentraj_root}/Wildtrack/annotations_positions'
        dataset = load_wildtrack(wildtrack_root, 
                                title=dataset_name,
                                sampling_rate=1, # FIXME: original_fps = 2
                                use_kalman=True)
    # ******************************

    # ========== Edinburgh ==============
    elif 'edinburgh' in dataset_name.lower():
        edinburgh_dir = f'{opentraj_root}/Edinburgh/annotations'

        if 'edinburgh' == dataset_name.lower():
            # edinburgh_path = edinburgh_dir
            # select 1-10 Sep
            selected_days = ['01Sep', '02Sep', '04Sep', '05Sep', '06Sep', '10Sep']
            partial_ds = []
            for day in selected_days:
                edinburgh_path = os.path.join(edinburgh_dir, 'tracks.%s.txt' % day)
                D = load_edinburgh(edinburgh_path, 
                                    title=dataset_name, 
                                    scene_id=day,
                                    sampling_rate=4, # FIXME: original_fps = 9
                                    use_kalman=True)
                partial_ds.append(D)
            dataset = merge_datasets(partial_ds)

        else:
            seq_date = dataset_name.split('-')[1]
            edinburgh_path = os.path.join(edinburgh_dir, 'tracks.%s.txt' % seq_date)
            dataset = load_edinburgh(edinburgh_path, 
                                    title=dataset_name,
                                    sampling_rate=4, # FIXME: original_fps = 9
                                    use_kalman=True)
    # ******************************

    # ========== Town-Center ==============
    elif 'towncenter' == dataset_name.lower():
        towncenter_root = f'{opentraj_root}/Town-Center'
        dataset = load_town(path=f'{towncenter_root}/TownCentre-groundtruth-top.txt',
                      calib_path=f'{towncenter_root}/TownCentre-calibration-ci.txt',
                    sampling_rate=10,       # FIXME: original_fps = 25
                        use_kalman=True,    # FIXME: might need Kalman Smoother
                            title=dataset_name)
        # ******************************

    # ========== SDD ==============
    elif 'sdd-' in dataset_name.lower():
        scene_name = dataset_name.split('-')[1]
        sdd_root = f"{opentraj_root}/SDD"
        sdd_annot_fmt = f'{sdd_root}/{scene_name}/**/annotations.txt'
        annot_files_sdd = sorted(glob(sdd_annot_fmt, recursive=True))

        sdd_scales_yaml_file = f'{sdd_root}/estimated_scales.yaml'
        with open(sdd_scales_yaml_file, 'r') as f:
            scales_yaml_content = yaml.load(f, Loader=yaml.FullLoader)

        scene_datasets = []
        for file_name in annot_files_sdd:
            filename_parts = file_name.replace('\\','/').split('/')
            scene_name = filename_parts[-3]
            scene_video_id = filename_parts[-2]
            scale = scales_yaml_content[scene_name][scene_video_id]['scale']
            dataset_i = load_sdd(file_name,
                                scene_id=scene_name + scene_video_id.replace('video', ''),
                                scale=scale,
                     drop_lost_frames=False,
                        sampling_rate=12, # FIXME: original_fps = 30
                            use_kalman=True)
            scene_datasets.append(dataset_i)

        dataset = merge_datasets(scene_datasets, dataset_name)
    # ******************************

    else:
        dataset = None
    
    return dataset


def get_datasets(opentraj_root, dataset_names=all_dataset_names, use_kalman='default'):
    datasets = {}

    # Make a temp dir to store and load trajdatasets (no postprocess anymore)
    # trajdataset_dir = os.path.join(opentraj_root, 'trajdatasets__temp')
    trajdataset_dir = './cache/trajdatasets'
    if not os.path.exists(trajdataset_dir): 
        os.makedirs(trajdataset_dir)

    def log_error(dataset_name, e):
        print('\n', '-'*19, f"[{dataset_name}]", '\n', e, '-'*19, '\n')

    for dataset_name in dataset_names:
        print("\nLoading dataset:", dataset_name)

        dataset_h5_file = os.path.join(trajdataset_dir, dataset_name + '.h5')
        if os.path.exists(dataset_h5_file):
            datasets[dataset_name] = TrajDataset()
            datasets[dataset_name].data = pd.read_pickle(dataset_h5_file)
            datasets[dataset_name].title = dataset_name
            print("\t from pre-processed file:", dataset_h5_file)
            continue

        try:
            datasets[dataset_name] = get_dataset(opentraj_root, dataset_name, use_kalman)
        except Exception as e:
            log_error(dataset_name, e)
            continue

        if not datasets[dataset_name]:
            log_error(dataset_name, e='Fail to load')
            continue

        # save to h5 file
        datasets[dataset_name].data.to_pickle(dataset_h5_file)
        print("\tSaved into cache folder:", dataset_h5_file)

    return datasets
