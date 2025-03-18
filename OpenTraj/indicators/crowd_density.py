# Author: Pat Zhang
# Email: bingqing.zhang.18@ucl.ac.uk
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os

from copy import deepcopy

import math
import numpy as np
import pandas as pd
import sympy as sp 
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from ..core.trajlet import split_trajectories
from ..core.trajdataset import TrajDataset
from ..loaders.loader_all import all_dataset_names
from ..utils.histogram_sampler import histogram_sampler_norm


def local_density(all_frames, trajlets, name):
    '''
    Local density function
        for all pedestrians at that time, find its distance to nearest-neighbor (NN)
    '''
    a = 1
    distNN = []
    dens_t = []
    new_frames = []

    for frame in all_frames:
       
        if len(frame) > 1:
            # find pairwise min distance
            distNN.append([])
            dens_t.append([])
            pair_dist = []
            dist = squareform(pdist(frame[['pos_x','pos_y']].values))

            for pi in dist:                
                pair_dist.append(np.array(pi))
                min_pi = [j for j in pi if j > 0.01]
                if len(min_pi) == 0:
                    min_dist = 0.01
                else:
                    min_dist = np.min(min_pi)
                distNN[-1].append(min_dist)

            # calculate local density for agent pj
            for pj in range(len(dist)):
                d = np.array(distNN[-1])
                p = pair_dist[pj]
                dens_t_i = 1 / (2*np.pi) * np.sum(1 / ((a*d)**2) * np.exp(-np.divide((p**2), (2 * (a*d)**2))))
                dens_t[-1].append(dens_t_i)
                frame.loc[frame.index[pj], 'p_local'] = dens_t_i

        new_frames.append(frame)

    new_frames = pd.concat(new_frames)
    new_traj = TrajDataset()
    new_traj.data = new_frames
     
    trajs = new_traj.get_trajectories(label="pedestrian")
    trajlets[name] = split_trajectories(trajs, to_numpy=False)

    # average local density for each trajlet
    avg_traj_plocal = []
    for trajlet in trajlets[name]:
        avg_traj_plocal.append(np.max(trajlet['p_local']))
               
    return avg_traj_plocal


def global_density(all_frames, area):
    '''
    Calculate global density as number of agents in the scene area at time t
    '''
    frame_density_samples = []
    new_frames = []
    for frame in all_frames:
        if len(frame) > 0:
            oneArea = area.loc[frame['scene_id'].values[0], 'area']
            frame_density_samples.append(len(frame) / oneArea)
    return frame_density_samples 


def run(datasets, output_dir, num_samples: int = 500, num_bins: int = 50, 
                            quantile_interval = [0.05, 0.95]):

    # store all the results in pandas dataframe
    all_global_density = []
    all_local_density = []

    # Get trajectories from dataset
    all_datasets = []
    for ds_name in all_dataset_names:
        try:
            dataset = datasets[ds_name]
            frames = dataset.get_frames()
        except Exception as e:
            continue

        all_datasets.append(ds_name)
        print("\nProcessing", ds_name)

        trajlets = {}

        # Calculate scene area
        df = dataset.data.reset_index(drop=True)
        scenes_maxX = df.groupby(['scene_id'])['pos_x'].max() 
        scenes_minX = df.groupby(['scene_id'])['pos_x'].min()
        scenes_maxY = df.groupby(['scene_id'])['pos_y'].max()
        scenes_minY = df.groupby(['scene_id'])['pos_y'].min()
        
        area = pd.DataFrame(data=[], columns=['scene_id','area'])
        for idx in scenes_maxX.index:
            x_range = scenes_maxX.loc[idx] - scenes_minX.loc[idx]
            y_range = scenes_maxY.loc[idx] - scenes_minY.loc[idx]
            area.loc[idx,'area'] = x_range * y_range

        # calculate and store global density
        global_dens = global_density(frames, area)
        all_global_density.append(global_dens)

        g_density = pd.DataFrame(data=np.zeros((len(global_dens), 2)),
                                columns=['ds_name','global_density'])
        g_density.iloc[:,0] = [ds_name] * len(global_dens)
        g_density.iloc[:,1] = global_dens
        g_density.to_csv(f"{output_dir}/{ds_name}_globalDens.csv", index=False)

        # calculate and store local density
        local_dens = local_density(frames, trajlets, ds_name)
        all_local_density.append(local_dens) 
        
        l_density = pd.DataFrame(data=[], columns=['ds_name','local_density'])
        l_density.iloc[:,0] = [ds_name] * len(local_dens)
        l_density.iloc[:,1] = local_dens 
        l_density.to_csv(f"{output_dir}/{ds_name}_localDens.csv", index=False)

    # down-sample each group.
    print("\nSampling ...")
    sample_kwargs = dict(max_n_samples=num_samples, n_bins=num_bins, quantile_interval=quantile_interval)
    gdens_d = histogram_sampler_norm(all_global_density, **sample_kwargs)
    ldens_d = histogram_sampler_norm( all_local_density, **sample_kwargs)

    # put samples in a DataFrame (required for seaborn plots)
    df_gdens = pd.concat([pd.DataFrame({'title': all_datasets[ii],
                               'global_density':      gdens_d[ii]}) for ii in range(len(all_datasets))])
    df_ldens = pd.concat([pd.DataFrame({'title': all_datasets[ii],
                                'local_density':      ldens_d[ii]}) for ii in range(len(all_datasets))])

    print("\nVisualizing ...")
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 5), sharex=True)

    sns.swarmplot(y='global_density', x='title', data=df_gdens, size=1, ax=ax1)
    ax1.set_ylim([0, 0.08])
    ax1.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    ax1.set_xlabel('')
    ax1.yaxis.label.set_size(8)
    ax1.yaxis.set_tick_params(labelsize=8)

    sns.swarmplot(y='local_density', x='title', data=df_ldens, size=1, ax=ax2)
    ax2.set_ylim([0, 6])
    ax2.set_yticks([0, 2.0, 4.0, 6.0])
    ax2.yaxis.label.set_size(8)
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', labelrotation=-20)
    ax2.yaxis.set_tick_params(labelsize=8)

    # plt.setp(ax1.get_xticklabels(), visible=False)
    # plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xticks(rotation=-20)

    fig.align_ylabels()
    plt.subplots_adjust(hspace=0.18, wspace=0.12)
    plt.savefig(os.path.join(output_dir, 'density.png'), dpi=500, bbox_inches='tight')
    plt.show()
        

if __name__ == "__main__":

    from ..loaders.loader_all import get_datasets

    opentraj_root = "./datasets"
    output_dir = "./cache/benchmark/crowd_density"
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)

    datasets = get_datasets(opentraj_root)
    # datasets = {k: d for k, d in datasets.items() if k.startswith('ETH')}
    run(datasets, output_dir, num_samples=250, num_bins=50)

