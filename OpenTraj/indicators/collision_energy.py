# Author: Pat Zhang
# Email: bingqing.zhang.18@ucl.ac.uk

# calculate DCA, TTCA for each agent at time t
# find min ttc, dca, energy for each agent with respect to all other agent at time t
# then take min value in this trajlet
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
from copy import deepcopy

import math
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from ..core.trajlet import split_trajectories
from ..core.trajdataset import TrajDataset
from ..loaders.loader_all import all_dataset_names
from ..utils.histogram_sampler import histogram_sampler_norm


eps = 1E-6


def DCA_MTX(x_4d):
    """
    :param x_4d = 4d sample (position|velocity)
    """
    N = len(x_4d)

    tiled_x = np.tile(x_4d, (N, 1, 1))
    diff = tiled_x - tiled_x.transpose((1, 0, 2))

    D_4d = diff
    Dp = D_4d[:,:,:2]
    Dv = D_4d[:,:,2:]

    DOT_Dp_Dv = np.multiply(Dp[:,:,0], Dv[:,:,0]) + np.multiply(Dp[:,:,1], Dv[:,:,1])
    Dv_sq     = np.multiply(Dv[:,:,0], Dv[:,:,0]) + np.multiply(Dv[:,:,1], Dv[:,:,1]) + eps

    TTCA = np.array(-np.divide(DOT_Dp_Dv, Dv_sq))
    TTCA[TTCA < 0] = 0
    
    DCA = np.stack([Dp[:,:,0] + TTCA * Dv[:,:,0],
                    Dp[:,:,1] + TTCA * Dv[:,:,1]], axis=2)
    DCA = np.linalg.norm(DCA, axis=2)
    
    # tri_TTCA = TTCA[np.triu_indices(TTCA.shape[0], 1)]
    # tri_DCA  =  DCA[np.triu_indices( DCA.shape[0], 1)]
    
    return DCA, TTCA


def ttc(all_frames, name, trajlets, Rp: float = 0.33):
    '''
    Rp: pedestrians radius is 0.33
    '''
    all_ttc = []
    new_frames = []
    for frame in all_frames:
        frame = frame.reset_index(drop=True)
        # if there is only one pedestrian at that time, or encounter invalid vel value
        if len(frame.index) < 2 \
            or frame['vel_x'].isnull().values.any() \
            or frame['vel_y'].isnull().values.any():
            continue

        # calculate ttc for each pair
        x_4d = np.stack((frame.pos_x.values, frame.pos_y.values, 
                         frame.vel_x.values, frame.vel_y.values), axis=1)
        DCA,TTCA = DCA_MTX(x_4d)

        for i in range(len(TTCA)):
            # find out ttc of one agent
            ttc = [TTCA[i][j] for j in range(len(TTCA[i])) if DCA[i][j] < (2*Rp) and TTCA[i][j] > 0] 
            
            # find out min ttc for one agent
            if len(ttc)>0:
                min_ttc = np.min(ttc)
                frame.loc[i,'ttc'] = min_ttc
            
            min_dca = np.min([j for j in DCA[i] if j > 0])
            frame.loc[i,'dca'] = min_dca
     
        new_frames.append(frame)

    new_frames = pd.concat(new_frames)
    new_traj = TrajDataset()
    new_traj.data = new_frames
    trajs = new_traj.get_trajectories(label="pedestrian")
    trajlets[name] = split_trajectories(trajs, to_numpy=False)

    # average local density o each trajlet
    avg_traj_ttc = []
    avg_traj_dca = []
    for trajlet in trajlets[name]: 
        avg_traj_ttc.append(np.min(trajlet['ttc'].dropna())) # min of min
        avg_traj_dca.append(np.min(trajlet['dca'].dropna())) # min of min

    return avg_traj_ttc, avg_traj_dca
    

def energy(avg_ttc, upperbound, lowerbound):
    ttc = [i for i in avg_ttc if i < upperbound and i > lowerbound]
    tau0 = upperbound 
    k = 1
 
    # calculate collision energy
    E = []
    for tau in ttc:
        e = (k/tau**2) * math.exp(-tau/tau0)
        E.append(e)

    E = np.array(E).astype("float")
    return E


def run(datasets, output_dir, num_samples: int = 500, num_bins: int = 50):

    # interaction range
    upperbound = 3
    lowerbound = 0.2
    threshold = 3
    
    datasets_ttc = []
    datasets_dca = []
    datasets_coE = []   # collision enery

    # Get all datasets
    all_datasets = []
    for ds_name in all_dataset_names:
        try:
            dataset = datasets[ds_name]
            frames = dataset.get_frames()
        except Exception as e:
            continue

        print("\nProcessing", ds_name)

        # calculate and store ttc
        all_trajs = dataset.get_trajectories("pedestrian")
        trajlets = {}

        # calculate and store ttc and dca
        avg_ttc, avg_dca = ttc(frames, ds_name, trajlets)

        allttc_df = pd.DataFrame(data=np.zeros((len(avg_ttc), 2)), columns=['name','ttc'])
        allttc_df.iloc[:,0] = [ds_name] * len(avg_ttc)
        allttc_df.iloc[:,1] = avg_ttc
        allttc_df.to_csv(f"{output_dir}/{ds_name}_ttc.csv", index=False)

        datasets_ttc.append(avg_ttc)

        alldca_df = pd.DataFrame(data=np.zeros((len(avg_dca), 2)), columns=['name','dca'])
        alldca_df.iloc[:,0] = [ds_name] * len(avg_dca)
        alldca_df.iloc[:,1] = avg_dca
        alldca_df.to_csv(f"{output_dir}/{ds_name}_dca.csv", index=False)

        datasets_dca.append(avg_dca)

        # calculate and store collision energy 
        all_E = energy(avg_ttc, upperbound, lowerbound)

        coE_df = pd.DataFrame(data=np.zeros((len(all_E), 2)), columns=['name','CoE'])
        coE_df.iloc[:,0] = [ds_name] * len(all_E)
        coE_df.iloc[:,1] = all_E 
        coE_df.to_csv(f"{output_dir}/{ds_name}_collE.csv", index=False)

        datasets_coE.append(all_E)
  
    # down-sample each group.
    print("\nSampling ...")
    sample_kwargs = dict(max_n_samples=num_samples, n_bins=num_bins)
    ttc_d = histogram_sampler_norm(datasets_ttc, **sample_kwargs)
    dca_d = histogram_sampler_norm(datasets_dca, **sample_kwargs)
    coE_d = histogram_sampler_norm(datasets_coE, **sample_kwargs)
       
    # put samples in a DataFrame (required for seaborn plots)
    df_ttc = pd.concat([pd.DataFrame({'title': all_datasets[ii],
                                        'ttc':        ttc_d[ii],}) for ii in range(len(all_datasets))])
    df_dca = pd.concat([pd.DataFrame({'title': all_datasets[ii],
                                        'dca':        dca_d[ii],}) for ii in range(len(all_datasets))])
    df_coE = pd.concat([pd.DataFrame({'title': all_datasets[ii],
                                        'CoE':        coE_d[ii],}) for ii in range(len(all_datasets))])

    print("\nVisualizing ...")
    sns.set(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 5), sharex=True)

    sns.swarmplot(y='ttc', x='title', data=df_ttc, size=1, ax=ax1)
    ax1.set_xlabel('')
    ax1.set_yticks([0, 10, 25, 50])
    ax1.yaxis.label.set_size(8)
    ax1.yaxis.set_tick_params(labelsize=8)

    sns.swarmplot(y='dca', x='title', data=df_dca, size=1, ax=ax2)
    ax2.set_xlabel('')
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    ax2.yaxis.label.set_size(8)
    ax2.yaxis.set_tick_params(labelsize=8)

    sns.swarmplot(y='CoE', x='title', data=df_coE, size=1, ax=ax3)
    ax3.set_xlabel('')
    plt.xticks(rotation=-20)
    ax3.set_yticks([0,  4,  8,  12])
    ax3.yaxis.label.set_size(8)
    ax3.yaxis.set_tick_params(labelsize=8)

    fig.align_ylabels()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(os.path.join(output_dir, 'energy.png'), dpi=500, bbox_inches='tight')
    # plt.show()
    

if __name__ == "__main__":

    from ..loaders.loader_all import get_datasets

    opentraj_root = "./datasets"
    output_dir = "./cache/benchmark/collision_energy"
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)

    datasets = get_datasets(opentraj_root)
    run(datasets, output_dir, num_samples=250, num_bins=50)
