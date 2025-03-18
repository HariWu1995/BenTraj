# Author: Juan Baldelomar
# Email: juan.baldelomar@cimat.mx

import os
import argparse
import random as rd
from tqdm import tqdm

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

from ..core.trajlet import split_trajectories


def Gauss_K(x, y, h):
    N = len(x)
    return math.exp(-np.linalg.norm(x - y) ** 2 / (2 * h ** 2)) / (2 * math.pi * h ** 2) ** N


# separates every trajectory in its observed and predicted trajlets
def separate_trajectories(trajectories, separator=8, f_per_traj=20):
    N_t = len(trajectories)
    Trajm = []
    Trajp = []
    for tr in trajectories:
        Trajm.append(tr[range(separator), :])
        Trajp.append(tr[range(separator, f_per_traj), :])
    return Trajm, Trajp


# Weights for the GMM
def weights(Xm, k, h, Tobs=8, Tpred=12):
    N_t = len(Xm)

    aux = 0
    for l in range(N_t):
        aux += Gauss_K(Xm[k], Xm[l], h)

    w = []
    for l in range(N_t):
        var = Gauss_K(Xm[k], Xm[l], h) / aux
        w.append(var)

    return np.array(w)


def get_sample(Xp, w, M, h, replace=False):
    probs = w / np.sum(w)
    if np.count_nonzero(probs) > M:
        l_sample = np.random.choice(range(len(Xp)), M, p=probs, replace=replace)
    else:
        l_sample = np.random.choice(range(len(Xp)), M, p=probs, replace=True)

    sample = []
    for i in l_sample:
        sample.append(Xp[i])

    size = len(Xp[0]) * 2
    cov = h * np.identity(size)

    for i in range(M):
        s = sample[i].reshape(size)
        s = multivariate_normal.rvs(s, cov)
        s = s.reshape(int(size / 2), 2)
        sample[i] = s

    return sample


# Entropy estimation of the kth trajectory ones weights have been gotten
def entropy(Xp, k, h, w, M, replace=False):
    N_t = len(Xp)
    samples = get_sample(Xp, w, M, h, replace=replace)

    H = 0
    for m in range(M):
        aux = 0
        for l in range(N_t):
            aux += w[l] * Gauss_K(samples[m], Xp[l], h)
        if aux <= 1 / 10 ** 320: 
            aux = 1 / 10 ** 320
        H += -math.log(aux)
        H = H / M
    return H


def detect_separator(trajectories, secs):
    traj = trajectories[0]
    for i in range(len(traj)):
        if traj[i, 4] - traj[0, 4] > secs:
            break
    return i - 1


# Visualizing a trajectory and its comparison to others
def visualize_max_entropy(trajs, n, path, name, replace, output_dir):

    ref_path = os.path.join(path, 'reference.png')
    H_path = os.path.join(path, 'H.txt')

    if not os.path.exists(H_path):
        return None

    H = np.loadtxt(H_path)
    Hinv = np.linalg.inv(H)

    plt.figure(figsize=(10, 10))
    img = plt.imread(ref_path)
    plt.imshow(img)

    for i in range(len(trajs)):
        traj = trajs[i]
        cat = np.vstack([traj[:, 0], traj[:, 1], np.ones_like(traj[:, 0])]).T
        tCat = (Hinv @ cat.T).T

        # Get points in image
        x = tCat[:, 1] / tCat[:, 2]
        y = tCat[:, 0] / tCat[:, 2]

        if i == n:
            xn = x
            yn = y
        elif i % 2 == 0:
            plt.plot(x, y, c='blue', linewidth=2)

    plt.plot(xn, yn, c='yellowgreen', linewidth=4)

    if replace == True:
        R = 'R'
    else:
        R = 'NR'

    plt.savefig(f'{output_dir}/{name}-max_entropy-{R}.svg', format='svg')
    plt.clf()


def get_entropies(trajectories, M=30, replace=False):
    # Load dataset
    N_t = len(trajectories)

    # Number of frames in observed and predicted trajlets
    Tobs = detect_separator(trajectories, 3.2)
    Tpred = len(trajectories[0]) - Tobs

    h = 0.5  # Bandwidth for Gaussian Kernel

    # Leave just the position information
    trajs = []
    for i in range(len(trajectories)):
        trajs.append(trajectories[i][:, 0:2])

    # Obtain observed and predicted trajlets
    Xm, Xp = separate_trajectories(trajs, Tobs, Tpred + Tobs)

    # Estimate the entropy for every trajectory
    entropy_values = []
    for k in tqdm(range(N_t)):
        w = weights(Xm, k, h)
        entropy_values.append(entropy(Xp, k, h, w, M, replace=replace))

    return entropy_values


def entropies_set(trajlets, output_dir, M=30, replace=False, num_samples: int = 1_000):
    entropies_dir = os.path.join(output_dir, 'entropies')
    if os.path.exists(entropies_dir) is False: 
        os.makedirs(entropies_dir)

    dataset_names = list(trajlets.keys())
    entropy_values_set = dict()

    R = '-R' if replace else '-NR'

    for ds_name in dataset_names:
        print("\nProcessing", ds_name)
        ds_trajs = trajlets[ds_name]

        trajlet_entropy_file = os.path.join(entropies_dir, f'{ds_name}-{M}-entropy-{R}.npy')
        if os.path.exists(trajlet_entropy_file):
            entropy_values = np.load(trajlet_entropy_file)
            print("loading entropies from:", trajlet_entropy_file)
        else:
            N = len(ds_trajs)
            if 1 < num_samples < N:
                s = rd.sample(range(N), num_samples)
                ss_trajs = ds_trajs[list(s)]
                entropy_values = get_entropies(ss_trajs, M, replace=replace)
            else:
                entropy_values = get_entropies(ds_trajs, M, replace=replace)
            print("writing entropies into:", trajlet_entropy_file)
            np.save(trajlet_entropy_file, entropy_values)

        entropy_values_set.update({ds_name: entropy_values})
    
    return entropy_values_set


def run(trajlets, output_dir, replace=True):

    ds_S = entropies_set(trajlets, output_dir, M=30, replace=replace)

    all_labels = []
    all_maximum = []
    all_entropies = []

    for dataset, ds_s in ds_S.items():
        all_entropies.append(ds_s)
        max_index = np.argmax(ds_s)
        for i in range(len(ds_s)):
            all_labels.append(dataset)
            all_maximum.append(i == max_index)
            
    data = pd.DataFrame(data={'Entropy': np.concatenate(all_entropies, axis=0), 
                              'Dataset': all_labels})

    print("\nVisualizing ...")
    sns.set(style="whitegrid")    
    plt.figure(figsize=(20, 8))

    ax = sns.swarmplot(y='Entropy', x='Dataset', data=data, size=1) #, hue=maximum)
    if replace == True:
        title = 'Entropies with replacement'
        R = 'R'
    else:
        title = 'Entropies without replacement'
        R = 'NR'

    plt.title(title)
    plt.xticks(rotation=330)
    plt.savefig(os.path.join(output_dir, f'traj_entropy_{R}.png'), dpi=500, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    from ..loaders.loader_all import get_trajlets

    opentraj_root = "./datasets"
    output_dir = "./cache/benchmark/trajectory_entropy"
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)

    trajlets = get_trajlets(opentraj_root)
    run(trajlets, output_dir)
