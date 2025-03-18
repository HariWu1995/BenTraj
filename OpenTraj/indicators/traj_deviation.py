# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches
# matplotlib.use('PS')

from ..core.trajdataset import TrajDataset


def deviation_from_linear_pred(trajlets, output_dir, ds_name):
    dp_from_t0 = trajlets[:, :, :2] - np.expand_dims(trajlets[:, 0, :2], 1)

    displacement_1sig_idx = (np.linalg.norm(dp_from_t0, axis=2) > 0.25).argmax(axis=1)
    displacement_1sig = np.stack([dp_from_t0[i, displacement_1sig_idx[i], :2]
                                         for i in range(len(trajlets))])

    # start_thetas = np.arctan2(trajlets[:, 0, 2], trajlets[:, 0, 3])  # calculated from 1rst velocity vector
    # start_thetas = np.arctan2(trajlets[:, 2, 0] - trajlets[:, 0, 0],
    #                           trajlets[:, 2, 1] - trajlets[:, 0, 1])
    start_thetas = np.arctan2(displacement_1sig[:, 0], displacement_1sig[:, 1])

    rot_matrices = np.stack([np.array([[np.cos(theta), -np.sin(theta)],
                                       [np.sin(theta),  np.cos(theta)]]) for theta in start_thetas])
    
    trajs_zero = trajlets[:, :, :2] - trajlets[:, 0, :2].reshape((-1, 1, 2))

    trajs_aligned = np.matmul(rot_matrices, trajs_zero.transpose((0, 2, 1))).transpose((0, 2, 1))
    idx_not_null = ~np.any(np.any(np.isnan(trajs_aligned), axis=2), axis=1)
    trajs_aligned = trajs_aligned[idx_not_null, :, :]

    keypoints = np.mean(trajs_aligned[:, :, :], axis=0)
    keypoints_radius = np.linalg.norm(keypoints, axis=1)
    keypoints_dev_avg = np.rad2deg(np.arctan2(keypoints[:, 0], keypoints[:, 1]))
    keypoints_dev_std = np.std(np.rad2deg(np.arctan2(trajs_aligned[:, :, 0],
                                                     trajs_aligned[:, :, 1])), axis=0)

    # ======== PLOT ============
    fig1, ax1 = plt.subplots()
    
    trajs_plt = ax1.plot(trajs_aligned[:, :, 1].T, 
                         trajs_aligned[:, :, 0].T, alpha=0.3, color='blue')
    
    avg_plt = ax1.plot(keypoints[::2, 1], 
                       keypoints[::2, 0], 'o', color='red')

    for ii in range(2, len(keypoints), 2):
        arc_i = patches.Arc([0, 0], zorder=10,
                                     width=keypoints_radius[ii] * 2,
                                    height=keypoints_radius[ii] * 2,
                                    theta1=keypoints_dev_avg[ii] - keypoints_dev_std[ii],
                                    theta2=keypoints_dev_avg[ii] + keypoints_dev_std[ii])
        ax1.add_patch(arc_i)

    ax1.grid()
    ax1.set_aspect('equal')
    plt.title(ds_name)
    plt.xlim([-1.5, 10])
    plt.ylim([-4, 4])
    plt.legend(handles=[trajs_plt[0], avg_plt[0]], labels=["trajlets", "avg"], loc="lower left")

    plt.savefig(os.path.join(output_dir, f'dev-{ds_name}.png'))
    # plt.show()
    plt.close()

    return keypoints_dev_avg, keypoints_dev_std


def run(trajlets, output_dir):
    
    dataset_names = list(trajlets.keys())
    deviation_stats = {
        1.6: [], 
        2.4: [], 
        4.8: [],
    }

    for ds_name in dataset_names:
        print("\nProcessing", ds_name)
        ds_trajs = trajlets[ds_name]
        dev_avg, \
        dev_std = deviation_from_linear_pred(ds_trajs, output_dir, ds_name)

        for t in deviation_stats.keys():
            dt = np.diff(ds_trajs[0, :, 4])[0]
            time_index = int(round(t/dt))-1
            deviation_stats[t].append([dev_avg[time_index], dev_std[time_index]])

    deviation_stats[1.6] = np.array(deviation_stats[1.6])
    deviation_stats[2.4] = np.array(deviation_stats[2.4])
    deviation_stats[4.8] = np.array(deviation_stats[4.8])

    print("\nVisualizing ...")
    fig = plt.figure(figsize=(len(dataset_names)+2, 7))

    ax1 = fig.add_subplot(311)
    plt.bar(np.arange(len(dataset_names)), 
            abs(deviation_stats[4.8][:, 0]),
            yerr=deviation_stats[4.8][:, 1], alpha=0.7, color='red',
        error_kw=dict(ecolor='blue', lw=2, capsize=5, capthick=2))
    
    plt.xticks([])
    plt.yticks([-30, -15, 0, 15, 30], ['$-30^o$', '$-15^o$', '$0^o$', '$15^o$', '$30^o$'])
    # plt.yticks([-30, 0, 30])
    plt.grid(axis='y', linestyle='--')
    plt.ylabel('$t=4.8s$')

    ax2 = fig.add_subplot(312)
    plt.bar(np.arange(len(dataset_names)), 
            abs(deviation_stats[2.4][:, 0]),
            yerr=deviation_stats[2.4][:, 1], alpha=0.7, color='red',
        error_kw=dict(ecolor='blue', lw=2, capsize=5, capthick=2))
    
    plt.xticks([])
    plt.yticks([-20, -10, 0, 10, 20], ['$-20^o$', '$-10^o$', '$0^o$', '$10^o$', '$20^o$'])
    plt.grid(axis='y', linestyle='--')
    plt.ylabel('$t=2.4s$')

    ax3 = fig.add_subplot(313)
    plt.bar(np.arange(len(dataset_names)), 
            abs(deviation_stats[1.6][:, 0]),
            yerr=deviation_stats[1.6][:, 1], alpha=0.7, color='red',
        error_kw=dict(ecolor='blue', lw=2, capsize=5, capthick=2))

    plt.ylabel('$t=1.6s$')
    plt.xticks(np.arange(0, len(dataset_names), 1.0))
    plt.grid(axis='y', linestyle='--')
    # plt.xticks(rotation=-45)

    plt.subplots_adjust(wspace=0, hspace=.10)
    plt.savefig(os.path.join(output_dir, 'traj_deviation.png'), dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    
    from ..loaders.loader_all import get_trajlets

    opentraj_root = "./datasets"
    output_dir = "./cache/benchmark/trajectory_deviation"
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)

    trajlets = get_trajlets(opentraj_root)
    run(trajlets, output_dir)

