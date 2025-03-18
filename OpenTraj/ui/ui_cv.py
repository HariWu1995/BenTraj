import os
import json
import argparse

import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from ..loaders.loader_meta import load_meta


def error_msg(msg):
    print('Error:', msg)
    exit(-1)


def to_image_frame(Hinv, loc):
    """
    Given H^-1 and world coordinates, returns (u, v) in image coordinates.
    """
    if loc.ndim > 1:
        locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
        loc_tr = np.transpose(locHomogenous)
        loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
        locXYZ = np.transpose(loc_tr/loc_tr[2])  # to pixels (from millimeters)
        return locXYZ[:, :2].astype(int)
    else:
        locHomogenous = np.hstack((loc, 1))
        locHomogenous = np.dot(Hinv, locHomogenous)  # to camera frame
        locXYZ = locHomogenous / locHomogenous[2]  # to pixels (from millimeters)
        return locXYZ[:2].astype(int)


class Play:

    def __init__(self):
        pass

    def check(self, filename):
        return any([filename.endswith(ext) for ext in ['.mp4','.avi']])

    def set_background_im(self, im, timestamp=-1):
        self.bg_im = im.copy()
        cv2.putText(self.bg_im, '%d' % timestamp, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    def draw_trajectory(self, id, ll, color, width):
        for t1 in range(ll.shape[0] - 1):
            t2 = t1 + 1
            cv2.line(self.bg_im, (ll[t1][1], ll[t1][0]), 
                                 (ll[t2][1], ll[t2][0]), color, width)

    def draw_agent(self, id, pos, radius, color, width):
        cv2.circle(self.bg_im, (pos[1], pos[0]), radius, color, width)

    def play(self, traj_dataset, Hinv, media_file):

        traj_df = traj_dataset.data
        frame_ids = sorted(traj_df['frame_id'].unique())

        if os.path.exists(media_file):
            if self.check(media_file):
                cap = cv2.VideoCapture(media_file)
            else:
                ref_im = cv2.imread(media_file)

        ids_t = []
        frame_id = 0
        pause = False

        while True:
            if self.check(media_file) and not pause:
                ret, ref_im = cap.read()

            # ref_im_copy = np.copy(ref_im)
            self.set_background_im(ref_im, frame_id)

            if frame_id in frame_ids:
                xys_t = traj_df[['pos_x', 'pos_y']].loc[traj_df["frame_id"] == frame_id].to_numpy()
                ids_t = traj_df['agent_id'].loc[traj_df["frame_id"] == frame_id].to_numpy()

            all_trajs = traj_df[(traj_df['agent_id'].isin(ids_t)) &
                                (traj_df['frame_id'] <= frame_id) &
                                (traj_df['frame_id'] > frame_id - 50)  # TODO: replace it with timestamp
                                ].groupby('agent_id')
            ids_t = [key for key, value in all_trajs]
            all_trajs = [value[['pos_x','pos_y']].to_numpy() for key, value in all_trajs]

            for i, id in enumerate(ids_t):
                xy_i = np.array(xys_t[i])
                UV_i = to_image_frame(Hinv, xy_i)

                # fetch entire trajectory
                traj_i = all_trajs[i]
                TRAJ_i = to_image_frame(Hinv, traj_i)

                self.draw_trajectory(id, TRAJ_i, (255, 255, 0), 2)
                self.draw_agent(id, (UV_i[0], UV_i[1]), 5, (0, 0, 255), 2)

            # if not ids_t:
            #     print('No agent')

            if not pause and frame_id < frame_ids[-1]:
                frame_id += 1

            delay_ms = 20
            cv2.namedWindow('OpenTraj (Press ESC for exit)', cv2.WINDOW_NORMAL)
            cv2.imshow('OpenTraj (Press ESC for exit)', self.bg_im)

            key = cv2.waitKey(delay_ms * (1 - pause)) & 0xFF
            if key == 27:  
                # press ESCAPE to quit
                break
            elif key == 32:  
                # press SPACE to pause/play
                pause = not pause


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='OpenTraj - Human Trajectory Dataset Package')
    argparser.add_argument('--data-root', '-d', default='./datasets', 
                                                help='the root address of OpenTraj directory')
    argparser.add_argument('--metafile', '-m', help='dataset meta file (*.json)', 
                                                default='ETH/ETH-ETH.json')
    argparser.add_argument('--background', '-b', default='video', 
                                                choices=['video', 'image'],
                                                help='Select background type. (default: "video") '
                                                    'You might need to download video first.')
    args = argparser.parse_args()

    opentraj_root = args.data_root
    opentraj_metafile = os.path.join(opentraj_root, args.metafile).replace('\\','/')

    if not os.path.exists(opentraj_metafile) \
    or not opentraj_metafile.endswith('.json'):
        error_msg('Please Enter a valid dataset metafile (*.json)')

    with open(opentraj_metafile) as f:
        data = json.load(f)

    if 'calib_to_world_path' in data:
        homog2world_file = os.path.join(opentraj_root, data['calib_to_world_path'])
    else:
        homog2world_file = ""
    Homog_to_world = np.loadtxt(homog2world_file) if os.path.exists(homog2world_file) else np.eye(3)

    # if 'calib_to_camera_path' in data:
    #     homog2cam_file = os.path.join(opentraj_root, data['calib_to_camera_path'])
    if 'calib_path' in data:
        homog2cam_file = os.path.join(opentraj_root, data['calib_path'])
    else:
        homog2cam_file = ""
    Homog_to_camera = np.loadtxt(homog2cam_file) if os.path.exists(homog2cam_file) else np.eye(3)
    Hinv = np.linalg.inv(Homog_to_camera)

    traj_dataset = load_meta(opentraj_root, opentraj_metafile, homog_file=homog2world_file)
    if not traj_dataset:
        error_msg('dataset name is invalid')

    if args.background == 'image':
        media_file = os.path.join(opentraj_root, data['ref_image'])
    elif args.background == 'video':
        media_file = os.path.join(opentraj_root, data['video'])
    else:
        error_msg('background type is invalid')

    play = Play()
    play.play(traj_dataset, Hinv, media_file)
