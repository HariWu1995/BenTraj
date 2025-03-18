import os
import time
import argparse

import numpy as np
import cv2

from ..loaders.loader_eth import load_eth
from ..loaders.loader_sdd import load_sdd, load_sdd_dir
from ..loaders.loader_gcs import load_gcs
from ..loaders.loader_ucy import load_ucy
from ..loaders.loader_hermes import load_hermes

from ..ui.pyqt.qtui.opentrajui import OpenTrajUI


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

    def __init__(self, gui_mode: str = 'pyqt'):
        bg_im = None
        self.gui_mode_pyqt = (args.gui_mode == 'pyqt')
        self.gui_mode_opencv = (args.gui_mode == 'opencv')
        if self.gui_mode_pyqt:
            self.qtui = OpenTrajUI(reserve_n_agents=100)
            self.agent_index = -1

    def check(self, filename):
        return any([filename.endswith(ext) for ext in ['.mp4','.avi']])

    def set_background_im(self, im, timestamp=-1):
        self.bg_im = im.copy()
        if self.gui_mode_pyqt:
            self.qtui.update_im(im)
            self.qtui.erase_paths()
            self.qtui.erase_circles()
        if timestamp >= 0:
            if self.gui_mode_pyqt:
                self.qtui.setTimestamp(timestamp)
            elif self.gui_mode_opencv:
                cv2.putText(self.bg_im, '%d' % timestamp, (30, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    def draw_trajectory(self, id, ll, color, width):
        if self.gui_mode_pyqt:
            self.qtui.draw_path(ll[..., ::-1], color, [width])

        elif self.gui_mode_opencv:
            for t1 in range(ll.shape[0] - 1):
                t2 = t1 + 1
                cv2.line(self.bg_im, (ll[t1][1], ll[t1][0]), 
                                     (ll[t2][1], ll[t2][0]), color, width)

    def draw_agent(self, id, pos, radius, color, width):
        if self.gui_mode_pyqt:
            self.qtui.draw_circle(pos[::-1], radius, color, width)
        elif self.gui_mode_opencv:
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
            if self.gui_mode_pyqt:
                self.qtui.processEvents()
                pause = self.qtui.pause
                time.sleep(delay_ms/1000.)
            else:
                # cv2.namedWindow('OpenTraj (Press ESC for exit)', cv2.WINDOW_NORMAL)
                # cv2.imshow('OpenTraj (Press ESC for exit)', ref_im_copy)
                key = cv2.waitKey(delay_ms * (1 - pause)) & 0xFF
                if key == 27:  # press ESCAPE to quit
                    break
                elif key == 32:  # press SPACE to pause/play
                    pause = not pause


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='OpenTraj - Human Trajectory Dataset Package')
    argparser.add_argument('--data-root', '-d', 
                            default='./datasets', 
                            help='the root address of OpenTraj directory')
    argparser.add_argument('--dataset', '-ds',
                           default='eth',
                           choices=['eth','hotel','zara01','zara02','students03','gc','sdd-bookstore0','Hermes-xxx'],
                           help='select dataset (default: "eth")')
    argparser.add_argument('--gui-mode', '-g',
                           default='pyqt', 
                           choices=['pyqt', 'opencv'],
                           help='pick a specific mode for gui (default: "pyqt")')
    argparser.add_argument('--background', '-b', 
                            default='video', 
                            choices=['video', 'image'],
                            help='Select background type. (default: "video") '
                                'You might need to download video first.')

    args = argparser.parse_args()

    opentraj_root = args.data_root

    # #============================ ETH =================================
    if args.dataset == 'eth':
        annot_file = os.path.join(opentraj_root, 'ETH/seq_eth/obsmat.txt')
        traj_dataset = load_eth(annot_file)
        homog_file = os.path.join(opentraj_root, 'ETH/seq_eth/H.txt')
        if args.background == 'image':
            media_file = os.path.join(opentraj_root, 'ETH/seq_eth/reference.png')
        elif args.background == 'video':
            media_file = os.path.join(opentraj_root, 'ETH/seq_eth/video.avi')
        else:
            error_msg('background type is invalid')

    elif args.dataset == 'hotel':
        annot_file = os.path.join(opentraj_root, 'ETH/seq_hotel/obsmat.txt')
        traj_dataset = load_eth(annot_file)
        homog_file = os.path.join(opentraj_root, 'ETH/seq_hotel/H.txt')
        media_file = os.path.join(opentraj_root, 'ETH/seq_hotel/video.avi')
        # media_file = os.path.join(opentraj_root, 'ETH/seq_hotel/reference.png')

    # #============================ UCY =================================
    # elif args.dataset == 'zara01':
    #     traj_dataset = ParserETH()
    #     parser = ParserETH()
    #     annot_file = os.path.join(opentraj_root, 'UCY/zara01/obsmat.txt')
    #     homog_file = os.path.join(opentraj_root, 'UCY/zara01/H.txt')
    #     # media_file = os.path.join(opentraj_root, 'UCY/zara01/reference.png')
    #     media_file = os.path.join(opentraj_root, 'UCY/zara01/video.avi')
    #
    # elif args.dataset == 'zara01':
    #     traj_dataset = ParserETH()
    #     annot_file = os.path.join(opentraj_root, 'UCY/zara02/obsmat.txt')
    #     homog_file = os.path.join(opentraj_root, 'UCY/zara02/H.txt')
    #     # media_file = os.path.join(opentraj_root, 'UCY/zara02/reference.png')
    #     media_file = os.path.join(opentraj_root, 'UCY/zara02/video.avi')
    #
    # elif args.dataset == 'students03':
    #     traj_dataset = ParserETH()
    #     # annot_file = os.path.join(opentraj_root, 'UCY/st3_dataset/obsmat_px.txt')
    #     # media_file = os.path.join(opentraj_root, 'UCY/st3_dataset/reference.png')
    #     # homog_file = os.path.join(opentraj_root, 'UCY/st3_dataset/H_iw.txt')
    #     # # homog_file = ''
    #
    #     annot_file = os.path.join(opentraj_root, 'UCY/st3_dataset/obsmat.txt')
    #     homog_file = os.path.join(opentraj_root, 'UCY/st3_dataset/H.txt')
    #     # media_file = os.path.join(opentraj_root, 'UCY/st3_dataset/reference.png')
    #     media_file = os.path.join(opentraj_root, 'UCY/st3_dataset/video.avi')

    # #============================ SDD =================================
    # elif args.dataset == 'sdd':
    #     dataset_parser = ParserSDD()
    #     annot_file = os.path.join(opentraj_root, 'SDD/bookstore/video0/annotations.txt')
    #     media_file = os.path.join(opentraj_root, 'SDD/bookstore/video0/reference.jpg')
    #     homog_file = ''

    # #============================ GC ==================================
    # elif args.dataset == 'gc':
    #     gc_world_coord = True
    #     traj_dataset = ParserGC(world_coord=gc_world_coord)
    #     annot_file = os.path.join(opentraj_root, 'GC/Annotation')  # image coordinate
    #     if gc_world_coord:
    #         homog_file = os.path.join(opentraj_root, 'GC/H-world.txt')
    #         media_file = os.path.join(opentraj_root, 'GC/plan.png')
    #     else:
    #         homog_file = os.path.join(opentraj_root, 'GC/H-image.txt')
    #         media_file = os.path.join(opentraj_root, 'GC/reference.jpg')

    # ========================== HERMES =================================
    # elif args.dataset == 'hermes':
    #     dataset_parser = ParserHermes()
    #     annot_file = os.path.join(opentraj_root, 'HERMES/Corridor-1D/uo-070-180-180.txt')
    #     annot_file = os.path.join(opentraj_root, 'HERMES/Corridor-2D/boa-300-050-070.txt')
    #     media_file = os.path.join(opentraj_root, 'HERMES/cor-180.jpg')
    #     homog_file = os.path.join(opentraj_root, 'HERMES/H.txt')

    else:
        traj_dataset = None

    if not traj_dataset:
        error_msg('dataset name is invalid')

    Homog = (np.loadtxt(homog_file)) if os.path.exists(homog_file) else np.eye(3)
    Hinv = np.linalg.inv(Homog)

    play = Play(args.gui_mode)
    play.play(traj_dataset, Hinv, media_file)
    # qtui.app.exe()
