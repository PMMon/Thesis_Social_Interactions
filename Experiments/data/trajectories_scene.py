import os, sys

sys.path.append(os.getcwd())

import logging

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from data.analyse_dataset import analyse_dataset
from data.BaseTrajectories import BaseDataset
from data.tools import image_json
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math

from helper.debug_print import debug_print

import data.experiments

logger = logging.getLogger(__name__)


def rotate(X, center, alpha):
    XX = X.copy()

    XX[:, 0] = (X[:, 0] - center[0]) * np.cos(alpha) + (X[:, 1] - center[1]) * np.sin(alpha) + center[0]
    XX[:, 1] = - (X[:, 0] - center[0]) * np.sin(alpha) + (X[:, 1] - center[1]) * np.cos(alpha) + center[1]

    return XX


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def flatten(l):
    return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

def seq_collate_eval(data):
    obs_traj_list, pred_traj_list, obs_traj_rel_list, pred_traj_rel_list, pred_check, pad_mask = zip(*data)

    _len = [len(seq) for seq in obs_traj_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = torch.LongTensor([[start, end]for start, end in zip(cum_start_idx, cum_start_idx[1:])])

    obs_traj = torch.cat(obs_traj_list, dim=0).permute(1, 0, 2)
    pred_traj = torch.cat(pred_traj_list, dim=0).permute(1, 0, 2)
    obs_traj_rel = torch.cat(obs_traj_rel_list, dim=0).permute(1, 0, 2)
    pred_traj_rel = torch.cat(pred_traj_rel_list, dim=0).permute(1, 0, 2)


    return {"in_xy": obs_traj,
            "gt_xy": pred_traj,
            "in_dxdy": obs_traj_rel,
            "gt_dxdy": pred_traj_rel,
            "size": torch.LongTensor([len(seq_start_end)]),
            "seq_start_end": seq_start_end,
            "pred_check": pred_check,
            "pad_mask": pad_mask,
            }


class TrajectoryDatasetEval(BaseDataset):
    """Dataloder for the Trajectory datasets"""

    # TODOD: check framerates
    def __init__(self, **kwargs):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDatasetEval, self).__init__(**kwargs)

        #Todo: can I delete this safely?
        #BaseDataset.__init__(self, **kwargs)
        self.__dict__.update(locals())


    def load(self):

        scene_nr = []

        seq_list = []
        self.prediction_check = []
        self.pad_mask_list = []
        ped_skipped = 0
        skipped_ped = []
        total_ped = 0



        collect_data = True

        num_peds_in_seq = []
        for path in [file for file in self.all_files if ".txt" in file]:
            if not collect_data:
                break
            if self.special_scene and not self.special_scene in path:
                continue

            scene_path, data_type = path.split(".")
            scene = scene_path.split("/")[-1]

            if data_type == "txt":

                scene = '_'.join(scene.split("_")[1:])

                self.logger.info("preparing %s" % scene)
                data = self.load_file(path, self.delim)

                frames = np.unique(data[:, 0]).tolist()
                frame_data = []

                if self.dataset_type == "real":
                    frames = range(int(min(frames)), int(max(frames)+1))

                    if self.analyse_real_dataset:
                        # ==== Analyse real datasets =====
                        ad = analyse_dataset(scene, data)
                        ad.get_all_trajectories()

                for frame in frames:
                    if frame in data[:, 0]:
                        frame_data.append(data[frame == data[:, 0], :])
                    else:
                        recent_run = np.concatenate((np.array([frame]), frame_data[frame-1][0][1:]), axis=0)
                        recent_run = np.expand_dims(recent_run, axis=0)
                        frame_data.append(recent_run)


                if __name__ == "__main__":
                    print("Nr. of frames: " + str(len(frames)))

                num_sequences = int(math.ceil((len(frames) - self.seq_len) / self.skip))


                for idx in range(0, num_sequences * self.skip, self.skip):

                    # look at scenes idx to idx +  self.seq_len in format [framenr, pedid, x, y]
                    curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)

                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                    # for padding create mask-matrix to later identify which ped has necessary length for predicition
                    pad_mask = torch.zeros((len(peds_in_curr_seq), self.seq_len))

                    num_peds = 0
                    peds_scene = []
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        # Get sequence of pedestrian with id ped_id
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]


                        # get padding indicees for padding
                        pad_front = frames.index(curr_ped_seq[0,0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1,0]) - idx

                        if self.dataset_type == "real" and self.padding and (curr_ped_seq[1:, 0] - curr_ped_seq[:-1, 0] != 1).any():
                            temp_leaving = (curr_ped_seq[1:, 0] - curr_ped_seq[:-1, 0] != 1)
                            indicees = np.where(temp_leaving == True)[0]
                            indicees_mask = indicees + 1
                            indicees_mask =  np.concatenate((np.array([pad_front-1]),indicees_mask,np.array([pad_end+1])), axis=0)
                            print(indicees)

                            # pad mask
                            for i in range(len(indicees_mask)-1):
                                pad_mask[int(_), indicees_mask[i]+1:indicees_mask[i+1]] = 1

                            # padding
                            padding_start = np.ones((pad_front, curr_ped_seq.shape[1]))
                            padding_start[:, 1] *= ped_id
                            padding_start[:, 0] = np.cumsum(padding_start[:, 0]) - 1 + idx
                            padding_start[:, 2:] *= curr_ped_seq[0, 2:]
                            padding_end = np.ones(((self.seq_len - 1) - pad_end, curr_ped_seq.shape[1]))
                            padding_end[:, 1] *= ped_id
                            padding_end[:, 0] = np.cumsum(padding_end[:, 0]) + pad_end + idx
                            padding_end[:, 2:] *= curr_ped_seq[-1, 2:]

                            curr_ped_seq = np.concatenate((padding_start, curr_ped_seq, padding_end), axis=0)

                            for element in indicees:
                                curr_ped_seq = np.insert(curr_ped_seq, element, values=curr_ped_seq[element,:], axis=0)
                                curr_ped_seq[element + 1, 0] = curr_ped_seq[element + 1, 0] + 1

                        else:
                            pad_mask[int(_), pad_front:(pad_end+1)] = 1

                            # perform padding
                            if self.padding:
                                padding_start = np.ones((pad_front, curr_ped_seq.shape[1]))
                                padding_start[:,1] *= ped_id
                                padding_start[:,0] = np.cumsum(padding_start[:,0]) - 1 + idx
                                padding_start[:, 2:] *= curr_ped_seq[0,2:]
                                padding_end = np.ones(((self.seq_len-1)-pad_end, curr_ped_seq.shape[1]))
                                padding_end[:, 1] *= ped_id
                                padding_end[:, 0] = np.cumsum(padding_end[:, 0]) + pad_end + idx
                                padding_end[:, 2:] *= curr_ped_seq[-1,2:]

                                curr_ped_seq = np.concatenate((padding_start,curr_ped_seq,padding_end), axis=0)


                        # Only use data of ped if ped is over all timesteps in frame - if padding active, this should never be the case
                        if (curr_ped_seq[1:, 0] - curr_ped_seq[:-1, 0] != 1).any() or (len(curr_ped_seq) != self.seq_len):
                            print("here")
                            print("curr_ped_seq: " + str(curr_ped_seq))
                            quit()
                            if ped_id not in skipped_ped:
                                skipped_ped.append(ped_id)
                                ped_skipped += 1
                            continue

                        num_peds += 1

                        # Append coordinates of ped to list
                        peds_scene.append(curr_ped_seq[:, 2:])

                    total_ped += num_peds

                    if num_peds > 0:
                        # check for calculating prediction or not
                        if not self.padding:
                            pad_mask = torch.ones((num_peds, self.seq_len))
                        calc_pred = torch.sum(pad_mask, axis=1).reshape(pad_mask.shape[0], 1)
                        calc_pred[calc_pred[:, 0] < self.seq_len] = 0
                        self.prediction_check.append(calc_pred)
                        # store pad mask
                        self.pad_mask_list.append(pad_mask)

                        num_peds_in_seq.append(num_peds)

                        seq_list.append(np.stack((peds_scene), axis=0))
                        scene_nr.append(1)

        print("ped_skipped: " + str(ped_skipped))

        seq_list = np.concatenate(seq_list, axis=0)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()

        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        seq_list_rel = seq_list[:, 1:] - seq_list[:, :-1]

        self.trajectory = seq_list

        if __name__ == "__main__":
            print("total_nr_of_ped: " + str(total_ped))
            print("nr_of_skipped_ped: " + str(ped_skipped))

            print("trajectories: ", len(self.trajectory))

        self.num_seq = len(self.seq_start_end)
        self.scene_nr = torch.LongTensor(np.cumsum(scene_nr))

        self.obs_traj = torch.from_numpy(
            seq_list[:, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :self.obs_len - 1]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, self.obs_len - 1:]).type(torch.float)

        del seq_list
        del seq_list_rel

        if self.scale:
            self.scale_func()
        if self.norm2meters:
            self.scale2meters()


    def print_wall(self, wall_points):

        for wall in wall_points:
            for index in range(1, len(wall) + 2):
                ind = [index % len(wall), (index + 1) % len(wall)]

                pos = wall[ind, :] / self.img_scaling
                plt.plot(pos[:, 0], pos[:, 1], color='r')


    def get_scene(self, index):

        in_xy, gt_xy, in_dxdy, gt_dxdy, pred_check, pad_mask = self.__getitem__(index)

        return {"in_xy": in_xy.permute(1, 0, 2),
                "gt_xy": gt_xy.permute(1, 0, 2),
                "in_dxdy": in_dxdy.permute(1, 0, 2),
                "gt_dxdy": gt_dxdy.permute(1, 0, 2),
                "seq_start_end": (0, in_xy.size(0)),
                "pred_check": pred_check,
                "pad_mask": pad_mask,
                }

    def scale_func(self):

        for index in np.arange(self.num_seq):
            start, end = self.seq_start_end[index]
            scene = self.scene_list[index]
            self.obs_traj[start:end] *= ratio
            self.pred_traj[start:end] *= ratio
            self.obs_traj_rel[start:end] *= ratio
            self.pred_traj_rel[start:end] *= ratio
            self.trajectory[start:end] *= ratio

    def __getitem__(self, index):


        start, end = self.seq_start_end[index]
        return [self.obs_traj[start:end],
                self.pred_traj[start:end],
                self.obs_traj_rel[start:end],
                self.pred_traj_rel[start:end],
                self.prediction_check[index],
                self.pad_mask_list[index]
                ]


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("Start Trajectory")
    current_dir = os.curdir

    dataset = TrajectoryDatasetEval(dataset_name="squaresimulated_V06b2u6058",
                                phase="test",
                                obs_len=8,
                                pred_len=12,
                                data_augmentation=0,
                                skip=20,
                                max_num=1000,
                                logger=logger,
                                padding=False)

    print("get scene 0...")
    batch = dataset.get_scene(-1)
    last_scene = dataset.__getitem__(-1)
    print("THIS: " +str(last_scene[-2]))
    print("obs_traj_rel.size: " + str(dataset.obs_traj_rel.size()))
    print("seq_start_end: " + str(dataset.seq_start_end))
    print("num_scenes: "+ str(len(dataset.seq_start_end)))
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=seq_collate_eval)

    print("pred_check: " + str(batch["pred_check"]))
    for element in iter(loader):
        batch = element
        print("batch from loader: " + str((batch["pred_check"])))
    #batch = next(next(iter(loader)))
    #print("start_seq_from loader: " + str(batch["seq_start_end"]))
    #print("batch from loader: "+ str((batch["pred_check"])))


"""
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("Start Trajectory")
    current_dir = os.curdir
    dataset = TrajectoryDatasetEval(dataset_name="hotel",
                                    phase="test",
                                    obs_len=8,
                                    pred_len=12,
                                    data_augmentation=0,
                                    skip=20, max_num=1000,
                                    logger=logger)
    print("eval function load:")
    dataset.load()
"""