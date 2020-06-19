import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

from data.analyse_dataset import analyse_dataset
from data.BaseTrajectories import BaseDataset
from helper.debug_print import debug_print

logger = logging.getLogger(__name__)

# ====================================== Description ======================================
# Script to load data of human motion behavior. The defined dataset_fn class  -
# TrajectoryDatasetEval - loads trajectories of pedestrians as described in the respective datasets.
# The DataLoader processes one trajectory after another BUT provides additional information about
# which trajectories belong to the same sequence. Furthermore, it is possible to pad missing
# information for non-suitable trajectories of pedestrians that remain less than
# obs_len + pred_len consecutive time steps in the scene.
# =========================================================================================

def rotate(X, center, alpha):
    """
    Rotate input around center by angle alpha
    """
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
    """
    Customized collate_fn function for customized DataLoader class
    :param data: Processed data of human motion behavior
    :return: Dictionary containing loaded data categorized to:
    * in_xy: Pytorch Tensor of shape: obs_len x nr_trajectories x 2, comprising (x,y)-coordinates of input trajectories
    * gt_xy: Pytorch Tensor of shape: pred_len x nr_trajectories x 2, comprising (x,y)-coordinates of ground truth future trajectories
    * in_dxdy: Pytorch Tensor of shape: obs_len-1 x nr_trajectories x 2, comprising relative coordinates of input trajectories
    * gt_dxdy: Pytorch Tensor of shape: pred_len x nr_trajectories x 2, comprising relative coordinates ground truth future trajectories
    * size: Pytorch Tensor of shape: 1x1, defining the number of trajectories
    * seq_start_end: Tuple of size nr_sequences. The n-th entry contains a tensor of size 1x2 that specifies the ids of the pedestrians in
     the n-th sequence, i.e. pedestrians with ids seq_start_end[n][0] to seq_start_end[n][1] are present in sequence n.
    * pred_check: Tuple of size nr_sequences. The n-th entry contains a tensor of size nr_pedestrians_in_sequence_n x 1. The tensor contains the values
    0 and 1, where the value 0 indicates that the trajectory of the respective pedestrians was padded.
    * pad_mask: Tuple of size nr_sequences. The n-th entry contains a tensor of size nr_pedestrians_in_sequence_n x self.seq_len. The tensor contains the values
    0 and 1, where the value 0 indicates that for the respective time step the trajectory of the respective pedestrians needs to be padded.
    """

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
    """
    Customized dataset_fn class for the Trajectory datasets. This class loads trajectories of human motion behavior and provides
    information about which trajectories are present in a specific scene. It also pads non-suitable trajectories
    of pedestrians that remain less than obs_len + pred_len (= seq_len) consecutive time steps in the scene.
    """
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

        self.__dict__.update(locals())


    def load(self):
        """
        Load data. Dataset-files are .txt-file with format: <frame_id> <ped_id> <x> <y>
        """
        scene_nr = []
        seq_list = []
        self.prediction_check = []
        self.pad_mask_list = []

        total_ped = 0

        ped_skipped = 0
        skipped_ped = []

        collect_data = True

        num_peds_in_seq = []

        # Process txt-files with coordinates
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

                # Get data
                data = self.load_file(path, self.delim)

                # Get all frame-numbers (avoid duplicates)
                frames = np.unique(data[:, 0]).tolist()
                frame_data = []

                if self.dataset_type == "real":
                    frames = range(int(min(frames)), int(max(frames)+1))

                    if self.analyse_real_dataset:
                        # ==== Analyse real datasets =====
                        ad = analyse_dataset(scene, data, self.seq_len)
                        ad.get_all_trajectories()

                # Note that in real-world datasets, sometimes there are frames with no pedestrians moving in the scene. Therefore, pad these frames with old information
                for frame in frames:
                    if frame in data[:, 0]:
                        frame_data.append(data[frame == data[:, 0], :])
                    else:
                        recent_run = np.concatenate((np.array([frame]), frame_data[frame-1][0][1:]), axis=0)
                        recent_run = np.expand_dims(recent_run, axis=0)
                        frame_data.append(recent_run)


                if __name__ == "__main__":
                    print("Nr. of frames: " + str(len(frames)))

                # Get number of sequences
                num_sequences = int(math.ceil((len(frames) - self.seq_len) / self.skip))

                # Iterate over sequences
                for idx in range(0, num_sequences * self.skip, self.skip):

                    # Look at scenes idx to idx +  self.seq_len in format [framenr, pedid, x, y]
                    curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)

                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                    # For padding create mask-matrix to later identify which ped has necessary length for predicition
                    pad_mask = torch.zeros((len(peds_in_curr_seq), self.seq_len))

                    num_peds = 0
                    peds_scene = []
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        # Get sequence of pedestrian with id ped_id
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]

                        # Get padding indicees for padding
                        pad_front = frames.index(curr_ped_seq[0,0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1,0]) - idx

                        if self.dataset_type == "real" and self.padding and (curr_ped_seq[1:, 0] - curr_ped_seq[:-1, 0] != 1).any():
                            # For real datasets we observe that there are pedestrians that leave the scene and enter it at a later point in time -> pad the respective missing time steps
                            temp_leaving = (curr_ped_seq[1:, 0] - curr_ped_seq[:-1, 0] != 1)
                            indicees = np.where(temp_leaving == True)[0]
                            indicees_mask = indicees + 1
                            indicees_mask =  np.concatenate((np.array([pad_front-1]),indicees_mask,np.array([pad_end+1])), axis=0)

                            # Create Pad mask
                            for i in range(len(indicees_mask)-1):
                                pad_mask[int(_), indicees_mask[i]+1:indicees_mask[i+1]] = 1

                            # Perform padding
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
                            # For the synthetic datasets we generate, the pedestrians do not re-enter the scene
                            pad_mask[int(_), pad_front:(pad_end+1)] = 1

                            # Perform padding
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

                        # Only use data of pedestrian if pedestrian is over all timesteps in frame - if padding active, this should never be the case
                        if (curr_ped_seq[1:, 0] - curr_ped_seq[:-1, 0] != 1).any() or (len(curr_ped_seq) != self.seq_len):
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

                        # Store pad mask
                        self.pad_mask_list.append(pad_mask)

                        num_peds_in_seq.append(num_peds)

                        seq_list.append(np.stack((peds_scene), axis=0))
                        scene_nr.append(1)


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
        """
        Get specific scene with index.
        :param index: Index of scene
        :return: Information about scene
        """
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
    # Test customized DataLoader
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
