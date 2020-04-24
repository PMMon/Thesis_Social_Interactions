import os, sys
sys.path.append(os.getcwd())
sys.path.append( os.path.abspath('.'))
import logging
import numpy as np
import torch
from data.BaseTrajectories import BaseDataset
from PIL import Image, ImageOps
import copy


logger = logging.getLogger(__name__)

# Rotate input around center by angle alpha
def rotate(X, center, alpha):
    XX = X.copy()

    XX[:, 0] = (X[:, 0] - center[0]) * np.cos(alpha) + (X[:, 1] - center[1]) * np.sin(alpha) + center[0]
    XX[:, 1] = - (X[:, 0] - center[0]) * np.sin(alpha) + (X[:, 1] - center[1]) * np.cos(alpha) + center[1]

    return XX


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))



def seq_collate(data):
    obs_traj_list, pred_traj_list, obs_traj_rel_list, pred_traj_rel_list, scene_img_list, occupancy_list = zip(*data)

    obs_traj = torch.cat(obs_traj_list, dim=0).permute(1, 0, 2)
    pred_traj = torch.cat(pred_traj_list, dim=0).permute(1, 0, 2)
    obs_traj_rel = torch.cat(obs_traj_rel_list, dim=0).permute(1, 0, 2)
    pred_traj_rel = torch.cat(pred_traj_rel_list, dim=0).permute(1, 0, 2)




    return {"in_xy": obs_traj,
            "gt_xy": pred_traj,
            "in_dxdy": obs_traj_rel,
            "gt_dxdy": pred_traj_rel,
            "size" : torch.LongTensor([obs_traj.size(1)]),
            "scene_img": scene_img_list,
            "occupancy" : occupancy_list,

            }


    


class TrajectoryDataset(BaseDataset):
    """Dataloder for the Trajectory datasets"""
    #TODOD: check framerates
    def __init__(
        self,
        save = False,
        load_p = True,
        dataset_name  ="stanford",
        phase = "test",
        obs_len=8,
        pred_len=12,
        time_step = 0.4,
        skip=1,
        data_augmentation = 0,
        scale_img = True,
        max_num = 5,
        logger = logger,
        special_scene = None,
        load_occupancy = False
         ):

        super(TrajectoryDataset, self).__init__(save, load_p, dataset_name, phase,
                                                obs_len, pred_len,  time_step, skip,
                                                data_augmentation,  scale_img,  max_num,
                                                logger, special_scene, load_occupancy)

    def load(self):
        seqs = []
        seqs_rel = []
        scene_nr = []
        self.scene_list = []
        self.images = {}
        self.image_list = []


        collect_data = True

        # Process images
        for path in [file for file in self.all_files if ".jpg" in file]:
            print( path)
            scene_path, data_type = path.split(".")
            scene = scene_path.split("/")[-1]

            img_parts = scene.split("-")

            if self.load_occupancy and img_parts[-1] =="op":
              
                scene = img_parts[-2 ]
                self.load_image(path, scene)

            elif not self.load_occupancy and img_parts[-1] !="op":
                self.load_image(path, scene)

                continue

        if len(self.images)== 0:
            assert False, "No valid imges in folder"


        # Process txt-files with coordinates
        for path in [file for file in self.all_files if ".txt" in file]:
            if not collect_data:

                break
            if self.special_scene and not self.special_scene in path:
                continue

            scene_path, data_type = path.split(".")
            scene = scene_path.split("/")[-1]
           
            if data_type == "txt":
                print(scene)
                scene = '_'.join(scene.split("_")[1:])

                self.logger.info("preparing %s" % scene)

                # get data
                data = self.load_file(path,  self.delim)

                # Get all ids (avoid duplicates)
                unique_ids = np.unique(data[:, 1])

                # if max number get only first max_num ids
                if self.max_num:
                    unique_ids = unique_ids[:self.max_num]
                    collect_data = False

                for id in unique_ids:
                    # look at isolated trajectory of pedestr with respective id
                    trajectory = data[data[:, 1] == id][:, (0, 1, 2, 3)]

                    # define starting point for trajectory as 0
                    trajectory[:, 0] = (trajectory[:, 0] - np.min(trajectory[:, 0]))  # / 12.

                    # ensure that trajectory has desired seq length and that id appears to be in following sequences
                    if (trajectory[1:, 0] - trajectory[:-1, 0] != 1).any() or len(trajectory[:]) < self.seq_len:

                        continue

                    xy = trajectory[:, (2, 3)]
                    # Why this part?
                    nr_traj = int(len(xy) / self.seq_len)
                    xy = xy[:nr_traj * self.seq_len]
                    # ==========
                    xy = xy.reshape(-1, self.seq_len, 2)

                    for ped in np.arange(len(xy)):

                        img = self.images[scene]["scaled_image"]
                        if self.data_augmentation:

                            if self.format == "pixel":
                                scale2orig = 1 / self.images[scene]["scale_factor"]
                            elif self.format == "meter":
                                scale2orig = self.img_scaling
                            else:
                                assert False, " Not valid format '{}': 'meters' or 'pixel'".format(self.format)
                            alpha = np.random.rand() * 2 * np.pi
                            center = np.array(img.size) / 2.
                            corners = np.array([[0, 0], [0, img.height], [img.width, img.height], [img.width, 0]])

                            rand_num = np.random.choice(np.arange(3))

                            if rand_num != 0:
                                if rand_num == 1:

                                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                                    xy[ped, :, 0] = img.width * scale2orig - xy[ped, :, 0]


                                elif rand_num == 2:
                                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                                    xy[ped, :, 1] = img.height * scale2orig - xy[ped, :, 1]
                                    # transform wall

                            img = img.rotate(alpha / np.pi * 180, expand=True)

                            corners_trans = rotate(corners, center, alpha)
                            offset = corners_trans.min(axis=0)
                            corners_trans -= offset

                            xy[ped] = rotate(xy[ped], center * scale2orig, alpha) - offset * scale2orig



                        self.image_list.append({"ratio": self.images[scene]["ratio"], "scene": scene, "scaled_image": copy.copy(img)})
                        self.scene_list.append(scene)
                        scene_nr.append(1)

                    # get relative distance to previous spot
                    dxdy = xy[:, 1:] - xy[:, :-1]

                    # append distance and rel dist to previous spot
                    seqs.append(xy)
                    seqs_rel.append(dxdy)

        # Create one list for distances
        seq_list = np.concatenate(seqs,  axis = 0)
        seq_list_rel = np.concatenate(seqs_rel,  axis = 0)

        self.num_seq = len(seq_list)
        # Convert numpy -> Torch Tensor
        self.scene_nr = torch.LongTensor(np.cumsum(scene_nr))

        self.obs_traj = torch.from_numpy(
            seq_list[:,  :self.obs_len]).type(torch.float) 
        self.pred_traj = torch.from_numpy(
            seq_list[:, self.obs_len:]).type(torch.float) 
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :self.obs_len-1]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, self.obs_len-1:]).type(torch.float)

        del seq_list
        del seq_list_rel



        if self.scale:
            self.scale_func()
        if self.norm2meters:
            self.scale2meters()


        if self.save:
            self.save_dset()





    def get_scene(self, index):
        scene_img = self.image_list[index]

        return {"in_xy":    self.obs_traj[index].unsqueeze(1),
                "gt_xy" :   self.pred_traj[index].unsqueeze(1),
                "in_dxdy":  self.obs_traj_rel[index].unsqueeze(1),
                "gt_dxdy":  self.pred_traj_rel[index].unsqueeze(1),
                "scene":    self.scene_list[index],
                "scene_nr": self.scene_nr[index], 
                "img" : scene_img,
                "scene_img" : [scene_img],
                }
    def __getitem__(self, index):

        scene_img = self.image_list[index]

        if self.wall_available:
            walls = self.walls_list[index]
        else:
            walls = torch.empty(1)


        return [self.obs_traj[index].unsqueeze(0),
                self.pred_traj[index].unsqueeze(0),
                self.obs_traj_rel[index].unsqueeze(0),
                self.pred_traj_rel[index].unsqueeze(0),
                scene_img,
                walls,
                ]



if __name__ == "__main__":


    from torch.utils.data import DataLoader
    print("Start Trajectory")
    current_dir = os.curdir

    dataset = TrajectoryDataset( dataset_name = "hotel",
                                 phase = "test",
                                 obs_len=8,
                                 pred_len=12,
                                 data_augmentation = 0,
                                 skip=20, max_num= 1000 ,
                                 logger = logger)

    batch= dataset.get_scene(18)

    loader = DataLoader(
        dataset,
        batch_size = 32,
        shuffle=False,
        collate_fn = seq_collate)

    batch = next(iter(loader))
    print("Test batch", batch)
