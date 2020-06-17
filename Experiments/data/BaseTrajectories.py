import os, sys
sys.path.append(os.getcwd())
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import copy
from pathlib import Path

import data.experiments

root_path = Path(os.path.realpath(__file__)).parent.parent
logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    """
    Base Class that resembles basic functionality for the customized DataLoader classes.
    """
    def __init__(self,
                 save=False,
                 load_p=True,
                 dataset_name="zara1",
                 phase="test",
                 obs_len=8,
                 pred_len=12,
                 time_step=0.4,
                 skip=1,
                 data_augmentation=0,
                 scale_img=True,
                 max_num=5,
                 logger=logger,
                 special_scene=None,
                 load_occupancy = False,
                 padding=False,
                 dataset_type="squaresimulated",
                 analyse_real_dataset = False):

        super(BaseDataset, self).__init__()

        self.__dict__.update(locals())
        self.save_dict = copy.copy(self.__dict__)

        self.get_save_path()

        if "simulated" in dataset_name:
            self.dataset = getattr(data.experiments, "Simulated_data")()
            self.__dict__.update(self.dataset.get_dataset_args())
            self.data_dir = self.get_file_path_simulated(dataset_name, phase)
        else:
            self.dataset = getattr(data.experiments, dataset_name)()
            self.__dict__.update(self.dataset.get_dataset_args())
            self.data_dir = self.dataset.get_file_path(phase)

        self.seq_len = self.obs_len + self.pred_len

        all_files = os.listdir(self.data_dir)
        self.all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        self.load_dset()


    def get_file_path_simulated(self, dataset_name, phase):
        if phase == "test":
            dataDir = root_path / 'datasets' / dataset_name / 'test'
        elif phase == "train":
            dataDir =  root_path / 'datasets' / dataset_name / 'train'
        elif phase == "val":
            dataDir = root_path / 'datasets' / dataset_name / 'val'
        else:
            raise AssertionError('"phase" must be either train, val or test.')

        return str(dataDir)


    def get_save_path(self):
        path = ""
        for name, value in self.save_dict.items():
            if type(value) in [int, float, str]:
                path += "{}_{}_".format(name, value)
        path += ".p"
        self.save_path = path


    def load_image(self, _path, scene, ):
        img = Image.open(_path)
        if "stanford" in self.dataset_name:

            ratio = self.homography.loc[ ((self.homography["File"]=="{}.jpg".format(scene)) & (self.homography["Version"]=="A")), "Ratio" ]

            scale_factor = ratio /self.img_scaling
            scale_factor_small = ratio /self.scaling_small
            old_width = img.size[0]
            old_height = img.size[1]

            new_width = int(round(old_width*scale_factor))
            new_height = int(round(old_height*scale_factor))

            scaled_img = img.resize((new_width, new_height ), Image.ANTIALIAS)

        else:
            scaled_img = img
            scale_factor = 1
            ratio = 1.

        self.images.update({scene: {"ratio" : ratio, "scale_factor" : scale_factor,  "scaled_image": scaled_img }})


    def get_ratio(self, scene):
        return self.images[scene]["ratio"]


    def scale_func(self):

        for index in np.arange(self.num_seq):
            scene = self.scene_list[index]
            ratio = self.images[scene]["scale_factor"]
            self.obs_traj[index] *= ratio
            self.pred_traj[index] *= ratio
            self.obs_traj_rel[index] *= ratio
            self.pred_traj_rel[index] *= ratio


    def scale2meters(self):
        self.obs_traj *= self.img_scaling
        self.pred_traj *= self.img_scaling
        self.obs_traj_rel *= self.img_scaling
        self.pred_traj_rel *= self.img_scaling

        self.format = "meter"


    def load_file(self, _path, delim="tab", engine='python'):
        if delim == 'tab':
            delim = "\t"
        elif delim == 'space':
            delim = ' '

        df = pd.read_csv(_path, header=None, delimiter=delim)
        df.columns = self.data_columns

        if "label" and "lost" in df:
            data_settings = {  # "label": "Pedestrian",
                "lost": 0}

            for name, item in data_settings.items():
                df = df[df[name] == item]

            if not "synthetic" in self.dataset_name:
                df = df[df["frame"] % int(round(self.framerate * self.time_step)) == 0]
                df["frame"] /= int(round(self.framerate * self.time_step))

        columns_experiment = ['frame', 'ID', 'x', 'y']
        df = df[columns_experiment]

        return np.asarray(df.values)


    def __len__(self):
        return self.num_seq


    def plot(self, index, modes = ["in_xy", "gt_xy", "in_dxdy", "gt_dxdy"], image_type = "scaled", final_mask = False):
        # Define saving directory for figure
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        saving_directory = os.path.join(root_path, "Saved_Plots", "trajectories")

        out = self.get_scene(index)

        if image_type =="orig":
            img_label = "img"
            img = out["img"][img_label]
            scale= 1
        elif image_type =="scaled":
            img_label = "scaled_image"
            img = out["img"][img_label]
            if self.format == "meter":
                scale = 1. / self.img_scaling

        else:
            assert False,  "'{}' not valid <image_type>".format(image_type)


        center = out["in_xy"][-1, 0] * scale

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)

        for m in modes:
            if m =="gt_xy":
                marker = '-'
            elif m =="gt_dxdy" or m == "in_dxdy":
                marker = "--"
            else:
                marker = '-'

            traj = out["{}".format(m)][:, 0]*scale
            traj = traj.cpu().numpy()

            if image_type == "patch":

                traj= traj + (self.margin_in) - center.cpu().numpy()

            ax.plot((traj[:, 0]).astype(int), (traj[:, 1]).astype(int),
                    linestyle=marker, linewidth=int(3))

        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)
        fig.savefig(os.path.join(saving_directory, str(index) + "_" + str(image_type) + ".png"))
        plt.close(fig)

        #error = abs(final_pos_pixel/scale - out["gt_xy"][-1, 0])
        #error_bound =  rel_scaling*self.scaling_small
        #print("Error real and approximation: {} Threshold: {} ".format(error, error_bound))
        #assert (error < error_bound).all(), "Error above error bound"

        plt.show()


    def load_dset(self):
        pickle_path = os.path.join(self.data_dir, self.save_path)
        if os.path.isfile(pickle_path) and self.load_p:

            data = torch.load(pickle_path, map_location = 'cpu')

            self.__dict__.update(data)

            self.logger.info("data loaded to {}".format(pickle_path))
        else:
            self.load()


    def save_dset(self):
        pickle_path = os.path.join(self.data_dir, self.save_path)
        if not  os.path.isfile(pickle_path):

            data_save = {}

            os.path.join(self.data_dir, self.save_path)

            for name, value in self.__dict__.items():

                try:
                    data_save.update({name: value})
                    torch.save(data_save, pickle_path)

                except:
                    data_save.pop(name)

            self.logger.info("data saved to {}".format(pickle_path))