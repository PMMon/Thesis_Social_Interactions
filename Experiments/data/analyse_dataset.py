import numpy as np
import os
import pandas as pd

class analyse_dataset:
    """
    Class that analyzes datasets of human motion behavior with respect to the number of suitable trajectories for the prediction task.
    A Trajectory is considered as suitable if it comprises coordinates for at least obs_len + pred_len (= seq_len) consecutive time steps.
    """
    def __init__(self, dataset_name, data, seq_len):
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.path =  os.path.join(root_path, "Analyse_Datasets", "Real_datasets")
        self.filename = str(dataset_name) + ".xlsx"

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.data = data
        self.ped_dict_all = {}
        self.ped_all = 0
        self.ped_dict_suitable = {}
        self.ped_suitable = 0
        self.seq_len = seq_len

    def get_all_trajectories(self):
        print("Analyse dataset...")

        # Get all peds in data
        peds = np.unique(self.data[:, 1]).tolist()

        for ped_id in peds:
            frame_ids = self.data[ped_id == self.data[:, 1], 0]

            # Get number of frames when the pedestrian enters and leaves the scene
            start_frame = min(frame_ids)
            end_frame = max(frame_ids) + 1

            # Consider cases when pedestrians leave the scene and re-enter it after a period of time
            skip_frame = frame_ids[1:] - frame_ids[:-1] != 1
            indicees_skip = np.where(skip_frame == True)[0] + start_frame

            intervals = np.concatenate((np.array([start_frame - 2]), indicees_skip, np.array([end_frame])), axis=0)

            self.ped_dict_all[ped_id] = 0
            self.ped_dict_suitable[ped_id] = 0

            for i in range(len(intervals) - 1):
                self.ped_dict_all[ped_id] += np.ceil((intervals[i+1] - (intervals[i] + 2))/self.seq_len)
                self.ped_all += np.ceil((intervals[i+1] - (intervals[i] + 2))/self.seq_len)
                self.ped_dict_suitable[ped_id] += np.floor((intervals[i+1] - (intervals[i] + 2))/self.seq_len)
                self.ped_suitable += np.floor((intervals[i+1] - (intervals[i] + 2))/self.seq_len)

        self.write_results_to_xlsx()

    def write_results_to_xlsx(self):
        """
        Write results to .xlsx file and store it under Analyse_Datasets/Real_datasets/dataset_name.xlsx
        """
        first = True

        for ped_id, number_all in self.ped_dict_all.items():
            if first:
                df = pd.DataFrame({"ped_id": ped_id, "Nr traj all": number_all, "Nr traj suitable": self.ped_dict_suitable[ped_id], "total number all": self.ped_all, "total number suitable": self.ped_suitable}, index=[0])
                first = False
            else:
                df = df.append(pd.DataFrame({"ped_id": ped_id, "Nr traj all": number_all, "Nr traj suitable": self.ped_dict_suitable[ped_id]}, index=[0]), ignore_index=True, sort=True)

        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

        writer2 = pd.ExcelWriter(self.path + self.filename, engine="xlsxwriter")
        df.to_excel(writer2)
        writer2.save()