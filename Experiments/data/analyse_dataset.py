import numpy as np
import os
import pandas as pd

class analyse_dataset:
    def __init__(self, dataset_name, data):
        self.path = "Analyse_Datasets//Real_datasets//" + str(dataset_name) + "//"
        self.filename = str(dataset_name) + ".xlsx"

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.data = data
        self.ped_dict_all = {}
        self.ped_all = 0
        self.ped_dict_suitable = {}
        self.ped_suitable = 0

    def get_all_trajectories(self):
        print("analyse dataset...")
        # get all peds in data
        peds = np.unique(self.data[:, 1]).tolist()

        for ped_id in peds:

            frame_ids = self.data[ped_id == self.data[:, 1], 0]
            start_frame = min(frame_ids)
            end_frame = max(frame_ids) + 1

            skip_frame = frame_ids[1:] - frame_ids[:-1] != 1
            indicees_skip = np.where(skip_frame == True)[0] + start_frame

            intervals = np.concatenate((np.array([start_frame - 2]), indicees_skip, np.array([end_frame])), axis=0)

            self.ped_dict_all[ped_id] = 0
            self.ped_dict_suitable[ped_id] = 0

            for i in range(len(intervals) - 1):
                self.ped_dict_all[ped_id] += np.ceil((intervals[i+1] - (intervals[i] + 2))/20)
                self.ped_all += np.ceil((intervals[i+1] - (intervals[i] + 2))/20)
                self.ped_dict_suitable[ped_id] += np.floor((intervals[i+1] - (intervals[i] + 2))/20)
                self.ped_suitable += np.floor((intervals[i+1] - (intervals[i] + 2))/20)

        self.write_results_to_xlsx()

    def write_results_to_xlsx(self):
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