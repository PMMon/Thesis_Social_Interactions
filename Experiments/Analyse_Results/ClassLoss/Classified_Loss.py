import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
from os import strerror
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
import errno
import pandas as pd
import numpy as np
import socket
import visdom
import argparse

from helper.bool_flag import bool_flag

# ============================================== Description ==============================================
# Script to analyze the ADE and FDE of the Vanilla LSTM model and the Social LSTM model on trajectories that
# are classified with respect to their degree of nonlinearity -> with respect to the degree they are influenced
# by social interactions between pedestrians.
#
# Input:
# - Values for --V0 and --sigma that specify the dataset
# - Input-file: dataset_name/loss_classified_traj_dataset_name.xlsx -> captures ADE and FDE on different trajectory-classes
#
# Output:
# - Plot for distribution of classified trajectories: dataset_name/Evaluation/ClassLoss_PDF_dataset_name.jpg
# - Plot for classified ADE: dataset_name/Evaluation/ClassLoss_ADE_dataset_name.jpg
# - Plot for classified FDE: dataset_name/Evaluation/ClassLoss_FDE_dataset_name.jpg
# ==========================================================================================================

class nl_error_classified:
    """
    Class to create plots for ADE and FDE of Vanilla LSTM model and Social LSTM model on different trajectory-classes
    """
    def __init__(self, args):
        # Dataset
        self.socialforce = args.socialforce
        self.V0 = args.V0
        self.sigma = args.sigma

        if self.socialforce:
            self.dataset_name = args.dataset_type + "simulated_V0" + str(self.V0) + "b" + str(self.sigma).replace(".", "u")
        else:
            self.dataset_name = args.dataset_name

        self.root_path = os.path.abspath(os.path.dirname(__file__))
        self.root_path_losses = os.path.join(self.root_path, self.dataset_name)
        if not os.path.exists(self.root_path_losses):
            raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), str(self.root_path_losses))

        self.root_path_plots = os.path.join(self.root_path_losses, "Evaluation")
        if not os.path.exists(self.root_path_plots):
            os.mkdir(self.root_path_plots)

        self.without_other = args.without_other

        # Visdom
        self.visdom = args.visdom
        self.viz_port = args.viz_port
        self.viz_server = args.viz_server


    def read_xlsx(self):
        """
        Read in xlsx-file with ADE and FDE values of classified trajectories.
        :return: Dataframe with ADE values and additional information
        """
        file_name = "loss_classified_traj_" + self.dataset_name + ".xlsx"
        filepath = os.path.join(self.root_path_losses, file_name)

        df = pd.read_excel(filepath)

        return df


    def plot_distribution_PDF(self):
        """
        Plot distribution of classified trajectories
        """
        if self.visdom:
            print("Plot distribution of classified trajectories via visdom.")
            viz = self.connect2visdom("EvalMetrics_ClassLoss_PDF")
        else:
            print("Save plot of distribution of classified trajectories.")
            filename = "ClassLoss_PDF_" + self.dataset_name + ".jpg"
            file_spec = os.path.join(self.root_path_plots, filename)

        # Read in file
        df = self.read_xlsx()
        groups = np.array((df["group"].unique()))

        pdf = {}
        total_occ = 0

        for group in groups:
            pdf[group] =  np.array(df["nr_traj"][(df["group"] == group)])[0]
            total_occ += pdf[group]

        if self.without_other:
            total_occ -= pdf["other"]
            del pdf["other"]

        # Get percentages
        for class_name, occurences in pdf.items():
            pdf[class_name] /= total_occ
            pdf[class_name] *= 100


        labels = list(pdf.keys())
        for i in range(len(labels)):
            labels[i] = labels[i].replace("_", " \n ")

        fig, ax = plt.subplots(figsize=(10,6))
        plt.bar(labels, list(pdf.values()), align='center')
        if self.socialforce:
            plt.title("Distribution of classified trajectories for dataset V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
        else:
            plt.title("Distribution of classified trajectories for dataset: " + str(self.dataset_name))

        ax.set_ylabel('Proportion [%]')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)

        if self.visdom:
            viz.matplot(fig, win=self.dataset_name)
        else:
            fig.savefig(file_spec)

        plt.close(fig)


    def plot_ADE_loss_2d(self):
        """
        Plot ADE bar plots for each model and each classified Trajectory-group
        """
        if self.visdom:
            print("Plot Average Displacement Errors for each trajectory-class...")
            viz = self.connect2visdom("EvalMetrics_ClassLoss_ADE")
        else:
            print("Save plot of ADE for each trajectory-class...")

        # Read in loss-file
        df = self.read_xlsx()

        # Define groups
        groups = np.array(["gradually_nonlinear", "highly_nonlinear", "strictly_linear", "linear"])
        final_displ = np.array(df["final_displ"].unique())

        path_examples = os.path.join(self.root_path, "Trajectory_Repr")
        if not os.path.exists(path_examples):
            raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), str(path_examples))

        subplotlist = ["traj", "ADE", "empty", "traj", "ADE"]

        for fd in final_displ:
            model_types = np.array(df["model_type"][(df["final_displ"] == fd)].unique())
            ns_list = np.array(df["ns"][(df["final_displ"] == fd)].unique())
            gs_list = np.array(df["gs"][(df["final_displ"] == fd)].unique())
            all_ADE = np.array(df["ADE_nonlinear"][(df["nr_traj"] != 0) & (df["final_displ"] == fd)].unique())
            max_ADE = np.max(all_ADE)

            if fd:
                fd_text = " - with final displacement information "
                if not self.visdom:
                    plot_name =  "ClassLoss_ADE_" + str(self.dataset_name) + "_with_fd.jpg"
                    fig_spec = os.path.join(self.root_path_plots, plot_name)
            else:
                fd_text = ""
                if not self.visdom:
                    plot_name = "ClassLoss_ADE_" + str(self.dataset_name) + ".jpg"
                    fig_spec = os.path.join(self.root_path_plots, plot_name)

            fig, axs = plt.subplots(figsize=(19, 12), nrows=2, ncols=5,  gridspec_kw={'width_ratios': [0.7, 1, 0.7, 0.7, 1]})
            fig.subplots_adjust(wspace=0.25, hspace=0.30, top=0.85, bottom=0.05)

            fig.text(0.55, 0.96, 'Average Displacement Error for categorized Trajectories - V0: ' + str(self.V0) + ' - sigma: ' + str(self.sigma) + fd_text,
                     horizontalalignment='center', color='black', weight='bold',
                     fontsize=14)

            i = 0
            k = 0

            for ax in axs.flat:
                k = k%5

                if subplotlist[k] == "traj":
                    filepath = os.path.join(path_examples, groups[i] + ".PNG")
                    img = cv2.imread(filepath)
                    ax.imshow(img)
                    ax.axis('Off')
                    i += 1
                    k += 1

                elif subplotlist[k] == "ADE":
                    rel_occurence = round(np.array(df["nr_traj"][(df["group"] == groups[i - 1]) & (df["final_displ"] == fd)])[0]/np.array(df["total_nr_traj"][(df["group"] == groups[i - 1]) & (df["final_displ"] == fd)])[0] * 100, 2)
                    ax.set_title(str(groups[i - 1]).replace("_", " ") + " - " + str(rel_occurence) + "%", weight='bold', size='medium', position=(-0.2, 1.07), horizontalalignment='center', verticalalignment='center', fontsize = 14)
                    ax.set_ylim(0,max_ADE*1.1)

                    for model_type in model_types:
                        if np.array(df["nr_traj"][(df["group"] == groups[i - 1]) & (df["model_type"] == model_type) & (df["final_displ"] == fd)])[0] == 0:
                            continue

                        ADE_dict = {}

                        strictly_linear_ADE = np.array(df["ADE_nonlinear"][(df["model_type"] == model_type) & (df["group"] == "strictly_linear") & (df["final_displ"] == fd)])

                        if model_type == "social-lstm":
                            for ns in ns_list:
                                for gs in gs_list:
                                    #ADE_dict[str(model_type) + " - ns: " + str(ns) + " - gs: " + str(gs)] = np.array(df["ADE_nonlinear"][(df["group"] == groups[i-1]) & (df["final_displ"] == fd) & (df["model_type"] == model_type) & (df["ns"] == ns) & (df["gs"] == gs)])[0]
                                    ADE_dict[str(model_type)] = np.array(df["ADE_nonlinear"][(df["group"] == groups[i - 1]) & (df["final_displ"] == fd) & (df["model_type"] == model_type) & (df["ns"] == ns) & (df["gs"] == gs)])[0]
                                    bar = ax.bar(list(ADE_dict.keys()), list(ADE_dict.values()), align='center', label= model_type)
                                    ax.plot(list(ADE_dict.keys()), strictly_linear_ADE, 'r--')

                                    if groups[i-1] != "strictly_linear":
                                        height_bar = bar[0].get_height()
                                        percent = int(round((list(ADE_dict.values())[0] - strictly_linear_ADE[0])/ strictly_linear_ADE[0] * 100))
                                        if percent >= 0:
                                            ax.text(bar[0].get_x() + bar[0].get_width()/2.0, height_bar, r'$\Delta$ %2.f %%' % percent, ha='center', va='bottom', color="r", fontsize = 14)
                                        else:
                                            ax.text(bar[0].get_x() + bar[0].get_width() / 2.0, height_bar, r'$\Delta$ %2.f %%' % percent, ha='center', va='top', color="r", fontsize = 14)
                                        ax.axhline(strictly_linear_ADE, 0.55, 0.95, linestyle='--', color='r', linewidth=1.8, label=" strictly linear ADE")
                        else:
                            ADE_dict[str(model_type)] = np.array(df["ADE_nonlinear"][(df["model_type"] == model_type) & (df["group"] == groups[i-1]) & (df["final_displ"] == fd)])[0]
                            bar = ax.bar(list(ADE_dict.keys()), list(ADE_dict.values()), align='center', label= model_type)

                            if groups[i - 1] != "strictly_linear":
                                height_bar = bar[0].get_height()
                                percent = int(round((list(ADE_dict.values())[0] - strictly_linear_ADE[0]) / strictly_linear_ADE[0] * 100))
                                if percent >= 0:
                                    ax.text(bar[0].get_x() + bar[0].get_width() / 2.0, height_bar,r'$\Delta$ %2.f %%' % percent, ha='center', va='bottom', color="r", fontsize = 14)
                                else:
                                    ax.text(bar[0].get_x() + bar[0].get_width() / 2.0, height_bar, r'$\Delta$ %2.f %%' % percent, ha='center', va='top', color="r", fontsize = 14)
                                ax.axhline(strictly_linear_ADE, 0.05, 0.45, linestyle='--', color='r', linewidth=1.8)


                    ax.legend(loc='center left', bbox_to_anchor=(1., 0.95), prop={'size': 10})
                    for item in ([ax.xaxis.label, ax.yaxis.label] +
                                 ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(13)
                    k += 1

                elif subplotlist[k] == "empty":
                    ax.axis('Off')
                    k += 1
                    continue

                else:
                    raise ValueError("Invalid input for subplotlist!")

            if self.visdom:
                viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_" + fd_text)
            else:
                fig.savefig(fig_spec)
            plt.close(fig)
            print("Plot " + fd_text + "created.")


    def plot_FDE_loss_2d(self):
        """
        Plot FDE bar plots for each model and each classified Trajectory-group
        """
        if self.visdom:
            print("Plot Final Displacement Errors for each trajectory-class...")
            viz = self.connect2visdom("EvalMetrics_ClassLoss_FDE")
        else:
            print("Save plot of FDE for each trajectory-class...")

        # Read in losses
        df = self.read_xlsx()

        # Define grouos
        groups = np.array(["gradually_nonlinear", "highly_nonlinear", "strictly_linear", "linear"])
        final_displ = np.array(df["final_displ"].unique())

        path_examples = os.path.join(self.root_path, "Trajectory_Repr")
        if not os.path.exists(path_examples):
            raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), str(path_examples))

        subplotlist = ["traj", "FDE", "empty", "traj", "FDE"]

        for fd in final_displ:

            model_types = np.array(df["model_type"][(df["final_displ"] == fd)].unique())
            ns_list = np.array(df["ns"][(df["final_displ"] == fd)].unique())
            gs_list = np.array(df["gs"][(df["final_displ"] == fd)].unique())
            all_FDE = np.array(df["FDE_nonlinear"][(df["nr_traj"] != 0) & (df["final_displ"] == fd)].unique())
            max_FDE = np.max(all_FDE)

            if fd:
                fd_text = " - with final displacement information "
                if not self.visdom:
                    plot_name = "ClassLoss_FDE_" + str(self.dataset_name) + "_with_fd.jpg"
                    fig_spec = os.path.join(self.root_path_plots, plot_name)
            else:
                fd_text = ""
                if not self.visdom:
                    plot_name = "ClassLoss_FDE_" + str(self.dataset_name) + ".jpg"
                    fig_spec = os.path.join(self.root_path_plots, plot_name)

            fig, axs = plt.subplots(figsize=(19, 12), nrows=2, ncols=5, gridspec_kw={'width_ratios': [0.7, 1, 0.7, 0.7, 1]})
            fig.subplots_adjust(wspace=0.25, hspace=0.30, top=0.85, bottom=0.05)

            fig.text(0.55, 0.96, 'Final Displacement Error for categorized Trajectories - V0: ' + str(self.V0) + ' - sigma: ' + str(self.sigma) + fd_text,
                     horizontalalignment='center', color='black', weight='bold',
                     fontsize=14)

            i = 0
            k = 0

            for ax in axs.flat:
                k = k%5

                if subplotlist[k] == "traj":
                    filepath = os.path.join(path_examples, groups[i] + ".PNG")
                    img = cv2.imread(filepath)
                    ax.imshow(img)
                    ax.axis('Off')
                    i += 1
                    k += 1

                elif subplotlist[k] == "FDE":
                    rel_occurence = round(np.array(df["nr_traj"][(df["group"] == groups[i - 1]) & (df["final_displ"] == fd)])[0]/np.array(df["total_nr_traj"][(df["group"] == groups[i - 1]) & (df["final_displ"] == fd)])[0]*100, 2)
                    ax.set_title(str(groups[i-1].replace("_", " ")) + " - " + str(rel_occurence) + "%", weight='bold', size='medium', position=(-0.2, 1.07), horizontalalignment='center', verticalalignment='center', fontsize = 14)
                    ax.set_ylim(0,max_FDE*1.1)

                    for model_type in model_types:
                        if np.array(df["nr_traj"][(df["group"] == groups[i - 1]) & (df["model_type"] == model_type) & (df["final_displ"] == fd)])[0] == 0:
                            continue

                        FDE_dict = {}

                        strictly_linear_ADE = np.array(df["FDE_nonlinear"][(df["model_type"] == model_type) & (df["group"] == "strictly_linear") & (df["final_displ"] == fd)])

                        if model_type == "social-lstm":
                            for ns in ns_list:
                                for gs in gs_list:
                                    #FDE_dict[str(model_type) + " - ns: " + str(ns) + " - gs: " + str(gs)] = np.array(df["FDE_nonlinear"][(df["group"] == groups[i-1]) & (df["model_type"] == model_type) & (df["ns"] == ns) & (df["gs"] == gs) & (df["final_displ"] == fd)])[0]
                                    FDE_dict[str(model_type)] = np.array(df["FDE_nonlinear"][(df["group"] == groups[i-1]) & (df["model_type"] == model_type) & (df["ns"] == ns) & (df["gs"] == gs) & (df["final_displ"] == fd)])[0]
                                    bar = ax.bar(list(FDE_dict.keys()), list(FDE_dict.values()), align='center', label= model_type)
                                    ax.plot(list(FDE_dict.keys()), strictly_linear_ADE, 'r--')

                                    if groups[i-1] != "strictly_linear":
                                        height_bar = bar[0].get_height()
                                        percent = int(round((list(FDE_dict.values())[0] - strictly_linear_ADE[0])/ strictly_linear_ADE[0] * 100))
                                        if percent >= 0:
                                            ax.text(bar[0].get_x() + bar[0].get_width()/2.0, height_bar, r'$\Delta$ %2.f %%' % percent, ha='center', va='bottom', color="r", fontsize = 14)
                                        else:
                                            ax.text(bar[0].get_x() + bar[0].get_width() / 2.0, height_bar, r'$\Delta$ %2.f %%' % percent, ha='center', va='top', color="r", fontsize = 14)
                                        ax.axhline(strictly_linear_ADE, 0.55, 0.95, linestyle='--', color='r', linewidth=1.8, label="strictly linear FDE")
                        else:
                            FDE_dict[str(model_type)] = np.array(df["FDE_nonlinear"][(df["model_type"] == model_type) & (df["group"] == groups[i-1]) & (df["final_displ"] == fd)])[0]
                            bar = ax.bar(list(FDE_dict.keys()), list(FDE_dict.values()), align='center', label= model_type)

                            if groups[i - 1] != "strictly_linear":
                                height_bar = bar[0].get_height()
                                percent = int(round((list(FDE_dict.values())[0] - strictly_linear_ADE[0]) / strictly_linear_ADE[0] * 100))
                                if percent >= 0:
                                    ax.text(bar[0].get_x() + bar[0].get_width() / 2.0, height_bar,r'$\Delta$ %2.f %%' % percent, ha='center', va='bottom', color="r", fontsize = 14)
                                else:
                                    ax.text(bar[0].get_x() + bar[0].get_width() / 2.0, height_bar, r'$\Delta$ %2.f %%' % percent, ha='center', va='top', color="r", fontsize = 14)
                                ax.axhline(strictly_linear_ADE, 0.05, 0.45, linestyle='--', color='r', linewidth=1.8)


                    ax.legend(loc='center left', bbox_to_anchor=(1., 0.95),  prop={'size': 10})
                    for item in ([ax.xaxis.label, ax.yaxis.label] +
                                 ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(13)
                    k += 1

                elif subplotlist[k] == "empty":
                    ax.axis('Off')
                    k += 1
                    continue

                else:
                    raise ValueError("Invalid input for subplotlist!")

            if self.visdom:
                viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_" + fd_text)
            else:
                fig.savefig(fig_spec)
            plt.close(fig)
            print("Plot " + fd_text + "created.")


    def connect2visdom(self, viz_env):
        """
        Connect to Visdom server
        """
        servername = socket.gethostname()
        if "node" in servername:
            self.server = self.viz_server
        else:
            self.server = 'http://localhost'

        return visdom.Visdom(server=self.server, port=self.viz_port, env=viz_env)


if __name__ == "__main__":
    # Get input arguments from shell
    parser = argparse.ArgumentParser("Loss on classified trajectories")

    # Configs for Dataset
    parser.add_argument("--socialforce", default=True, type=bool_flag, help="Specify whether dataset is created by SocialForce model or not")
    parser.add_argument("--dataset_name", default="", type=str, help="Specify name of dataset if dataset was not created by SocialForce model")
    parser.add_argument("--dataset_type", default="square", type=str, help="Specify dataset-type. Choose either square, rectangle or real")
    parser.add_argument("--V0", default=-1, type=int, help="Specify V0 of dataset")
    parser.add_argument("--sigma", default=-1, type=float, help="Specify sigma of dataset")

    # Configs for Script-behavior
    parser.add_argument("--distr", default=True, type=bool_flag, help="Specify whether to plot distribution of trajectory-classes")
    parser.add_argument("--without_other", default=False, type=bool_flag, help="Specify whether to plot distribution of trajectory-classes without 'other'-type")

    # Visdom configurations
    parser.add_argument("--visdom", default=False, type=bool_flag, help="Specify whether to plot via visdom")
    parser.add_argument("--viz_port", default=8090, type=int, help="Specify port for visdom")
    parser.add_argument("--viz_server", default="", type=str, help="Specify server for visdom")

    # Get arguments
    args = parser.parse_args()

    if not args.socialforce:
        if args.dataset_name == "":
            raise ValueError("please enter dataset name!")

        eval_nl_error_classified = nl_error_classified(args)
        eval_nl_error_classified.plot_distribution_PDF()

    else:
        if args.V0 == -1 or args.sigma == -1.0:
            raise ValueError("please insert valid V0 and sigma for dataset")

        eval_nl_error_classified = nl_error_classified(args)

        if args.distr:
            eval_nl_error_classified.plot_distribution_PDF()
        eval_nl_error_classified.plot_ADE_loss_2d()
        eval_nl_error_classified.plot_FDE_loss_2d()
