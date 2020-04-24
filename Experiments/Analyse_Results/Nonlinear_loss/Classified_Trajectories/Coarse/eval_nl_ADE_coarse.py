import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helper.bool_flag import bool_flag
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
import cv2
import os
import math

import pandas as pd
import numpy as np
import socket
import visdom
import argparse
import matplotlib.image as mpimg

class nl_ADE:
    def __init__(self, args):
        self.model_type = args.model_type
        self.ns = args.ns
        self.gs = args.gs
        self.padding = args.padding
        self.fd = args.fd

        self.socialforce = args.socialforce
        self.V0 = args.V0
        self.sigma = args.sigma

        if self.socialforce:
            self.dataset_name = "squaresimulated_V0" + str(self.V0) + "b" + str(self.sigma).replace(".", "u")
        else:
            self.dataset_name = args.dataset_name

        self.figpath = "Analyse_Results/Nonlinear_loss/Classified_Trajectories/Coarse/" + self.dataset_name + "/distribution/"
        if not os.path.exists(self.figpath):
            os.mkdir(self.figpath)

        self.without_other = args.without_other

        self.viz_port = args.viz_port
        self.viz_server = args.viz_server

    def read_xlsx(self):
        """
        Read in xlsx-file with ADE values for coarsed grouped Trajectories
        :return: Dataframe with ADE values and additional information
        """
        file_name = "nl_traj_ADE_coarse_" + self.dataset_name + ".xlsx"

        filepath = "Analyse_Results/Nonlinear_loss/Classified_Trajectories/Coarse/" + self.dataset_name + "/" + file_name

        print("Read: " + filepath + " ...")

        if not os.path.exists(filepath):
            raise ValueError("File does not exist!")

        df = pd.read_excel(filepath)

        return df

    def plot_distribution_PDF(self):
        """
        Plot distribution of classified trajectories
        """

        file = self.figpath +  self.dataset_name + ".png"

        viz = self.connect2visdom("Nonlinear_coarse_PDF")

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

        # in order to get percentages
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
        viz.matplot(fig, win=self.dataset_name)
        fig.savefig(file)
        plt.close(fig)


    def plot_ADE_loss_2d(self):
        """
        Plot ADE bar plots for each model and each classified Trajectory-group
        """

        viz = self.connect2visdom("Nonlinear_coarse_ADE2D")

        df = self.read_xlsx()
        groups = np.array(["gradually_nonlinear", "highly_nonlinear", "strictly_linear", "linear"])
        final_displ = np.array(df["final_displ"].unique())


        path = "Analyse_Results/Nonlinear_loss/Classified_Trajectories/Coarse/squaresimulated_V06b2u6058/Trajectory_Repr/"
        subplotlist = ["traj", "ADE", "empty", "traj", "ADE"]

        print("Plot ADE for different trajectory-classes...")

        for fd in final_displ:
            model_types = np.array(df["model_type"][(df["final_displ"] == fd)].unique())
            ns_list = np.array(df["ns"][(df["final_displ"] == fd)].unique())
            gs_list = np.array(df["gs"][(df["final_displ"] == fd)].unique())
            all_ADE = np.array(df["ADE_nonlinear"][(df["nr_traj"] != 0) & (df["final_displ"] == fd)].unique())
            max_ADE = np.max(all_ADE)

            if fd:
                fd_text = " - with final displacement information "
                fig_name = self.figpath + "ADE_plot_" + self.dataset_name + "_with_fd.jpg"
            else:
                fd_text = ""
                fig_name = self.figpath + "ADE_plot_" + self.dataset_name + ".jpg"

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
                    filepath = path + groups[i] + ".PNG"
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
                    raise ValueError("invalid input for subplotlist!")

            viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_" + fd_text)
            plt.close(fig)
            fig.savefig(fig_name)
            print("plot " + fd_text + "created.")


    def plot_FDE_loss_2d(self):
        """
        Plot ADE bar plots for each model and each classified Trajectory-group
        """

        viz = self.connect2visdom("Nonlinear_coarse_FDE2D")

        df = self.read_xlsx()
        groups = np.array(["gradually_nonlinear", "highly_nonlinear", "strictly_linear", "linear"])
        final_displ = np.array(df["final_displ"].unique())

        path = "Analyse_Results/Nonlinear_loss/Classified_Trajectories/Coarse/squaresimulated_V06b2u6058/Trajectory_Repr/"
        subplotlist = ["traj", "FDE", "empty", "traj", "FDE"]

        print("Plot ADE for different trajectory-classes...")

        for fd in final_displ:

            model_types = np.array(df["model_type"][(df["final_displ"] == fd)].unique())
            ns_list = np.array(df["ns"][(df["final_displ"] == fd)].unique())
            gs_list = np.array(df["gs"][(df["final_displ"] == fd)].unique())
            all_FDE = np.array(df["FDE_nonlinear"][(df["nr_traj"] != 0) & (df["final_displ"] == fd)].unique())
            max_FDE = np.max(all_FDE)

            if fd:
                fd_text = " - with final displacement information "
                fig_name = self.figpath + "FDE_plot_" + self.dataset_name + "_with_fd.jpg"
            else:
                fd_text = ""
                fig_name = self.figpath + "FDE_plot_" + self.dataset_name + ".jpg"

            fig, axs = plt.subplots(figsize=(19, 12), nrows=2, ncols=5, gridspec_kw={'width_ratios': [0.7, 1, 0.7, 0.7, 1]})
            fig.subplots_adjust(wspace=0.25, hspace=0.30, top=0.85, bottom=0.05)

            fig.text(0.55, 0.96, 'Final Displacement Error for categorized Trajectories - V0: ' + str(self.V0) + ' - sigma: ' + str(self.sigma) + fd_text,
                     horizontalalignment='center', color='black', weight='bold',
                     fontsize=14)

            i = 0
            k = 0


            print("Plot FDE for different trajectory-classes...")

            for ax in axs.flat:
                k = k%5

                if subplotlist[k] == "traj":
                    filepath = path + groups[i] + ".PNG"
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
                    raise ValueError("invalid input for subplotlist!")

            viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_" + fd_text)
            plt.close(fig)
            fig.savefig(fig_name)
            print("plot " + fd_text + "created.")


    def connect2visdom(self, env):
        """
        Connect to Visdom server
        """
        servername = socket.gethostname()
        if "node" in servername:
            self.server = self.viz_server
        else:
            self.server = 'http://localhost'

        return visdom.Visdom(server=self.server, port=self.viz_port, env=env)



class rate_datasets:
    def __init__(self, args):
        self.path =  "Analyse_Results/Nonlinear_loss/Classified_Trajectories/Coarse/"
        self.viz_port = args.viz_port
        self.viz_server = args.viz_server

    def weighted_sum(self):
        """
        Read in distribution of classified datasets for different values of V0 and sigma. Then rate dataset according to the following metric:
        ========
        weighted sum = m1*relative_frequency(strictly linear) + m2*relative_frequency(linear) + m3*relative_frequency(nonlinear) + m4*relative_frequency(highly nonlinear)
        second approach: m1*relative_frequency(linear) + m2*relative_frequency(nonlinear) + m3*relative_frequency(highly nonlinear)
        weighted sum = m1
        ========
        with:
        - m1 = 0
        - m2 = 1/3
        - m3 = 2/3
        - m4 = 3/3

        :return: Heatmap plot of weighted sum against V0 & sigma values
        """
        viz = self.connect2visdom("Nonlinear_coarse_weighted sum")

        V0_values = np.array([0, 1, 2, 4, 6])
        sigma_values = np.array([0.2171, 0.4343, 0.8686, 1.303, 1.7371, 2.171, 2.6058])
        weighted_sum = np.zeros((len(V0_values), len(sigma_values)))

        #weights_dict = {"strictly_linear": 0, "linear": 1/3, "med_nonlinear": 2/3, "highly_nonlinear": 1}
        weights_dict = {"linear": 0, "med_nonlinear": 1/2, "highly_nonlinear": 1}
        weighted_sum_dict = {}


        for i, V0 in enumerate(V0_values):
            weighted_sum_dict[V0] = {}
            for j, sigma in enumerate(sigma_values):
                filenpath_extension = "squaresimulated_V0" + str(V0) + "b" + str(sigma).replace(".", "u") + "/"
                filename = "nl_traj_ADE_coarse_" + filenpath_extension[:-1] + ".xlsx"

                filepath = self.path + filenpath_extension +filename
                if not os.path.exists(filepath):
                    print("File " + str(filename) + " does not exist!")
                    continue

                df = pd.read_excel(filepath)

                weighted_sum_dict[V0][sigma] = 0
                weighted_sum[i,j] = 0

                #total_nr = np.array(df["nr_traj"][df["group"] == "strictly_linear"].unique())[0] + np.array(df["nr_traj"][df["group"] == "linear"].unique())[0] + np.array(df["nr_traj"][df["group"] == "med_nonlinear"].unique())[0] + np.array(df["nr_traj"][df["group"] == "highly_nonlinear"].unique())[0]
                total_nr = np.array(df["nr_traj"][df["group"] == "linear"].unique())[0] + np.array(df["nr_traj"][df["group"] == "med_nonlinear"].unique())[0] + np.array(df["nr_traj"][df["group"] == "highly_nonlinear"].unique())[0]

                for group, weight in weights_dict.items():
                    weighted_sum_dict[V0][sigma] += weight*(np.array(df["nr_traj"][df["group"] == group].unique())[0]/total_nr)
                    weighted_sum[i,j] += weight*(np.array(df["nr_traj"][df["group"] == group].unique())[0]/total_nr)

        sigma, V0 = np.meshgrid(sigma_values, V0_values)
        circles = 0.5*np.ones(weighted_sum.shape)

        fig, ax = plt.subplots(figsize=(10,7))

        plt.scatter(sigma, V0, s=circles*100, c=weighted_sum, cmap=cm.jet) #c=weighted_sum
        ax.set_xlabel('sigma')
        ax.set_ylabel('V0')
        ax.set_xticks(sigma_values)
        ax.set_yticks(V0_values)
        ax.set_title('Metric for Nonlinearity of Datasets')
        plt.colorbar()

        # Write text next to datapoint
        for i in range(weighted_sum.shape[0]):
            for j in range(weighted_sum.shape[1]):
                ax.annotate(round(weighted_sum[i,j],2), (sigma[i,j], V0[i,j]))

        viz.matplot(fig, win="weighted sums of datasets")
        plt.close(fig)

        print("weighted sum plot for datasets created.")

        self.metric_on_ADE_FDE(V0, sigma, weighted_sum)

    def metric_on_ADE_FDE(self, V0, sigma, weighted_sum):
        """
        Projects metric with weighted sums on ADE & FDE of LSTM and social-LSTM for respective datasets
        :param V0: Meshgrid V0 for all datasets
        :param sigma: Meshgrid sigma for all datasets
        :param weighted_sum: Tensor with weighted_sum values for different datasets (defined trough V0 and sigma values)
        :return: 2D plot with metric to ADE/FDE value of LSTM and Social-LSTM
        """
        viz = self.connect2visdom("Nonlinear_coarse_weighted sum")

        path = self.path + "overall_Results/"

        df_lstm = pd.read_excel(path+"socialforce_lstm_results.xlsx")
        df_social_lstm = pd.read_excel(path+"socialforce_social-lstm_results_ns_10_gs_10.xlsx")

        weights = np.array([])
        lstm_ADE = np.array([])
        lstm_FDE = np.array([])
        social_lstm_ADE = np.array([])
        social_lstm_FDE = np.array([])

        for i in range(weighted_sum.shape[0]):
            for j in range(weighted_sum.shape[1]):
                weights = np.append(weights, weighted_sum[i,j])

                # LSTM values
                lstm_ADE = np.append(lstm_ADE, np.array(df_lstm["ADE"][(df_lstm["V0"] == V0[i,j]) & (df_lstm["sigma"] == sigma[i,j])]))
                lstm_FDE = np.append(lstm_FDE, np.array(df_lstm["FDE"][(df_lstm["V0"] == V0[i,j]) & (df_lstm["sigma"] == sigma[i,j])]))

                # Social LSTM values
                social_lstm_ADE = np.append(social_lstm_ADE, np.array(df_social_lstm["ADE"][(df_social_lstm["V0"] == V0[i, j]) & (df_social_lstm["sigma"] == sigma[i,j])]))
                social_lstm_FDE = np.append(social_lstm_FDE, np.array(df_social_lstm["FDE"][(df_social_lstm["V0"] == V0[i, j]) & (df_social_lstm["sigma"] == sigma[i,j])]))

        # Plot for ADE
        fig, ax = plt.subplots(figsize=(10, 7))

        plt.scatter(weights, lstm_ADE, color='r', label='LSTM')
        # Write text next to datapoint for high values
        for i in range(lstm_ADE.shape[0]):
            if lstm_ADE[i] >= 0.3:
                ax.annotate('', xy=(weights[i], lstm_ADE[i]), xycoords='data',
                                xytext=(weights[i], social_lstm_ADE[i]), textcoords='data',
                                arrowprops={'arrowstyle': '<->'})
                #ax.annotate(round(lstm_ADE[i], 2), (weights[i], lstm_ADE[i]))

        plt.scatter(weights, social_lstm_ADE, color='b', label='Social-LSTM')
        # Write text next to datapoint
        #for i in range(social_lstm_ADE.shape[0]):
            #if social_lstm_ADE[i] >= 0.26:
            #    ax.annotate(round(social_lstm_ADE[i], 2), (weights[i], social_lstm_ADE[i]))

        ax.set_xlabel('metric')
        ax.set_ylabel('ADE')
        #ax.set_xticks(sigma_values)
        #ax.set_yticks(V0_values)
        ax.set_title('ADE for different datasets projected through metric')
        plt.legend()
        #plt.grid()
        viz.matplot(fig, win="ADE - metric")
        plt.close(fig)

        # Plot for FDE
        fig, ax = plt.subplots(figsize=(10, 7))

        plt.scatter(weights, lstm_FDE, color='r', label='LSTM')
        for i in range(lstm_FDE.shape[0]):
            if lstm_FDE[i] >= 0.6:
                ax.annotate('', xy=(weights[i], lstm_FDE[i]), xycoords='data',
                            xytext=(weights[i], social_lstm_FDE[i]), textcoords='data',
                            arrowprops={'arrowstyle': '<->'})
        # Write text next to datapoint
        #for i in range(lstm_ADE.shape[0]):
        #    ax.annotate(round(lstm_FDE[i], 2), (weights[i], lstm_FDE[i]))
        plt.scatter(weights, social_lstm_FDE, color='b', label='Social-LSTM')
        ax.set_xlabel('metric')
        ax.set_ylabel('FDE')
        # ax.set_xticks(sigma_values)
        # ax.set_yticks(V0_values)
        ax.set_title('FDE for different datasets projected through metric')
        plt.legend()
        viz.matplot(fig, win="FDE - metric")
        plt.close(fig)

        print("Projection metric to ADE & FDE loss created.")


    def connect2visdom(self, env):
        """
        Connect to Visdom server
        """
        servername = socket.gethostname()
        if "node" in servername:
            self.server = self.viz_server
        else:
            self.server = 'http://localhost'

        return visdom.Visdom(server=self.server, port=self.viz_port, env=env)



if __name__ == "__main__":
    import os

    # Get input arguments from shell
    parser = argparse.ArgumentParser("Visualize ADE in nonlinear regions")

    # Configs about plot_type and Dataset
    parser.add_argument("--socialforce", default=True, type=bool_flag, help="Specify whether dataset of social force model or not")
    parser.add_argument("--dataset_name", default="", type=str, help="Specify datasetname if dataset not created by social force model")
    parser.add_argument("--ADE", default=False, type=bool_flag, help="specify whether to plot nonlinear ADE")
    parser.add_argument("--distr", default=False, type=bool_flag, help="specify whether to plot distribution of trajectories with nonlinearities")
    parser.add_argument("--V0", default=-1, type=int, help="Specify V0 of dataset")
    parser.add_argument("--sigma", default=-1, type=float, help="Specify sigma of dataset")

    # Special configs on plot
    parser.add_argument("--without_other", default=False, type=bool_flag, help="specify whether to plot distribution without other type")

    # Configs about Model
    parser.add_argument("--model_type", default="", type=str,help="Define model type. Either Linear or LSTM Model")
    parser.add_argument("--ns", default=0.0, type=float, help="specify neighborhood_size to one side")
    parser.add_argument("--gs", default=0, type=int, help="specify grid_size")

    parser.add_argument("--padding", default=False, type=bool_flag, help="Specify if padding should be active")
    parser.add_argument("--fd", default=False, type=bool_flag, help="Specify if final position should be passed to model or not")

    # Settings about connection to Visdom server
    parser.add_argument("--viz_port", default=8098, type=int, help="specify port for visdom")
    parser.add_argument("--viz_server", default="http://atcremers10", type=str, help="specify server for visdom")

    # Get arguments
    args = parser.parse_args()

    if not args.socialforce:
        if args.dataset_name == "":
            raise ValueError("please enter dataset name!")

        eval_nl_ADE = nl_ADE(args)
        eval_nl_ADE.plot_distribution_PDF()

    else:
        if args.V0 == -1 or args.sigma == -1.0:
            raise ValueError("please insert valid V0 and sigma for dataset")


        eval_nl_ADE = nl_ADE(args)
        eval_nl_ADE.plot_distribution_PDF()
        eval_nl_ADE.plot_ADE_loss_2d()
        eval_nl_ADE.plot_FDE_loss_2d()


        #rating = rate_datasets(args)
        #rating.weighted_sum()