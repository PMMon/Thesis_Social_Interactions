import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
import socket
import visdom
import argparse

from helper.bool_flag import bool_flag

# ============================================== Description ==============================================
# Script to analyze the distribution of classified trajectories for various datasets. The proportions of
# trajectory-classes are weighted according to the following formula:
#
# ws = m1*relative_frequency(linear) + m2*relative_frequency(gradually nonlinear) + m3*relative_frequency(highly nonlinear)
#
# with:
# - m1 = 0
# - m2 = 1/2
# - m3 = 1
#
# The calculated weighted sum gives an impression of how strong the motions of pedestrians in the dataset are influenced
# by social interactions between pedestrians. As ws = 1 indicated that all trajectories are classified as highly
# nonlinear, this would imply a very strong impact of social interactions on the motion behavior of pedestrians. Contrarily,
# a value of ws = 0 would imply that the pedestrians are not at all influenced by social interactions.
# ==========================================================================================================

class rate_datasets:
    def __init__(self, args):
        self.root_path = os.path.abspath(os.path.dirname(__file__))
        self.root_path_plots = os.path.join(self.root_path, "Weighted_Sum")
        if not os.path.exists(self.root_path_plots):
            os.mkdir(self.root_path_plots)

        self.dataset_type = args.dataset_type

        self.ws_ADE_FDE = args.ws_ADE_FDE
        self.ns = args.ns
        self.gs = args.gs

        # Configs visdom
        self.visdom = args.visdom
        self.viz_port = args.viz_port
        self.viz_server = args.viz_server


    def weighted_sum(self, V0_values, sigma_values):
        """
        Creates a heatmap that shows the weighted sum ws for various values of V0 and sigma.
        :param V0_values: Numpy array with values of V0
        :param sigma_values: Numpy array with values of sigma
        """
        if self.visdom:
            viz = self.connect2visdom("EvalMetrics_ClassLoss_WeightedSum")
        else:
            filename = "WeightedSum_Plot.jpg"
            file_spec = os.path.join(self.root_path_plots, filename)

        # Initialize weighted sum array
        weighted_sum = np.zeros((len(V0_values), len(sigma_values)))
        weights_dict = {"linear": 0, "gradually_nonlinear": 1/2, "highly_nonlinear": 1}
        weighted_sum_dict = {}

        for i, V0 in enumerate(V0_values):
            weighted_sum_dict[V0] = {}
            for j, sigma in enumerate(sigma_values):
                # Iterate over files
                filenpath_extension_proportions = self.dataset_type + "simulated_V0" + str(V0) + "b" + str(sigma).replace(".", "u")
                filename_proportions = "loss_classified_traj_" + filenpath_extension_proportions + ".xlsx"

                filepath_proportions = os.path.join(self.root_path, filenpath_extension_proportions, filename_proportions)
                if not os.path.exists(os.path.join(self.root_path, filenpath_extension_proportions)):
                    print("File " + str(filepath_proportions) + " does not exist!")
                    continue

                df = pd.read_excel(filepath_proportions)

                weighted_sum_dict[V0][sigma] = 0
                weighted_sum[i,j] = 0

                total_nr = np.array(df["nr_traj"][df["group"] == "linear"].unique())[0] + np.array(df["nr_traj"][df["group"] == "gradually_nonlinear"].unique())[0] + np.array(df["nr_traj"][df["group"] == "highly_nonlinear"].unique())[0]

                # Get proportions
                for group, weight in weights_dict.items():
                    weighted_sum_dict[V0][sigma] += weight*(np.array(df["nr_traj"][df["group"] == group].unique())[0]/total_nr)
                    weighted_sum[i,j] += weight*(np.array(df["nr_traj"][df["group"] == group].unique())[0]/total_nr)

        # Plot ws
        sigma, V0 = np.meshgrid(sigma_values, V0_values)
        circles = 0.5*np.ones(weighted_sum.shape)

        fig, ax = plt.subplots(figsize=(10,7))

        plt.scatter(sigma, V0, s=circles*100, c=weighted_sum, cmap=cm.jet)
        ax.set_xlabel(r'$\sigma$')
        ax.set_ylabel(r'$V^{0}$')
        ax.set_xticks(sigma_values)
        ax.set_yticks(V0_values)
        ax.set_title('Weighted sums')
        plt.colorbar()

        # Write text next to datapoint
        for i in range(weighted_sum.shape[0]):
            for j in range(weighted_sum.shape[1]):
                ax.annotate(round(weighted_sum[i,j],2), (sigma[i,j], V0[i,j]))

        if self.visdom:
            viz.matplot(fig, win="weighted sums of datasets")
        else:
            fig.savefig(file_spec)

        plt.close(fig)

        print("Weighted sum plot for datasets created.")

        if self.ws_ADE_FDE:
            self.metric_on_ADE_FDE(V0, sigma, weighted_sum)


    def metric_on_ADE_FDE(self, V0, sigma, weighted_sum):
        """
        Projects metric with weighted sums on ADE & FDE of Vanilla LSTM model and Social LSTM model for respective datasets
        :param V0: Meshgrid V0 for all datasets
        :param sigma: Meshgrid sigma for all datasets
        :param weighted_sum: Tensor with weighted_sum values for different datasets (defined trough V0 and sigma values)
        :return: 2D plot with metric to ADE/FDE value of LSTM and Social-LSTM
        """
        if self.visdom:
            print("Visualize ADE/FDE to ws plots in visdom")
            viz = self.connect2visdom("EvalMetrics_ClassLoss_WeightedSum")
        else:
            print("Create ADE/FDE to ws plots...")

        # Specify file paths
        root_path_overall_loss = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Overall_Loss"))

        # Specify file name
        file_name_losses_lstm = "socialforce_lstm_results.xlsx"
        file_name_losses_social_lstm = "socialforce_social-lstm_results_ns_" + str(int(args.ns)) + "_gs_" + str(int(args.gs)) + ".xlsx"

        file_spec_lstm = os.path.join(root_path_overall_loss, file_name_losses_lstm)
        file_spec_social_lstm = os.path.join(root_path_overall_loss, file_name_losses_social_lstm)

        df_lstm = pd.read_excel(file_spec_lstm)
        df_social_lstm = pd.read_excel(file_spec_social_lstm)

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
        # Display arrows that visualize difference in performance
        for i in range(lstm_ADE.shape[0]):
            if lstm_ADE[i] >= 0.3:
                ax.annotate('', xy=(weights[i], lstm_ADE[i]), xycoords='data',
                                xytext=(weights[i], social_lstm_ADE[i]), textcoords='data',
                                arrowprops={'arrowstyle': '<->'})

        plt.scatter(weights, social_lstm_ADE, color='b', label='Social-LSTM')
        ax.set_xlabel('metric')
        ax.set_ylabel('ADE')
        #ax.set_xticks(sigma_values)
        #ax.set_yticks(V0_values)
        ax.set_title('ADE for different datasets projected through metric')
        plt.legend()

        if self.visdom:
            viz.matplot(fig, win="ADE - metric")
        else:
            filename = "WeightedSum_ADE.jpg"
            file_spec = os.path.join(self.root_path_plots, filename)
            fig.savefig(file_spec)

        plt.close(fig)

        # Plot for FDE
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.scatter(weights, lstm_FDE, color='r', label='LSTM')
        for i in range(lstm_FDE.shape[0]):
            # Display arrows that visualize difference in performance
            if lstm_FDE[i] >= 0.6:
                ax.annotate('', xy=(weights[i], lstm_FDE[i]), xycoords='data',
                            xytext=(weights[i], social_lstm_FDE[i]), textcoords='data',
                            arrowprops={'arrowstyle': '<->'})

        plt.scatter(weights, social_lstm_FDE, color='b', label='Social-LSTM')
        ax.set_xlabel('metric')
        ax.set_ylabel('FDE')
        # ax.set_xticks(sigma_values)
        # ax.set_yticks(V0_values)
        ax.set_title('FDE for different datasets projected through metric')
        plt.legend()
        if self.visdom:
            viz.matplot(fig, win="FDE - metric")
        else:
            filename = "WeightedSum_FDE.jpg"
            file_spec = os.path.join(self.root_path_plots, filename)
            fig.savefig(file_spec)

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
    # Get input arguments from shell
    parser = argparse.ArgumentParser("Weighted Sum")

    # Configs for Dataset
    parser.add_argument("--dataset_type", default="square", type=str, help="Specify dataset-type. Choose either square or rectangle.")

    # Configs for script-behavior
    parser.add_argument("--ws_ADE_FDE", default=False, type=bool_flag, help="Specify whether to plot ADE and FDE against weighted sum ws")

    # Configs for Model
    parser.add_argument("--ns", default=0.0, type=float, help="Specify neighborhood size")
    parser.add_argument("--gs", default=0, type=int, help="Specify grid size")

    # Visdom configurations
    parser.add_argument("--visdom", default=False, type=bool_flag, help="Specify whether to plot via visdom")
    parser.add_argument("--viz_port", default=8090, type=int, help="Specify port for visdom")
    parser.add_argument("--viz_server", default="", type=str, help="Specify server for visdom")

    # Get arguments
    args = parser.parse_args()

    V0_values = np.array([0, 1, 2, 4, 6])
    sigma_values = np.array([0.2171, 0.4343, 0.8686, 1.303, 1.7371, 2.171, 2.6058])

    rating = rate_datasets(args)
    rating.weighted_sum(V0_values, sigma_values)