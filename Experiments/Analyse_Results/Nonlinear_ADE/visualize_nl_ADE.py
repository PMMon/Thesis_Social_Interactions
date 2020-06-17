import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
import pandas as pd
import numpy as np
import socket
import visdom
import argparse

from helper.bool_flag import bool_flag

# ============================================== Description ==============================================
# Script to create a plot for the nonlinear Average displacement error of trajectory prediction models on
# synthetic datasets that are characterized by their values of V0 and sigma.
#
# Input:
# - Values for --V0 and --sigma that specify the dataset
# - Input-file: socialforce_nl_loss.xlsx -> captures nonlinear ADE for different threshold values td. File is
# created when running the script main/Experiment_Control.py while setting the flag --nl_ADE to True and defining
# the threshold value via --threshold_nl
#
# Output:
# - Plot for nonlinear ADE: dataset_name/nonlinear_ADE_V0_value_sigma_value.jpg
# ==========================================================================================================

class visualize_nl_ADE:
    """
    Creates plot to visualize nonlinear ADE
    """
    def __init__(self, args):
        self.V0 = args.V0
        self.sigma = args.sigma

        self.visdom = args.visdom

        if args.visdom:
            self.viz = self.connect2visdom(args.viz_server, args.viz_port, args.viz_env)

    def read_xlsx(self, input_file, output_directory):
        df = pd.read_excel(input_file)

        self.nl_ADEs = {}

        final_displacements = df["final_displ"].unique()

        # Check whether or not models have information about destination of each pedestrian
        for fdispl in final_displacements:
            legend = ()
            fig, ax = plt.subplots()
            models = df["model_type"][(df["V0"] == self.V0) & (df["sigma"] == self.sigma) & (df["final_displ"] == fdispl)].unique()
            for model in models:

                if model == "lstm":
                    c = np.array([0.8, 0.2, 0.6])
                elif model == "social-lstm":
                    c = np.array([0.2, 0.2, 0.8])
                else:
                    c = np.random.rand(1,3).squeeze()

                paddings = df["padding"][(df["V0"] == self.V0) & (df["sigma"] == self.sigma) & (df["final_displ"] == fdispl) & (df["model_type"] == model)].unique()

                # Check whether or not data was padded
                for padding in paddings:
                    threshold = np.sort(np.array(df["threshold"][(df["V0"] == self.V0) & (df["sigma"] == self.sigma) & (df["final_displ"] == fdispl) & (df["model_type"] == model) & (df["padding"] == padding)]), axis=-1)
                    non_linear_ADE = np.array([])
                    for element in threshold:
                        non_linear_ADE = np.append(non_linear_ADE, np.array(df["ADE_nonlinear"][(df["threshold"] == element) & (df["V0"] == self.V0) & (df["sigma"] == self.sigma) & (df["final_displ"] == fdispl) & (df["model_type"] == model) & (df["padding"] == padding)]))
                    ax.plot(threshold, non_linear_ADE, color=c, linewidth = int(2), markersize = 5)

                    legend_str = str(model)
                    if padding:
                        legend_str += " - padding"
                    if fdispl:
                        legend_str += " - fd"
                    legend += (legend_str,)

                    self.nl_ADEs[legend_str] = {}
                    self.nl_ADEs[legend_str]["error"] = non_linear_ADE
                    self.nl_ADEs[legend_str]["threshold"] = threshold

            plt.legend(legend)

            plt.title("ADE in nonlinear regions - V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
            plt.xlabel("threshold td")
            plt.ylabel("nonlinear ADE [m]")
            if self.visdom:
                self.viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_" + str(fdispl))
            else:
                plot_name = "Nonlinear_ADE_V0_" + str(self.V0) + "_sigma_" + str(self.sigma).replace(".", "u") + "_fd_" + str(fdispl) + ".jpg"
                fig.savefig(os.path.join(output_directory, plot_name))

            plt.close(fig)

        if self.visdom:
            print("Plot nonlinear ADE for V0: " + str(self.V0) + " and sigma: " + str(self.sigma) + " in visdom...")
        else:
            print("Save plot of nonlinear ADE for V0: " + str(self.V0) + " and sigma: " + str(self.sigma) + "...")

    def create_diff(self, output_directory):
        """
        Plot difference in the nonlinear ADE of the Social LSTM model and the Vanilla LSTM model for different threshold values
        """
        if "social-lstm - padding" in self.nl_ADEs.keys() and "lstm - padding" in self.nl_ADEs.keys():
            diff_non_fd = self.nl_ADEs["lstm - padding"]["error"] - self.nl_ADEs["social-lstm - padding"]["error"]
            fig, ax = plt.subplots()
            ax.plot(self.nl_ADEs["lstm - padding"]["threshold"], diff_non_fd, color="black", linewidth=int(2), markersize=5)
            plt.legend(("LSTM loss - SLSTM loss",))
            plt.title("ADE difference in nonlinear regions - V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
            plt.xlabel("threshold t")
            plt.ylabel("nl ADE [m]")
            if self.visdom:
                self.viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_difference")
            else:
                plot_name = "diffplot_V0_" + str(self.V0) + "_sigma_" + str(self.sigma).replace(".", "u") + "_padding.jpg"
                fig.savefig(os.path.join(root_path_diffplot, plot_name))
            plt.close(fig)

        if "social-lstm - padding - fd" in self.nl_ADEs.keys() and "lstm - padding - fd" in self.nl_ADEs.keys():
            diff_non_fd = self.nl_ADEs["lstm - padding - fd"]["error"] - self.nl_ADEs["social-lstm - padding - fd"]["error"]
            fig, ax = plt.subplots()
            ax.plot(self.nl_ADEs["lstm - padding - fd"]["threshold"], diff_non_fd, color="black", linewidth=int(2), markersize=5)
            plt.legend(("LSTM loss fd - SLSTM loss fd",))
            plt.title("Fd ADE difference in nonlinear regions - V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
            plt.xlabel("threshold t")
            plt.ylabel("nl ADE [m]")
            if self.visdom:
                self.viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_fd_difference")
            else:
                plot_name = "diffplot_V0_" + str(self.V0) + "_sigma_" + str(self.sigma).replace(".", "u") + "_fd.jpg"
                fig.savefig(os.path.join(root_path_diffplot, plot_name))
            plt.close(fig)


    def connect2visdom(self, viz_server, viz_port, viz_env):
        """
        Connect to Visdom server
        """
        servername = socket.gethostname()
        if "node" in servername:
            self.server = viz_server
        else:
            self.server = 'http://localhost'

        return visdom.Visdom(server=self.server, port=viz_port, env=str(viz_env))


if __name__ == "__main__":
    # Get input arguments from shell
    parser = argparse.ArgumentParser("Visualize ADE in nonlinear regions")

    # Dataset configurations
    parser.add_argument("--V0", default=-1, type=int, help="Specify V0 of dataset")
    parser.add_argument("--sigma", default=-1, type=float, help="Specify sigma of dataset")
    parser.add_argument("--dataset_type", default="square", type=str, help="Specify dataset-type")

    # General configurations
    parser.add_argument("--diff_plot", default=False, type=bool_flag, help="Specify whether to plot difference-plot")

    # Visdom configurations
    parser.add_argument("--visdom", default=False, type=bool_flag, help="Specify whether to plot via visdom")
    parser.add_argument("--viz_port", default=8090, type=int, help="Specify port for visdom")
    parser.add_argument("--viz_server", default="", type=str, help="Specify server for visdom")
    parser.add_argument("--viz_env", default="EvalMetrics_NonlinearADE_plot", type=str, help="Specify environment for visdom")

    # Get arguments
    args = parser.parse_args()

    if args.V0 == -1 or args.sigma == -1.0:
        raise ValueError("please insert valid V0 and sigma for dataset")

    # Get name of synthetic dataset
    parameterinfo = "V0" + str(args.V0).replace(".", "u") + "b" + str(args.sigma).replace(".", "u")
    dataset_name = args.dataset_type + "simulated_" + parameterinfo

    # Specify file paths
    root_path = os.path.abspath(os.path.dirname(__file__))
    root_path_losses = root_path
    loss_file = os.path.join(root_path_losses, "socialforce_nl_loss.xlsx")
    root_path_plot = os.path.join(root_path, dataset_name)
    root_path_diffplot = os.path.join(root_path_plot, "Diff_Plots")

    if not os.path.exists(root_path_plot):
        os.makedirs(root_path_plot)

    visualizing = visualize_nl_ADE(args)
    visualizing.read_xlsx(input_file=loss_file, output_directory=root_path_plot)

    # Create difference-plot
    if args.diff_plot:
        visualizing.create_diff(root_path_diffplot)
