import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helper.bool_flag import bool_flag
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import math

import pandas as pd
import numpy as np
import socket
import visdom
import argparse

class nl_ADE:
    def __init__(self, args):
        if args.model_type.lower() not in ["linear", "lstm", "social-lstm"]:
            raise ValueError("Please choose model_type:\n- linear\n- lstm\n- social-lstm")

        self.model_type = args.model_type
        self.ns = args.ns
        self.gs = args.gs
        self.padding = args.padding
        self.fd = args.fd

        self.V0 = args.V0
        self.sigma = args.sigma

        self.viz_port = args.viz_port
        self.viz_server = args.viz_server

    def read_xlsx(self):
        dataset_name = "squaresimulated_V0" + str(self.V0) + "b" + str(self.sigma).replace(".", "u")
        file_name = "nl_traj_ADE_" + self.model_type
        if self.padding:
            file_name += "_padding"
        if self.fd:
            file_name += "_final_dp"
        if self.model_type == "social-lstm":
            file_name += "_ns_" + str(int(self.ns)) + "_gs_" + str(int(self.ns))

        file_name += "_" + dataset_name + ".xlsx"

        filepath = "Analyse_Results/Nonlinear_loss/Classified_Trajectories/Fine/" + dataset_name + "/" + file_name

        print("Read: " + filepath + " ...")

        if not os.path.exists(filepath):
            raise ValueError("File does not exist!")

        df = pd.read_excel(filepath)

        return df

    def plot_distribution(self):
        viz = self.connect2visdom("Nonlinear_distr")

        df = self.read_xlsx()
        N = np.sort(np.array((df["N"].unique())), axis=-1)
        k = np.sort(np.array((df["k"].unique())), axis=-1)


        major_ticks = np.arange(0, N.max() + 0.1, 1)
        minor_ticks = np.arange(0, N.max() + 0.1, 0.5)

        width = 0.7 * (N[1] - N[0])
        for k_value in k:
            occurence_traj = np.zeros((len(N)))
            for i, N_value in enumerate(N):
                occurence_traj[i] = df["nr_traj"][(df["N"] == N_value) & (df["k"] == k_value)]

            fig, ax = plt.subplots()

            plt.bar(N, occurence_traj, align='center', width=width)
            text_targets = "k: " + str(k_value) + "\nTotal: " + str(int(occurence_traj.sum()))
            #text_targets = "Max value: " + str(max.round(2)) + "\nMin Value: " + str(min.round(2)) + "\nMean: " + str(mean.round(2)) + "\nstd_deviation: " + str(std_dev.round(2))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.text(0.85, 0.95, text_targets, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

            plt.title("Occurences of classified trajectories for dataset V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
            plt.xlabel("Nr. points of curvature with curvature >= k")
            plt.ylabel("Occurrence")
            viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_" + str(k_value))
            plt.close(fig)

    def calculate_max_N(self, k, N, df, percent=0.95):
        """
         Calculate value for N for which 95% of all trajectories for given threshold k are given
        :param k: threshold curvature
        :param N: Number of points of trajectory that have curvature >= k
        :param df: dataframe
        :return: max_N_value
        """
        occurence_traj = np.zeros((len(N)))
        for i, N_value in enumerate(N):
            occurence_traj[i] = df["nr_traj"][(df["N"] == N_value) & (df["k"] == k)]

        return N[(occurence_traj.cumsum()/occurence_traj.sum())>=percent][0]


    def create_CDF_plot(self):
        viz = self.connect2visdom("Nonlinear_CDF")

        df = self.read_xlsx()

        N = np.sort(np.array((df["N"].unique())), axis=-1)
        k = np.sort(np.array((df["k"].unique())), axis=-1)

        major_ticks = np.arange(0, N.max() + 0.1, 1)
        minor_ticks = np.arange(0, N.max() + 0.1, 0.5)

        width = 0.7 * (N[1] - N[0])
        for k_value in k:
            occurence_traj = np.zeros((len(N)))
            for i, N_value in enumerate(N):
                occurence_traj[i] = df["nr_traj"][(df["N"] == N_value) & (df["k"] == k_value)]

            # Create CDF for targets
            fig_CDF, ax_CDF = plt.subplots()
            #ax_CDF.hist(occurence_traj,  bins=N,  normed=True, histtype='step', cumulative=True, label="k: " + str(k_value))
            ax_CDF.step(N, (occurence_traj.cumsum()/occurence_traj.sum()), label="k: " + str(k_value))
            max_valid_N = N[(occurence_traj.cumsum()/occurence_traj.sum())>=0.95][0]
            ax_CDF.axvline(ymin=0.0, ymax=0.95, x=max_valid_N,  color='r', label='95% of whole data')

            ax_CDF.set_xticks(major_ticks)
            ax_CDF.set_xticks(minor_ticks, minor=True)
            ax_CDF.set_yticks(np.arange(0, 1.1, 0.2))
            ax_CDF.set_yticks(np.arange(0, 1.1, 0.1), minor=True)
            plt.title("CDF of Nonlinear Traj. Distr. for V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
            plt.xlabel("N")
            plt.ylabel("Likelihood of occurrence")
            ax_CDF.grid(True)
            ax_CDF.legend(loc='upper left')
            viz.matplot(fig_CDF, win="CDF_all" + str(self.V0) + str(self.sigma) + str(k_value))
            plt.close(fig_CDF)

    def plot_ADE_loss_3d(self):
        viz = self.connect2visdom("Nonlinear_ADE3D")

        df = self.read_xlsx()

        nl_ADE = np.zeros((len(df["k"].unique()), len(df["N"].unique())))

        k = np.sort(np.array([(df["k"].unique())]), axis=-1)
        N = np.sort(np.array([(df["N"].unique())]), axis=-1)
        k = np.repeat(k, N.shape[-1], axis=0).transpose()
        N = np.repeat(N, k.shape[0], axis=0)

        for i in range(k.shape[0]):
            max_N = self.calculate_max_N(k[i,0], N[i], df, 0.95)
            for j in range(max_N+1):
                if math.isnan(df["ADE_nonlinear"][(df["N"] == N[i, j]) & (df["k"] == k[i, j])]):
                    nl_ADE[i, j] = nl_ADE[i, j-1]
                else:
                    nl_ADE[i, j] = df["ADE_nonlinear"][(df["N"] == N[i, j]) & (df["k"] == k[i, j])]

        nl_ADE_flat = nl_ADE.flatten()
        N_flat = N.flatten()
        k_flat = k.flatten()
        N_flat = N_flat[nl_ADE_flat != 0.0]
        k_flat = k_flat[nl_ADE_flat != 0.0]
        nl_ADE_flat = nl_ADE_flat[nl_ADE_flat != 0.0]


        fig = plt.figure(figsize=(10,7))
        ax = fig.gca(projection='3d')
        ax.view_init(30,225)
        # Plot the surface.
        #surf = ax.plot_surface(N, k, nl_ADE, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
        scat = ax.scatter(N_flat, k_flat, c = nl_ADE_flat, marker='o', s=35, cmap="autumn_r")
        #trisurf = ax.plot_wireframe(N_flat, k_flat, nl_ADE_flat)


        # Customize the z axis.
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(scat, ax=ax)

        ax.set_title(self.model_type + " - ADE of nonlinear Trajectories - V0: " + str(self.V0) + " - sigma: " + str(self.sigma), y=1.08)
        ax.set_xlabel("N")
        ax.set_ylabel("threshold k")
        ax.set_zlabel("ADE [m]")
        ax.set_zlim([0,0.8])
        viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_" + str(self.model_type) + "_" + str(self.padding) + "_" + str(self.fd))
        plt.close(fig)

    def plot_ADE_loss_2d(self):
        viz = self.connect2visdom("Nonlinear_ADE2D")

        df = self.read_xlsx()

        N = np.sort(np.array((df["N"].unique())), axis=-1)
        k = np.sort(np.array((df["k"].unique())), axis=-1)

        width = 0.7 * (N[1] - N[0])

        tot_nr_traj = np.zeros((len(N)))
        for i, N_value in enumerate(N):
            tot_nr_traj[i] = df["nr_traj"][(df["N"] == N_value) & (df["k"] == k[0])]

        tot_nr_traj = tot_nr_traj.sum()

        for N_value in N:
            x_values = np.array([])
            ADE_loss = np.array([])
            nr_traj = np.array([])
            for k_value in k:
                amount = np.array(df["nr_traj"][(df["N"] == N_value) & (df["k"] == k_value)])[0]
                max_N = self.calculate_max_N(k_value, N, df, 0.95)
                if 0.03*tot_nr_traj > amount:
                    continue
                if max_N < 4:
                    continue
                x_values = np.append(x_values, k_value)
                ADE_loss = np.append(ADE_loss, df["ADE_nonlinear"][(df["N"] == N_value) & (df["k"] == k_value)])
                nr_traj = np.append(nr_traj, df["nr_traj"][(df["N"] == N_value) & (df["k"] == k_value)])

            # Create Plot for ADE
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(x_values, ADE_loss, 'o-', color='r', label="N: " + str(N_value))
            for i in range(len(x_values)):
                ax.annotate(str("%d" % (nr_traj[i])), xy=(x_values[i], ADE_loss[i]))


            major_ticks = np.arange(0, k.max() + 0.1, 0.1)
            minor_ticks = np.arange(0, k.max() + 0.1, 0.05)

            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            plt.title(self.model_type + " - ADE of nonlinear Trajectories - V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
            plt.xlabel("k")
            plt.ylabel("ADE [m]")
            plt.ylim([0.1, 0.6])
            ax.grid(True)
            ax.legend(loc='upper right')
            viz.matplot(fig, win="CDF_all" + str(self.V0) + str(self.sigma) + str(N_value) + str(self.model_type))
            plt.close(fig)

        """
                for k_value in k:
            max_N = self.calculate_max_N(k_value, N, df, 0.95)
            ADE_loss = np.zeros(max_N+1)

            for i, N_value in enumerate(np.arange(0,max_N+1)):
                ADE_loss[i] = df["ADE_nonlinear"][(df["N"] == N_value) & (df["k"] == k_value)]

            # Create Plot for ADE
            fig, ax = plt.subplots()
            ax.plot(np.arange(0,max_N+1), ADE_loss, 'o-', color='r', label="k: " + str(k_value))

            major_ticks = np.arange(0, max_N+1 + 0.1, 1)
            minor_ticks = np.arange(0, max_N+1 + 0.1, 0.5)

            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            plt.title("Nonlinear ADE for dataset V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
            plt.xlabel("N")
            plt.ylabel("ADE [m]")
            ax.grid(True)
            ax.legend(loc='upper right')
            viz.matplot(fig, win="CDF_all" + str(self.V0) + str(self.sigma) + str(k_value))
            plt.close(fig)
        """

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

    # Show plot via Visdom module
    servername = socket.gethostname()
    if "node" in servername:
        server = "http://atcremers10"
    else:
        server = 'http://localhost'

    viz = visdom.Visdom(server=server, port=8097, env="Nonlinear_ADE")

    # Get input arguments from shell
    parser = argparse.ArgumentParser("Visualize ADE in nonlinear regions")

    # Configs about plot_type and Dataset
    parser.add_argument("--ADE", default=False, type=bool_flag, help="specify whether to plot nonlinear ADE")
    parser.add_argument("--distr", default=False, type=bool_flag, help="specify whether to plot distribution of trajectories with nonlinearities")
    parser.add_argument("--V0", default=-1, type=int, help="Specify V0 of dataset")
    parser.add_argument("--sigma", default=-1, type=float, help="Specify sigma of dataset")

    parser.add_argument("--padding", default=False, type=bool_flag, help="Specify if padding should be active")
    parser.add_argument("--fd", default=False, type=bool_flag, help="Specify if final position should be passed to model or not")

    # Configs about Model
    parser.add_argument("--model_type", default="", type=str,help="Define model type. Either Linear or LSTM Model")
    parser.add_argument("--ns", default=0.0, type=float, help="specify neighborhood_size to one side")
    parser.add_argument("--gs", default=0, type=int, help="specify grid_size")

    # Settings about connection to Visdom server
    parser.add_argument("--viz_port", default=8098, type=int, help="specify port for visdom")
    parser.add_argument("--viz_server", default="http://atcremers10", type=str, help="specify server for visdom")

    # Get arguments
    args = parser.parse_args()

    if args.V0 == -1 or args.sigma == -1.0:
        raise ValueError("please insert valid V0 and sigma for dataset")

    eval_nl_ADE = nl_ADE(args)
    #eval_nl_ADE.plot_ADE_loss_3d()
    eval_nl_ADE.plot_ADE_loss_2d()
    #eval_nl_ADE.plot_distribution()
    #eval_nl_ADE.create_CDF_plot()