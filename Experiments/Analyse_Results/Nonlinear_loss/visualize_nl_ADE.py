import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import socket
import visdom
import argparse

class visualize_nl_ADE:
    def __init__(self, V0, sigma, viz):
        self.V0 = V0
        self.sigma = sigma
        self.viz = viz

    def read_xlsx(self, filepath):
        df = pd.read_excel(filepath)

        self.nl_ADEs = {}

        final_displacements = df["final_displ"].unique()
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

                for padding in paddings:
                    threshold = np.sort(np.array(df["threshold"][(df["V0"] == self.V0) & (df["sigma"] == self.sigma) & (df["final_displ"] == fdispl) & (df["model_type"] == model) & (df["padding"] == padding)]), axis=-1)
                    non_linear_ADE = np.array([])
                    for element in threshold:
                        non_linear_ADE = np.append(non_linear_ADE, np.array(df["ADE_nonlinear"][(df["threshold"] == element) & (df["V0"] == self.V0) & (df["sigma"] == self.sigma) & (df["final_displ"] == fdispl) & (df["model_type"] == model) & (df["padding"] == padding)]))
                    ax.plot(threshold, non_linear_ADE, color=c, linewidth = int(2), markersize = 5)

                    legend_str = str(model)
                    #if padding:
                    #    legend_str += " - padding"
                    #if fdispl:
                    #    legend_str += " - fd"
                    legend += (legend_str,)

                    self.nl_ADEs[legend_str] = {}
                    self.nl_ADEs[legend_str]["error"] = non_linear_ADE
                    self.nl_ADEs[legend_str]["threshold"] = threshold

            plt.legend(legend)

            plt.title("ADE in nonlinear regions - V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
            plt.xlabel("threshold td")
            plt.ylabel("nonlinear ADE [m]")
            self.viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_" + str(fdispl))
            plt.close(fig)

        print("figure created.")

    def create_diff(self):

        if "social-lstm - padding" in self.nl_ADEs.keys() and "lstm - padding" in self.nl_ADEs.keys():
            diff_non_fd = self.nl_ADEs["lstm - padding"]["error"] - self.nl_ADEs["social-lstm - padding"]["error"]
            fig, ax = plt.subplots()
            ax.plot(self.nl_ADEs["lstm - padding"]["threshold"], diff_non_fd, color="black", linewidth=int(2), markersize=5)
            plt.legend(("LSTM loss - SLSTM loss",))
            plt.title("ADE difference in nonlinear regions - V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
            plt.xlabel("threshold t")
            plt.ylabel("nl ADE [m]")
            self.viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_difference")
            plt.close(fig)

        if "social-lstm - padding - fd" in self.nl_ADEs.keys() and "lstm - padding - fd" in self.nl_ADEs.keys():
            diff_non_fd = self.nl_ADEs["lstm - padding - fd"]["error"] - self.nl_ADEs["social-lstm - padding - fd"]["error"]
            fig, ax = plt.subplots()
            ax.plot(self.nl_ADEs["lstm - padding - fd"]["threshold"], diff_non_fd, color="black", linewidth=int(2), markersize=5)
            plt.legend(("LSTM loss fd - SLSTM loss fd",))
            plt.title("Fd ADE difference in nonlinear regions - V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
            plt.xlabel("threshold t")
            plt.ylabel("nl ADE [m]")
            self.viz.matplot(fig, win=str(self.V0) + "_" + str(self.sigma) + "_fd_difference")
            plt.close(fig)



if __name__ == "__main__":
    import os

    # Show plot via Visdom module
    servername = socket.gethostname()
    if "node" in servername:
        server = "http://atcremers10"
    else:
        server = 'http://localhost'

    viz = visdom.Visdom(server=server, port=8098, env="Nonlinear_ADE")

    # Get input arguments from shell
    parser = argparse.ArgumentParser("Visualize ADE in nonlinear regions")
    parser.add_argument("--V0", default=-1.0, type=float, help="Specify V0 of dataset")
    parser.add_argument("--sigma", default=-1, type=float, help="Specify sigma of dataset")

    # Get arguments
    args = parser.parse_args()

    if args.V0 == -1.0 or args.sigma == -1.0:
        raise ValueError("please insert valid V0 and sigma for dataset")

    root = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "Analyse_Results", "Nonlinear_loss")
    filepath = "socialforce_nl_loss.xlsx"
    visualizing = visualize_nl_ADE(V0=args.V0, sigma=args.sigma, viz=viz)
    visualizing.read_xlsx(filepath=root+"//"+filepath)

    # Create difference-plot
    #visualizing.create_diff()
