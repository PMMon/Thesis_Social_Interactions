import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import visdom

class create_histogram:
    """
    Class for creation of histograms to analyze distributions of e.g. curvature-values of the observed trajectories,
    euclidean distances between pedestrians in a scene, etc.
    """
    def __init__(self, args, server):
        self.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.args = args
        self.server = server

    def plot_histogram_curvatures(self, values, bins):
        """
        Creates histogram for the curvature-values of the observed trajectories
        :param values: Numpy array of shape (nr_traj, pred_len - 2) with curvature values of trajectories
        :param bins: Numpy array defining bin-size
        """
        if self.args.visdom:
            self.vis = visdom.Visdom(server=self.server, port=self.args.viz_port, env="EvalMetrics_NonlinearADE_CurvValues")
        else:
            path = os.path.join(self.root_path, "Analyse_Results", "Nonlinear_ADE", self.args.dataset_name)
            if not os.path.exists(path):
                os.makedirs(path)

        mean = values.mean()
        std_dev = values.std()
        max = values.max()
        min = values.min()

        hist, bins = np.histogram(values, bins=bins)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        # First plot - wide overview
        fig, ax = plt.subplots()
        plt.bar(center, hist, align='center', width=width)
        text_targets = "Max value: " + str(max.round(2)) + "\nMin Value: " + str(min.round(2)) + "\nMean: " + str(mean.round(2)) + "\nstd_deviation: " + str(std_dev.round(2))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        major_ticks = np.arange(0, values.max() + 0.1, 2.0)
        minor_ticks = np.arange(0, values.max() + 0.1, 0.5)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.text(0.7, 0.95, text_targets, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

        plt.title("Curvature values of dataset: " + str(self.args.dataset_name))
        plt.xlabel("Curvature k")
        plt.ylabel("Occurrence")

        if self.args.visdom:
            self.vis.matplot(fig, win= str(self.args.dataset_name) + "_wide")
        else:
            plot_name = "Curv_values_" + str(self.args.dataset_name) + "_wide.jpg"
            plot_file = os.path.join(path, plot_name)
            fig.savefig(plot_file)

        plt.close(fig)

        # Second Plot - zoom into important regions
        fig2, ax2 = plt.subplots()
        bins = bins[bins <= mean+2*std_dev]
        sum_hist = np.sum(hist)
        hist = hist[:len(bins)]/sum_hist * 100

        plt.bar(bins, hist, align='center', width=width)
        major_ticks = np.arange(0, bins.max() + 0.1, 0.4)
        minor_ticks = np.arange(0, bins.max() + 0.1, 0.1)
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)

        plt.title("Curvature values of dataset: " + str(self.args.dataset_name))
        plt.xlabel("Curvature k")
        plt.ylabel("Occurrence [%]")

        if self.args.visdom:
            self.vis.matplot(fig2, win= str(self.args.dataset_name) + "_zoom")
        else:
            plot_name = "Curv_values_" + str(self.args.dataset_name) + "_zoom.jpg"
            plot_file = os.path.join(path, plot_name)
            fig2.savefig(plot_file)

        plt.close(fig2)


    def plot_histogram_distances(self, solver):
        """
        This Function will create a histogram to to analyze the distances between pedestrians in a given scene.
        """
        # Specify filepath for saving plots
        self.path = os.path.join(self.root_path, "Analyse_Results", "Collision_Behavior", self.args.dataset_name)
        if self.args.visdom:
            self.vis = visdom.Visdom(server=self.server, port=self.args.viz_port, env="EvalMetrics_CollBehavior_Histogram")
        else:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

            ground_truth_jpg = "distances_ground_truth.jpg"
            if "social" in self.args.model_type:
                model_jpg = "distances_" + str(self.args.model_type) + "_NS_" +  str(self.args.neighborhood_size).replace(".", "u") + "_GS_" + str(self.args.grid_size).replace(".", "u") + ".jpg"
            else:
                model_jpg = "distances_" + str(self.args.model_type) + ".jpg"

        # The sigma defines the range at which V0(r) = 0.1*V0
        sigma_r0_rel = {0.2171: 0.5, 0.4343: 1, 0.8686: 2, 1.303: 3, 1.7371: 4, 2.171: 5, 2.6058: 6}

        histo_distances_targets_all, bins_targets_all = np.histogram(solver.distances_targets, bins=np.arange(0, solver.distances_targets.max()+0.1, 0.1))
        histo_distances_outputs_all, bins_outputs_all = np.histogram(solver.distances_outputs, bins=np.arange(0, solver.distances_outputs.max()+0.1, 0.1))

        # Calculate fractions
        total_occ_gt = np.sum(histo_distances_targets_all)
        total_occ_pred = np.sum(histo_distances_outputs_all)

        histo_distances_targets, bins_target = np.histogram(solver.distances_targets, bins=np.arange(0, sigma_r0_rel[self.args.sigma] + 0.2, 0.1))
        histo_distances_pred, bins_pred = np.histogram(solver.distances_outputs, bins = np.arange(0, sigma_r0_rel[self.args.sigma] + 0.2, 0.1))

        rel_occ_gt = histo_distances_targets / total_occ_gt * 100
        rel_occ_pred = histo_distances_pred / total_occ_pred * 100

        max_occ = np.max(np.concatenate((rel_occ_gt, rel_occ_pred), axis=0))

        # Create PDF for targets
        ticks_y = np.arange(0, max_occ, 1)
        major_ticks = np.arange(0, sigma_r0_rel[self.args.sigma] + 0.1, 0.5)
        minor_ticks = np.arange(0, sigma_r0_rel[self.args.sigma] + 0.1, 0.1)

        collision_bins_targets = bins_target[bins_target < 1.0]
        collision_targets = histo_distances_targets[:len(collision_bins_targets)]

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        text_targets = "Total = " + str(total_occ_gt) + "\n< 1.0m = " + str(round(float((np.sum(collision_targets) / total_occ_gt) * 100), 1)) + "%\nV0: " + str(self.args.V0) + "\nsigma: " + str(self.args.sigma)

        # Plot histogram
        fig, ax = plt.subplots()
        plt.bar(bins_target[:-1], rel_occ_gt, align='edge', width=0.09)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(ticks_y)
        ax.text(0.7, 0.95, text_targets, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
        plt.title("Histogram for distances from ground truth data")
        plt.xlabel("r [m]")
        plt.ylabel("Occurences [%]")

        if self.args.visdom:
            self.vis.matplot(fig, win="gt_pdf" + str(self.args.V0) + str(self.args.sigma))
        else:
            fig.savefig(os.path.join(self.path, ground_truth_jpg))

        plt.close(fig)

        # Create PDF for Predictions
        collision_bins_predictions = bins_pred[bins_pred < 1.0]
        collision_predictions = histo_distances_pred[:len(collision_bins_predictions)]

        text_outputs = "Total = " + str(total_occ_pred) + "\n< 1.0m = " + str(round(float((np.sum(collision_predictions) / total_occ_pred) * 100), 1)) + "%\nV0: " + str(self.args.V0) + "\nsigma: " + str(self.args.sigma)

        fig2, ax2 = plt.subplots()
        plt.bar(bins_pred[:-1], rel_occ_pred, align='edge', width=0.09)
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.set_yticks(ticks_y)
        ax2.text(0.7, 0.95, text_outputs, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props)

        if "social" in self.args.model_type:
            plt.title("Histogram for distances from model " + str(self.args.model_type) + " - NS: " + str(self.args.neighborhood_size) + " - GS: " + str(self.args.grid_size))
        else:
            plt.title("Histogram for distances from model " + str(self.args.model_type))

        plt.xlabel("r [m]")
        plt.ylabel("Occurences [%]")

        if self.args.visdom:
            self.vis.matplot(fig2, win=str(self.args.model_type) + "_" + str(self.args.V0) + "_" + str(self.args.sigma) + "_" + str(self.args.neighborhood_size) + "_" + str(self.args.grid_size) + "_pdf")
        else:
            fig2.savefig(os.path.join(self.path, model_jpg))

        plt.close(fig2)

        # Create CDF of distributions
        if self.args.CDF_CollAvoid:
            json_filename = "CDF_dict_V0" + str(self.args.V0) + "_b" + str(self.args.sigma).replace(".", "u") + ".json"
            file_spec = os.path.join(self.path, "Json_file", json_filename)
            if not os.path.exists(os.path.join(self.path, "Json_file")):
                os.makedirs(os.path.join(self.path, "Json_file"))

            if os.path.isfile(file_spec):
                with open(file_spec) as json_file:
                    values_for_CDF = json.load(json_file)
            else:
                values_for_CDF = {}
                values_for_CDF["ground truth"] = solver.distances_targets.tolist()

            values_for_CDF[self.args.model_type] = solver.distances_outputs.tolist()

            with open(file_spec, "w") as output_json:
                json.dump(values_for_CDF, output_json)

            # Create CDF plot
            self.create_CDF_plot(file_spec)


    def create_CDF_plot(self, file_spec):
        if not os.path.exists(file_spec):
            raise ValueError("Can not create CDF plot since json-file of data does not exist!")

        with open(file_spec) as json_file:
            values_for_CDF = json.load(json_file)

        sigma_r0_rel = {0.2171: 0.5, 0.4343: 1, 0.8686: 2, 1.303: 3, 1.7371: 4, 2.171: 5, 2.6058: 6}

        major_ticks = np.arange(0, sigma_r0_rel[self.args.sigma] + 0.1, 0.5)
        minor_ticks = np.arange(0, sigma_r0_rel[self.args.sigma] + 0.1, 0.1)

        # Create CDF for targets
        fig_CDF, ax_CDF = plt.subplots()

        for key, value in values_for_CDF.items():
            ax_CDF.hist(value,  bins = np.arange(0, sigma_r0_rel[self.args.sigma] + 0.2, 0.1),density=True, histtype='step', cumulative=True, label=key)

        ax_CDF.set_xticks(major_ticks)
        ax_CDF.set_xticks(minor_ticks, minor=True)
        ax_CDF.set_yticks(np.arange(0, 1.1, 0.2))
        ax_CDF.set_yticks(np.arange(0, 1.1, 0.1), minor=True)
        plt.title("CDF for distances V0: " + str(self.args.V0) + " - sigma: " + str(self.args.sigma))
        plt.xlabel("r [m]")
        plt.ylabel("Likelihood of occurrence")
        ax_CDF.grid(True)
        ax_CDF.legend(loc='upper left')

        if self.args.visdom:
            self.vis.matplot(fig_CDF, win="CDF_all" + str(self.args.V0) + str(self.args.sigma))
        else:
            fig_CDF.savefig(os.path.join(self.path, "CDF_V0_" + str(self.args.V0) + "_sigma_" + str(self.args.sigma).replace(".", "u") + ".jpg"))

        plt.close(fig_CDF)
