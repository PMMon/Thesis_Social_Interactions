import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import visdom

class create_histogram:
    def __init__(self, args, server):
        self.args = args
        self.server = server

    def plot_histogram_curvatures(self, values, bins):
        self.vis = visdom.Visdom(server=self.server, port=self.args.viz_port, env="Nonlinear_ADE")
        fig, ax = plt.subplots()

        mean = values.mean()
        std_dev = values.std()
        max = values.max()
        min = values.min()

        hist, bins = np.histogram(values, bins=bins)
        width = 0.7 * (bins[1] - bins[0])

        # First plot - wide overview
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        text_targets = "Max value: " + str(max.round(2)) + "\nMin Value: " + str(min.round(2)) + "\nMean: " + str(mean.round(2)) + "\nstd_deviation: " + str(std_dev.round(2))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        major_ticks = np.arange(0, values.max() + 0.1, 2.0)
        minor_ticks = np.arange(0, values.max() + 0.1, 0.5)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.text(0.7, 0.95, text_targets, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

        plt.title("Histogram for curvature values of dataset V0: " + str(self.args.V0) + " - sigma: " + str(self.args.sigma))
        plt.xlabel("curvature k")
        plt.ylabel("Occurrence")
        self.vis.matplot(fig, win=str(self.args.V0) + "_" + str(self.args.sigma))
        plt.close(fig)

        # Second Plot - zoom into important regions
        fig2, ax2 = plt.subplots()
        bins = bins[bins <= mean+2*std_dev]
        center = (bins[:-1] + bins[1:]) / 2
        sum_hist = np.sum(hist)
        hist = hist[:len(bins)]/sum_hist * 100

        plt.bar(bins, hist, align='center', width=width)
        #text_targets = "Max value: " + str(max.round(2)) + "\nMin Value: " + str(min.round(2)) + "\nMean: " + str(mean.round(2)) + "\nstd_deviation: " + str(std_dev.round(2))
        #text_targets = "Mean curv.: " + str(mean.round(2))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        major_ticks = np.arange(0, bins.max() + 0.1, 0.4)
        minor_ticks = np.arange(0, bins.max() + 0.1, 0.1)
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        #ax2.text(0.7, 0.95, text_targets, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props)

        plt.title("Histogram for curvature values of dataset V0: " + str(self.args.V0) + " - sigma: " + str(self.args.sigma))
        plt.xlabel("curvature k")
        plt.ylabel("Occurrence [%]")
        self.vis.matplot(fig2, win=str(self.args.V0) + "_" + str(self.args.sigma) + "_zoom")
        plt.close(fig2)

    def plot_histogram_distances(self, solver):
        """
        This Function will create several plots and histograms for analyzing the distances between pedestrians in a given scene.
        Plots being created are:    -
        """
        self.vis = visdom.Visdom(server=self.server, port=self.args.viz_port, env="histogram")

        # Specify filepath for saving plots
        root = os.path.join("Analyse_Results", "Distances", self.args.dataset_name)
        ground_truth_jpg = "distances_ground_truth.jpg"
        model_jpg = "distances_" + str(self.args.model_type) + ".jpg"
        if not os.path.exists(root):
            os.makedirs(root)

        # The sigma defines the range at which V0(r) = 0.1*V0
        sigma_r0_rel = {0.2171: 0.5, 0.4343: 1, 0.8686: 2, 1.303: 3, 1.7371: 4, 2.171: 5, 2.6058: 6}

        histo_distances_targets_all, bins_targets_all = np.histogram(solver.distances_targets, bins=np.arange(0, solver.distances_targets.max()+0.1, 0.1))
        histo_distances_outputs_all, bins_outputs_all = np.histogram(solver.distances_outputs, bins=np.arange(0, solver.distances_outputs.max()+0.1, 0.1))

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

        fig, ax = plt.subplots()

        plt.bar(bins_target[:-1], rel_occ_gt, align='edge', width=0.1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(ticks_y)
        ax.text(0.7, 0.95, text_targets, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
        plt.title("Histogram for distances from ground truth data")
        plt.xlabel("r [m]")
        plt.ylabel("Occurences [%]")
        self.vis.matplot(fig, win="gt_pdf" + str(self.args.V0) + str(self.args.sigma))
        fig.savefig(root + "//" + ground_truth_jpg)
        plt.close(fig)


        # Create Plots for Predictions
        collision_bins_predictions = bins_pred[bins_pred < 1.0]
        collision_predictions = histo_distances_pred[:len(collision_bins_predictions)]

        text_outputs = "Total = " + str(total_occ_pred) + "\n< 1.0m = " + str(round(float((np.sum(collision_predictions) / total_occ_pred) * 100), 1)) + "%\nV0: " + str(self.args.V0) + "\nsigma: " + str(self.args.sigma)

        fig2, ax2 = plt.subplots()
        plt.bar(bins_pred[:-1], rel_occ_pred, align='edge', width=0.1)
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
        self.vis.matplot(fig2, win=str(self.args.model_type) + "_" + str(self.args.V0) + "_" + str(self.args.sigma) + "_" + str(self.args.neighborhood_size) + "_" + str(self.args.grid_size) + "_pdf")
        fig2.savefig(root + "//" + model_jpg)
        plt.close(fig2)

        if self.args.CDF_name != "":
            json_filename = "CDF_dict_V0" + str(self.args.V0) + "_b" + str(self.args.sigma).replace(".", "u") + ".json"
            file_spec = root + "//" + json_filename

            if os.path.exists(file_spec):
                with open(file_spec) as json_file:
                    values_for_CDF = json.load(json_file)
            else:
                values_for_CDF = {}
                values_for_CDF["ground truth"] = solver.distances_targets.tolist()

            values_for_CDF[self.args.CDF_name] = solver.distances_outputs.tolist()

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
        self.vis.matplot(fig_CDF, win="CDF_all" + str(self.args.V0) + str(self.args.sigma))
        plt.close(fig_CDF)
