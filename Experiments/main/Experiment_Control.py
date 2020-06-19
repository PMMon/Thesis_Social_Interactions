# ==Imports==
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os, sys
import logging
import torch
import visdom
import numpy as np
import socket
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from data.loader import data_loader
from configs.config_dataprep import config as config_dataprep
from configs.config_opt import *
from helper.bool_flag import bool_flag
from helper.move_opt_stat_to_GPU import move_opt_state_to_GPU
from helper.results_to_xlxs import note_best_val, note_test_results_for_socialforce, note_nonlinear_loss
from helper.histogram import create_histogram
from solver.solver import  Solver
from model.modules import LINEAR, LSTM
from arguments.configurations import  load_args
#=============

# ================================================== Description =======================================================
# Main script to conduct experiments on Trajectory Prediction Models on real and synthetic datasets.
#
# The script allows you to train, validate and test your models on real and synthetic datasets.
#
# For real datasets:
# 1) Make sure flag --socialforce is set to False
# 2) Ensure that the dataset exists and is saved in folder: Experiments/datasets/dataset_name
# 3) Specify name of dataset with flag --dataset_name
#
# For synthetic datasets that focus on social interactions between pedestrians:
# 1) Generate dataset with script ./Generate_Datasets/main/GenPedPedScene.py
# 2) Ensure that respective dataset was created in folder Experiments/datasets/dataset_name
# 3) Make sure flag --socialforce is set to True
# 4) Specify scenario with flag --dataset_type (either square or rectangle)
# 5) Specify values of V0 and sigma with flags --V0 and --sigma of the respective dataset
#
# ======================================================================================================================

class Experiment:
    """
    Main Class to conduct experiments on trajectory prediction models.
    It is possible to train, validate and test your Model (Linear Model, LSTM Model or Social-LSTM Model) on real or synthetic datasets.
    """
    def __init__(self, args):
        self.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        if args.model_type.lower() not in ["linear", "lstm", "social-lstm"]:
            raise ValueError("Please choose model_type:\n- linear\n- lstm\n- social-lstm")

        if args.args_set != "":
            self.args = load_args(args)
        else:
            self.args = args

    def run_experiment(self):
        """
        Run Experiment with different configurations (input via --argument):
        1) --model_type: linear, lstm, social-lstm
        2) --socialforce: specify whether dataset is synthetic dataset (True), or real (False)
        3) --phase: train or test

        Trained Models are saved under "Saved_Models/dataset_name/model_name". For a list of possible input arguments run script with --help.
        """

        # ==============================
        #   Check input-configurations
        # ==============================
        viz = self.connect2visdom(self.args.viz_server, self.args.viz_port, "losses")

        if self.args.lstm_pool and self.args.model_type != "social-lstm":
            print("WARNING: model_type was changed to social-lstm since Social-Pooling was set to True!")
            self.args.model_type = "social-lstm"
        elif self.args.model_type == "social-lstm" and not self.args.lstm_pool:
            print("WARNING: Social-Pooling was set to True because model_type is specified as social-lstm!")
            self.args.lstm_pool = True

        if self.args.socialforce:
            # Get name of dataset and model - for synthetic datasets model name will be of the form: ModelType_UniqueIdentifier
            parameterinfo = "V0" + str(self.args.V0).replace(".", "u") + "b" + str(self.args.sigma).replace(".", "u")
            self.args.model_name = self.args.model_type + "_" + parameterinfo
            self.args.dataset_name = self.args.dataset_type + "simulated_" + parameterinfo
        else:
            # Get name of dataset and model - for real datasets model name will be of the form: ModelType_DatasetName_InputModelName
            if (self.args.model_type not in self.args.model_name) or (self.args.dataset_name not in self.args.model_name):
               self.args.model_name = self.args.model_type + "_" + self.args.dataset_name + "_" + self.args.model_name
            if self.args.model_name[-1] == "_":
                self.args.model_name = self.args.model_name[:-1]

        if not os.path.exists(os.path.join(self.root_path, "datasets", self.args.dataset_name)):
            raise ValueError("Specified dataset: %s does not exist! Please check existing datasets and specify one with flag: --dataset_name" % (self.args.dataset_name))

        if self.args.phase == "test":
            # Ensure that for testing we only run one epoch and do not augment the data
            self.args.num_epochs = 1
            self.args.load_model = True
            self.args.data_augmentation = False

        # Get configs for data preparation
        config_data = config_dataprep()

        # Configurations for logging losses
        self.log_configs()

        # ==============
        #   Data prep
        # ==============
        dset, loader, dset_val, val_loader = self.load_data(config_data, logger)

        # =============
        #   Get model
        # =============
        if self.args.model_type.lower() == "linear":
            model = LINEAR(self.args)
        elif self.args.model_type.lower() == "lstm":
            model = LSTM(self.args)
        elif self.args.model_type.lower() == "social-lstm":
            self.args.lstm_pool = True
            model = LSTM(self.args)
        else:
            raise ValueError("Please choose model_type either: linear, lstm or social-lstm")

        # ===============
        #   Optimizer
        # ===============

        # Define optimizer
        if self.args.optim.lower() == "adam":
            config_opt = config_Adam(lr=self.args.lr, weight_decay=self.args.wd)
            optimizer = torch.optim.Adam(model.parameters(),**{"lr": config_opt.lr, "betas": config_opt.betas, "eps": config_opt.eps,"weight_decay": config_opt.weight_decay})
        elif self.args.optim.lower() == "rmsprop":
            config_opt = config_RMSprop(lr=self.args.lr, weight_decay=self.args.wd)
            optimizer = torch.optim.RMSprop(model.parameters(),**{"lr": config_opt.lr, "alpha": config_opt.alpha, "eps": config_opt.eps,"weight_decay": config_opt.weight_decay})
        elif self.args.optim.lower() == "sgd":
            config_opt = config_SGD(lr=self.args.lr, weight_decay=self.args.wd)
            optimizer = torch.optim.SGD(model.parameters(),**{"lr": config_opt.lr, "weight_decay": config_opt.weight_decay,"momentum": config_opt.momentum, "dampening": config_opt.dampening, "nesterov": config_opt.nesterov})
        else:
            raise ValueError("Please specify a valid optimizer with optim!")

        # =================
        #   load Model
        # ================

        # Define saving directory for Model
        saving_directory = os.path.join(self.root_path, "Saved_Models", str(self.args.dataset_name))
        if not os.path.exists(saving_directory):
            print("Create path for saving model: %s" % (saving_directory))
            os.makedirs(saving_directory)

        if self.args.save_model:
            print("Model will be saved under: " + saving_directory + "/" + self.args.model_name)
        else:
            print("Model " + self.args.model_name + " will not be saved")

        loaded_states = {}
        loss_history_all = {self.args.dataset_name: {"train": {}, "val": {}, "test": {}}}
        epochs = 0

        if self.args.load_model:
            print("Loading Model...")
            if os.path.isfile(os.path.join(str(saving_directory), self.args.model_name)):
                loaded_states = torch.load(os.path.join(str(saving_directory), self.args.model_name))
                model.load_state_dict(loaded_states["model_state_dict"])
                device = torch.device("cpu")
                optimizer.load_state_dict(loaded_states["optimizer_state_dict"])
                move_opt_state_to_GPU(optimizer)

                loss_history_all = loaded_states["loss"]
                if self.args.dataset_name not in loss_history_all:
                    loss_history_all[self.args.dataset_name] = {"train": {}, "val": {}, "test": {}}
                if self.args.phase == "train":
                    epochs = loaded_states["epochs"]
                print("Model loaded...")
            else:
                print("Tried to load model, but model does not exist!\nContinued with new model")
        else:
            print("Create new Model...")

        # =================
        #   Train Model
        # =================

        # Define solver
        solver = Solver(optim=optimizer, loss_all=loss_history_all, epochs=epochs, args=self.args, server=self.server)

        # Train/eval Model
        solver.train(model, self.args.model_type.lower(), loader, val_loader, phase=self.args.phase, log_nth=self.args.log_nth, num_epochs=self.args.num_epochs)

        # Save Model
        if self.args.save_model:
            model.save(solver.optim, solver.loss_history_all, solver.last_epoch, os.path.join(str(saving_directory), self.args.model_name))

        # ==============
        #   RESULTS
        # =============

        # Plot Histogram for Distances to other pedestrians in order to avoid collision behavior of models
        if self.args.phase == "test":
            if self.args.analyse_coll_avoidance:
                histo = create_histogram(self.args, self.server)
                histo.plot_histogram_distances(solver)

        # Note Results of testing of Socialforce Experiment for further Analysis
        if self.args.socialforce:
            if self.args.phase == "test":
                print("Writing test losses for Socialforce-Experiment to .xlsx-file...")

                # ==============================================================
                # For Analysis of overall ADE and FDE for specific V0 and sigma
                # ==============================================================

                result_path = os.path.join(self.root_path, "Analyse_Results","Overall_Loss")
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                file_name = "socialforce_" + str(self.args.model_type) + "_results"
                if self.args.model_type == "social-lstm":
                    file_name += "_ns_" + str(int(self.args.neighborhood_size)) + "_gs_" + str(int(self.args.grid_size))
                file_name += ".xlsx"
                filepath = os.path.join(result_path, file_name)

                final_ADE_loss = solver.loss_history_all[self.args.dataset_name][self.args.phase]["G_ADE"][-1]
                final_FDE_loss = solver.loss_history_all[self.args.dataset_name][self.args.phase]["G_FDE"][-1]

                # Create or Append Excel sheet with ADE and FDE of respective dataset
                note_test_results_for_socialforce(filepath, self.args, final_ADE_loss, final_FDE_loss)

                # ===========================================
                # For Analysis of ADE in non-linear regions
                # ===========================================

                if "G_ADE_nl_regions" in model.losses:
                    print("Writing loss in nonlinear Regions to .xlsx-file...")
                    result_path = os.path.join(self.root_path, "Analyse_Results", "Nonlinear_ADE")
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)

                    file_name = "socialforce_nl_loss.xlsx"
                    filepath = os.path.join(result_path, file_name)

                    final_nl_ADE_loss = solver.loss_history_all[self.args.dataset_name][self.args.phase]["G_ADE_nl_regions"][-1]
                    note_nonlinear_loss(filepath, self.args, final_ADE_loss, final_nl_ADE_loss)

        # Note results of best validation ADE loss of training
        if self.args.phase == "train":
            print("Writing train losses with configs to .xlsx-file...")
            result_path = os.path.join(self.root_path, "Stats", "BestVal", self.args.dataset_name)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            xlsx_file = self.args.dataset_name + "_" + self.args.model_type + ".xlsx"
            filepath = os.path.join(result_path, xlsx_file)

            note_best_val(filepath, self.args, solver.best_val, solver.best_val_FDE, solver.best_val_epoch)

        # Plot losses
        if self.args.visdom:
            self.plot_loss_visdom(viz, solver.loss_history_all)
        else:
            self.plot_loss(solver.loss_history_all)

        # Print Hyperparameters as Summary
        self.log_HyperParameters()

        # ============================


    def plot_loss_visdom(self, viz, loss_history):
        """
        Show plot of losses with Visdom module
        :param viz: Visdom connection
        :param loss_history: Dictionary with losses. Has structure dict[dataset_name][phase][loss_type] = np.array([loss per epoch])
        """
        first_plot = True
        for dataset in loss_history:
            for phase, phase_dict in loss_history[dataset].items():
                for loss_type, loss_list in phase_dict.items():
                    if loss_type == "G_MSE" or loss_type == "MSE":
                        continue
                    if phase.lower() == "train" or phase.lower() == "test":
                        first_plot = True
                    if loss_list:
                        if phase.lower() == "train" or phase.lower() == "val":
                            identifier = str(dataset) + " - " + str(self.args.model_type) + " - " + str(loss_type)
                            if first_plot:
                                viz.line(env="losses_" + str(self.args.model_type), name=str(loss_type) + "_" + str(phase),
                                         win=identifier, Y=np.asarray(loss_list), X=np.arange(len(loss_list)) + 1,
                                         opts=dict(showlegend=True, env=dataset, title=identifier, xlabel="Epochs", ylabel="Loss", layoutopts=dict(plotly=dict(yaxis=dict(range=[0, max(loss_list) * 1.5 + 1])))))
                                first_plot = False
                            else:
                                viz.line(env="losses_" + str(self.args.model_type), name=str(loss_type) + "_" + str(phase), update="append",
                                         win=identifier, Y=np.asarray(loss_list), X=np.arange(len(loss_list)) + 1, opts=dict(showlegend=True))
                        else:
                            print(str(dataset) + "-" + str(phase) + "-" + str(loss_type) + ": " + str(loss_list[-1]))


    def plot_loss(self, loss_history):
        """
        Create and save plot of model losses
        :param loss_history: Dictionary with losses. Has structure dict[dataset_name][phase][loss_type] = np.array([loss per epoch])
        """
        for dataset in loss_history:
            path = os.path.join(self.root_path, "Stats", "LossPlots", str(dataset), str(self.args.model_type))
            if not os.path.exists(path):
                os.makedirs(path)
            if "train" not in loss_history[dataset].keys():
                print("No suitable loss history found. Skip plotting losses!")
                continue

            for loss_type in loss_history[dataset]["train"]:
                if loss_type == "G_MSE" or loss_type == "MSE":
                    continue

                if loss_history[dataset]["train"][loss_type]:
                    fig, ax = plt.subplots()
                    plt.title(str(dataset) + " - " + str(self.args.model_type) + " - " + str(loss_type))
                    plt.xlabel("Epochs")
                    plt.ylabel(loss_type)

                    # Loss of training
                    loss_train = loss_history[dataset]["train"][loss_type]
                    max_train = max(loss_train)
                    ax.plot(np.arange(len(loss_train)) + 1, np.asarray(loss_train), label="train")

                    # Loss of validation
                    loss_val = loss_history[dataset]["val"][loss_type]
                    max_val = max(loss_val)
                    ax.plot(np.arange(len(loss_val)) + 1, np.asarray(loss_val), label="val")

                    ax.set_ylim([0, max(max_train, max_val) * 1.5 + 1])
                    plt.grid(linestyle='dotted')
                    plt.legend(loc="upper right")

                    fig_name = str(self.args.model_type) + "_" + str(loss_type) + ".jpg"
                    file_spec = os.path.join(path, fig_name)

                    fig.savefig(file_spec)
                    plt.close(fig)

                if loss_type in loss_history[dataset]["test"].keys():
                    if loss_history[dataset]["test"][loss_type]:
                        print(str(dataset) + "-" + str("test") + "-" + str(loss_type) + ": " + str(loss_history[dataset]["test"][loss_type][-1]))


    def load_data(self, config_dataprep, logger):
        """
        Load data according to phase (train or test). Note that the data can either be padded or not, depending on the boolean self.args.lstm_pool.
        :param config_dataprep: Configurations for data-loading
        :param logger: Logger in order to log during data-loading
        :return: dataset, loader
        """
        print("load data...")

        if self.args.phase == "train":
            dset, loader = data_loader(self.args, config_dataprep, phase="train", logger=logger)
            self.args.data_augmentation = False
            dset_val, val_loader = data_loader(self.args, config_dataprep, phase="val", logger=logger)
        elif self.args.phase == "test":
            dset, loader = data_loader(self.args, config_dataprep, phase="test", logger=logger)
            dset_val, val_loader = 0, 0
        else:
            raise ValueError("Please choose either train or test as phase!")

        print("data loaded.")

        return dset, loader, dset_val, val_loader


    def log_configs(self):
        """
        Print Configurations
        """
        print("\n==Configs=="
              "\nModel_Name: " + self.args.model_name + "\nModel_Type: " + self.args.model_type + "\nDataset: " + self.args.dataset_name +
              "\nSocialForce Experiment: " + str(self.args.socialforce) + "\nPhase: " + self.args.phase + "\nNumber of Epochs: " + str(self.args.num_epochs) +
              "\nLoad Model: " + str(self.args.load_model) + "\nBatch Size: " + str(self.args.batch_size) + "\nOptimizer: " + self.args.optim +
              "\nPadding: " + str(self.args.padding) + "\nFd: " + str(self.args.final_position) + "\nTrain on: " + self.args.train_loss +
              "\n===========\n")


    def log_HyperParameters(self):
        """
        Print HyperParameters
        """
        print("Configs:"
              "\nDropout: " + str(self.args.dropout) + " - Batch_size: " + str(self.args.batch_size) +
              "\nlr: " + str(self.args.lr) + " - wd: " + str(self.args.wd) + " - emb_dim: " + str(self.args.emb_dim) + " - enc_h_dim: " + str(self.args.encoder_h_dim) +
              " - dec_h_dim: " + str(self.args.decoder_h_dim) + " - padding: " + str(self.args.padding) + " - final_position: " + str(self.args.final_position))


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

    logger = logging.getLogger(__name__)

    # Get input arguments from shell
    parser = argparse.ArgumentParser("Trajectory Prediction Basics")

    # Configs for Experiment
    parser.add_argument("--socialforce", default=False, type=bool_flag, help="Specify whether to run experiment on synthetic data that is created by the Social Force Model")
    parser.add_argument("--phase", default="train", type=str, help="Specify whether to train (and validate) or test your model")
    parser.add_argument("--show_traj", default=False, type=bool_flag, help="Specify whether to plot predicted trajectories")
    parser.add_argument("--log_nth", default=1, type=int, help="Define when to log training/validation losses")
    parser.add_argument("--config", default=1, type=int, help="Specify config-file")
    parser.add_argument("--log", default="", type=str, help="Specify log-level. Choose either: Info, Debug, Error, Critical or leave unspecified")

    # Configs for SocialForce-Experiment
    parser.add_argument("--V0", default=0, type=int, help="Define V0")
    parser.add_argument("--sigma", default=0, type=float, help="Define sigma")

    # Configs for Model
    parser.add_argument("--model_name", default="", type=str, help="Define model name for saving")
    parser.add_argument("--model_type", default="lstm", type=str,help="Define type of model. Choose either: linear, lstm or social-lstm")
    parser.add_argument("--save_model", default=True, type=bool_flag, help="Save trained model")
    parser.add_argument("--load_model", default=False, type=bool_flag, help="Specify whether to load existing model")
    parser.add_argument("--lstm_pool", default=False, type=bool_flag, help="Specify whether to enable social pooling")
    parser.add_argument("--pooling_type", default="social_pooling", type=str, help="Specify pooling method")
    parser.add_argument("--neighborhood_size", default=10.0, type=float, help="Specify neighborhood size to one side")
    parser.add_argument("--grid_size", default=10, type=int, help="Specify grid size")
    parser.add_argument("--args_set", default="", type=str, help="Specify predefined set of configurations for respective model. Choose either: lstm or social-lstm")

    # Config for data-preparation
    parser.add_argument("--dataset_name", default="to_be_defined", type=str, help="Specify dataset")
    parser.add_argument("--dataset_type", default="square", type=str, help="Specify dataset-type. For real datasets choose: 'real'. For synthetic datasets choose either 'square' or 'rectangle'")
    parser.add_argument("--obs_len", default=8, type=int, help="Specify length of observed trajectory")
    parser.add_argument("--pred_len", default=12, type=int, help="Specify length of predicted trajectory")
    parser.add_argument("--data_augmentation", default=True, type=bool_flag, help="Specify whether or not you want to use data augmentation")
    parser.add_argument("--batch_norm", default=False, type=bool_flag, help="Batch Normalization")
    parser.add_argument("--max_num", default=1000000, type=int, help="Specify maximum number of ids")
    parser.add_argument("--skip", default=20, type=int, help="Specify skipping rate")
    parser.add_argument("--PhysAtt", default="", type=str, help="Specify physicalAtt")
    parser.add_argument("--padding", default=True, type=bool_flag, help="Specify if padding should be active")
    parser.add_argument("--final_position", default=False, type=bool_flag, help="Specify whether final positions of pedestrians should be passed to model or not")

    # Configs for training, val, testing
    parser.add_argument("--batch_size", default=32, type=int, help="Specify batch size")
    parser.add_argument("--wd", default=0.0, type=float, help="Specify weight decay")
    parser.add_argument("--lr", default=0.001, type=float, help="Specify learning rate")
    parser.add_argument("--encoder_h_dim", default=64, type=int, help="Specify hidden state dimension h of encoder")
    parser.add_argument("--decoder_h_dim", default=32, type=int, help="Specify hidden state dimension h of decoder")
    parser.add_argument("--emb_dim", default=32, type=int, help="Specify dimension of embedding")
    parser.add_argument("--num_epochs", default=250, type=int, help="Specify number of epochs")
    parser.add_argument("--dropout", default=0.0, type=float, help="Specify dropout rate")
    parser.add_argument("--num_layers", default=1, type=int, help="Specify number of layers of Vanilla/Social LSTM Model")
    parser.add_argument("--optim", default="Adam", type=str, help="Specify optimizer. Choose either: adam, rmsprop or sgd")

    # Configs for additional tools for analysis
    parser.add_argument("--approx_scheme", default="menger_curvature", type=str, help="Specify approximation scheme for calculation of curvature")
    parser.add_argument("--nl_ADE", default=False, type=bool_flag, help="Specify whether or not to calculate nonlinear ADE of model in specific nonlinear regions")
    parser.add_argument("--threshold_nl", default=0.5, type=float, help="Define threshold-value for nonlinear ADE")
    parser.add_argument("--nl_classified", default=False, type=bool_flag, help="Specify whether to analyse and classify trajectories with respect to their degree of nonlinearity")
    parser.add_argument("--visualize_classified", default=False, type=bool_flag, help="Specify whether to plot classified trajectories")
    parser.add_argument("--analyse_coll_avoidance", default=False, type=bool_flag, help="Specify whether to analyze collision avoidance behavior of models")
    parser.add_argument("--CDF_CollAvoid", default=False, type=bool_flag, help="Specify whether or not to create CDF-plot for collision avoidance behavior of models")
    parser.add_argument("--train_loss", default="ADE", type=str, help="Specify loss on which model trains")
    parser.add_argument("--plot_gradient_flow", default=False, type=bool_flag, help="Specify whether or not to plot values of gradients for monitoring possible vanishing gradients")
    parser.add_argument("--analyse_real_dataset", default=False, type=bool_flag, help="Specify whether to analyse real dataset with respect to the amount of suitable trajectories in the dataset")

    # Configs for visdom module
    parser.add_argument("--visdom", default=False, type=bool_flag, help="Specify whether or not to plot via visdom")
    parser.add_argument("--viz_port", default=8090, type=int, help="Specify port for visdom")
    parser.add_argument("--viz_server", default="", type=str, help="Specify server for visdom")

    # Get arguments
    args = parser.parse_args()

    if args.log.upper() in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]:
        level = getattr(logging, args.log.upper())
        logging.basicConfig(level=level)

    # Run Experiment
    Customized_Experiment = Experiment(args)
    Customized_Experiment.run_experiment()