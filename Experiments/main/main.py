# ==Imports==
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import logging
import torch
import visdom
import numpy as np
import socket

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

logger = logging.getLogger(__name__)

# Get input arguments from shell
parser = argparse.ArgumentParser("Trajectory Prediction Basics")

# General Configs
parser.add_argument("--phase", default="train", type=str, help="specify whether train, test or val")
parser.add_argument("--log_nth", default=50, type=int, help="specify when to logg training/val results")
parser.add_argument("--show_traj", default=False, type=bool_flag, help="specify whether to plot trajectories or not")
parser.add_argument("--visdom", default=True, type=bool_flag, help="specify whether plot loss in visdom")
parser.add_argument("--viz_port", default=8098, type=int, help="specify port for visdom")
parser.add_argument("--plot_name", default="main", type=str, help="specify plot name")
parser.add_argument("--config", default=1, type=int, help="specify config file")
parser.add_argument("--write_result", default=True, type=bool_flag, help="Write results in csv")
parser.add_argument("--args_set", default="", type=str, help="specify args_set")

# Config about Model
parser.add_argument("--model_name", default="", type=str, help="Define model name for saving")
parser.add_argument("--model_type", default="to_be_defined", type=str, help="Define model type. Either Linear or LSTM Model")
parser.add_argument("--save_model", default=True, type=bool_flag, help="Save trained model")
parser.add_argument("--load_model", default=False, type=bool_flag, help="specify whether to load existing model")
parser.add_argument("--lstm_pool", default=False, type=bool_flag, help="specify whether to enable social pooling")
parser.add_argument("--pooling_type", default="social_pooling", type=str, help="specify pooling method")
parser.add_argument("--neighborhood_size", default=2.0, type=float, help="specify neighborhood_size to one side")
parser.add_argument("--grid_size", default=8, type=int, help="specify grid_size")

# Config about data-preparation
parser.add_argument("--dataset_name", default="to_be_defined", type=str, help="Specify dataset")
parser.add_argument("--obs_len", default=8, type=int, help="Specify observed length")
parser.add_argument("--pred_len", default=12, type=int, help="specify predicted length")
parser.add_argument("--data_augmentation", default=True, type=bool_flag, help="Specify whether or not you want to use data augmentation")
parser.add_argument("--batch_norm", default=False, type=bool_flag, help="Batch Normalization")
parser.add_argument("--max_num", default=100000, type=int, help="specify maximum number of ids")
parser.add_argument("--skip", default=20, type=int, help="specify skipping rate")
parser.add_argument("--PhysAtt", default="", type=str, help="specify physicalAtt")
parser.add_argument("--padding", default=False, type=bool_flag, help="Specify if padding should be active")
parser.add_argument("--final_position", default=False, type=bool_flag, help="Specify if final position should be passed to model or not")

# Config about training, val, testing
parser.add_argument("--batch_size", default=32, type=int, help="specify batch size")
parser.add_argument("--wd", default=0.03, type=float, help="specify weight decay")
parser.add_argument("--lr", default=0.001, type=float, help="specify learning rate")
parser.add_argument("--decoder_h_dim", default=16, type=int, help="specify decoder_h_dim")
parser.add_argument("--encoder_h_dim", default=16, type=int, help="specify encoder_h_dim")
parser.add_argument("--emb_dim", default=8, type=int, help="specify embedding dimension of encoder")
parser.add_argument("--num_epochs", default=30, type=int, help="specify number of epochs")
parser.add_argument("--dropout", default=0.0, type=float, help="specify dropout rate")
parser.add_argument("--num_layers", default=1, type=int, help="specify number of layers for LSTM")
parser.add_argument("--optim", default="Adam", type=str, help="specify optimizer")
parser.add_argument("--train_loss", default="ADE", type=str, help="specify loss on which model trains")
parser.add_argument("--threshold_nl", default=0.5, type=float, help="specify threshold above which trajectory is seen as nonlinear")
parser.add_argument("--approx_scheme", default="menger_curvature", type=str, help="specify approximation scheme for curvature approx")
parser.add_argument("--CDF_name", default="", type=str, help="specify name of model for CDF creation")
parser.add_argument("--plot_gradient_flow", default=False, type=bool_flag, help="specify whether you want to plot values of gradients for monitoring vanishing gradients")

# Config about socialforce Experiment
parser.add_argument("--socialforce", default=False, type=bool_flag, help="specify whether run model for with data from SocialForceModel")
parser.add_argument("--V0", default=0, type=int, help="Define V0")
parser.add_argument("--sigma", default=0, type=float, help="Define sigma")
parser.add_argument("--histo", default=False, type=bool_flag, help="Specify whether to plot histogram")

# Get arguments
args = parser.parse_args()

if args.args_set != "":
    args = load_args(args)

# Connect to visdom server
servername = socket.gethostname()
if "node" in servername:
    server = "http://atcremers11"
else:
    server = 'http://localhost'

viz = visdom.Visdom(server=server, port=args.viz_port, env=str(args.plot_name))

if args.lstm_pool:
    args.model_type = "social-lstm"
elif args.model_type == "social-lstm":
    args.lstm_pool = True

# For socialforce model
if args.socialforce:
    # Get name of dataset and model
    parameterinfo = "V0"
    for element_V0 in str(args.V0).split("."):
        parameterinfo += str(element_V0) + "u"
    parameterinfo = parameterinfo[:-1] + "b"
    for element_sigma in str(args.sigma).split("."):
        parameterinfo += str(element_sigma) + "u"
    parameterinfo = parameterinfo[:-1]

    args.model_name = str(args.model_type) + "_" +  parameterinfo
    args.dataset_name = "squaresimulated" + "_" + str(parameterinfo)

else:
    args.model_name = str(args.model_type) + "_" + str(args.dataset_name) + "_" + args.model_name
    if args.model_name[-1] == "_":
        args.model_name = args.model_name[:-1]

# Print Configurations
print("\n==Configs==")
print("Model_Name: " + str(args.model_name) + "\nModel_Type: " + str(args.model_type) + "\nDataset: " + str(args.dataset_name) + "\nSocialForce Experiment: " + str(args.socialforce) +
     "\nPhase: " + str(args.phase) + "\nNumber of Epochs: " + str(args.num_epochs) + "\nLoad Model: " + str(args.load_model)  + "\nBatch Size: "+ str(args.batch_size) + "\nOptimizer: " + str(args.optim) + "\nPadding: " + str(args.padding) +
      "\nFd: " + str(args.final_position) + "\nTrain on: " + str(args.train_loss))
print("===========\n")
# =====================

if args.phase == "test":
    args.num_epochs = 1
    args.load_model = True
    args.data_augmentation = False

# Get configs
config = config_dataprep()

# ==============
#   Data prep
# ==============

# Load data
print("load data...")

if args.phase == "train":
    dset, loader = data_loader(args, config, phase="train", logger=logger)
    args.data_augmentation = False
    dset_val, val_loader = data_loader(args, config, phase="val", logger=logger)
elif args.phase == "test":
    dset, loader = data_loader(args, config, phase="test", logger=logger)
    val_loader = 0
else:
    raise ValueError("Please choose either train or test as phase!")

print("data loaded.")


# =============
#   Get model
# =============

if args.model_type.lower() == "linear":
    model = LINEAR(args)
elif args.model_type.lower() == "lstm":
    model = LSTM(args)
elif args.model_type.lower() == "social-lstm":
    args.lstm_pool = True
    model = LSTM(args)
else:
    raise ValueError("Please choose model_type either: Linear or LSTM or Social-LSTM")


# ===============
#   Optimizer
# ===============

# Define optimizer
if args.optim.lower() == "adam":
    config_opt = config_Adam(lr=args.lr, weight_decay=args.wd)
    optimizer = torch.optim.Adam(model.parameters(),**{"lr": config_opt.lr, "betas": config_opt.betas, "eps": config_opt.eps,"weight_decay": config_opt.weight_decay})
elif args.optim.lower() == "rmsprop":
    config_opt = config_RMSprop(lr=args.lr, weight_decay=args.wd)
    optimizer = torch.optim.RMSprop(model.parameters(), **{"lr": config_opt.lr, "alpha": config_opt.alpha, "eps": config_opt.eps, "weight_decay": config_opt.weight_decay})
elif args.optim.lower() == "sgd":
    config_opt = config_SGD(lr=args.lr, weight_decay=args.wd)
    optimizer = torch.optim.SGD(model.parameters(), **{"lr": config_opt.lr, "weight_decay": config_opt.weight_decay, "momentum": config_opt.momentum, "dampening": config_opt.dampening, "nesterov": config_opt.nesterov})
else:
    raise ValueError("Please Enter a correct optimizer!")

# Define saving directory for Model
saving_directory = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "Saved_Models", str(args.dataset_name))
if not os.path.exists(saving_directory):
    os.makedirs(saving_directory)

if args.save_model:
    print("Model will be saved under: "+str(saving_directory)+"/"+(args.model_name))
else:
    print("Model "+str(args.model_name)+" will not be saved")


# ================
#   load Model
# ================

# Load settings if wished
loaded_states = {}
loss_history_all = {args.dataset_name: {"train": {}, "val": {}, "test": {}}}
epochs = 0

if args.load_model:
    print("loading Model...")
    if os.path.isfile(str(saving_directory)+"/"+str(args.model_name)):
        loaded_states = torch.load(saving_directory+"/"+str(args.model_name))
        model.load_state_dict(loaded_states["model_state_dict"])
        device = torch.device("cpu")
        optimizer.load_state_dict(loaded_states["optimizer_state_dict"])
        move_opt_state_to_GPU(optimizer)

        loss_history_all = loaded_states["loss"]
        if args.dataset_name not in loss_history_all:
            loss_history_all[args.dataset_name] = {"train": {}, "val": {}, "test": {}}
        if args.phase == "train":
            epochs = loaded_states["epochs"]
        print("Model loaded...")
    else:
        print("tried to load model, but model does not exist!\nContinued with new model")
else:
    print("Create new Model...")

# =================
#   Train Model
# =================

# Define solver
solver = Solver(optim=optimizer, loss_all=loss_history_all, epochs=epochs, args=args, server=server)

# Train/eval Model
solver.train(model, args.model_type.lower(), loader, val_loader, phase=args.phase, log_nth = args.log_nth, num_epochs=args.num_epochs)

# save Model
if args.save_model:
    model.save(solver.optim, solver.loss_history_all, solver.last_epoch, saving_directory + "/" + str(args.model_name))


# ==============
#   RESULTS
# =============

# Plot Histogram
if args.histo:
    if args.phase == "test":
        histo = create_histogram(args, server)
        histo.plot_histogram_distances(solver)

        """
        histo = visdom.Visdom(server=server, port=args.viz_port, env="histogram")

        root = os.path.join("Analyse_Results", "Distances", args.dataset_name)
        ground_truth_jpg = "distances_ground_truth.jpg"
        model_jpg =  "distances_" + str(args.model_type) + ".jpg"
        if not os.path.exists(root):
            os.makedirs(root)

        sigma_r0_rel = {0.2171: 0.5, 0.4343: 1, 0.8686: 2, 1.303: 3, 1.7371: 4, 2.171: 5, 2.6058: 6}

        max_occ = np.max(np.concatenate((solver.histo_distances_targets[0], solver.histo_distances_outputs[0]), axis=0))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        histo_distances_targets = np.histogram(solver.distances_targets, np.arange(0,sigma_r0_rel[args.sigma]+0.1,0.5))
        total_occ_gt = np.sum(solver.histo_distances_targets[0])
        rel_occ_gt = histo_distances_targets[0]/total_occ_gt*100

        ticks_y = np.arange(0,26, 1)
        major_ticks = np.arange(0,sigma_r0_rel[args.sigma]+0.1,1.0)
        minor_ticks = np.arange(0,sigma_r0_rel[args.sigma]+0.1,0.5)

        text_targets = "Total = " + str(total_occ_gt) + "\n< 1.0m = " + str(round(float((np.sum(solver.histo_distances_targets[0][:11])/total_occ_gt)*100), 1)) + "%\nV0: " + str(args.V0) + "\nsigma: " + str(args.sigma)
        fig, ax = plt.subplots()
        plt.bar(histo_distances_targets[1][:-1], rel_occ_gt, align='edge', width=0.5)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(ticks_y)
        ax.text(0.7, 0.95, text_targets, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
        plt.title("Histogram for distances from ground truth data")
        plt.xlabel("r [m]")
        plt.ylabel("Occurences [%]")
        histo.matplot(fig, win="gt")
        fig.savefig(root+"//"+ground_truth_jpg)
        plt.close(fig)

        histo_distances_pred = np.histogram(solver.distances_outputs, np.arange(0,sigma_r0_rel[args.sigma]+0.1,0.5))
        total_occ_pred = np.sum(solver.histo_distances_outputs[0])
        rel_occ_pred =  histo_distances_pred[0]/total_occ_pred*100

        text_outputs = "Total = " + str(total_occ_pred) + "\n< 1.0m = " + str(round(float((np.sum(solver.histo_distances_outputs[0][:11])/total_occ_pred)*100), 1)) + "%\nV0: " + str(args.V0) + "\nsigma: " + str(args.sigma)
        fig2, ax2 = plt.subplots()
        plt.bar(histo_distances_pred[1][:-1], rel_occ_pred, align='edge', width=0.5)
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.set_yticks(ticks_y)
        ax2.text(0.7, 0.95, text_outputs, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props)
        if "social" in args.model_type:
            plt.title("Histogram for distances from model " + str(args.model_type) + " - NS: " + str(args.neighborhood_size) + " - GS: " + str(args.grid_size))
        else:
            plt.title("Histogram for distances from model " + str(args.model_type))
        plt.xlabel("r [m]")
        plt.ylabel("Occurences [%]")
        histo.matplot(fig2, win=str(args.model_type) + "_" + str(args.V0) + "_" + str(args.sigma) + "_" + str(args.neighborhood_size) + "_" + str(args.grid_size))
        fig2.savefig(root+"//"+model_jpg)
        plt.close(fig2)
        """

# Note results of best validation ADE loss
result_path = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "Results", args.dataset_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)
xlsx_file = args.dataset_name + "_" + args.model_type + ".xlsx"
note_best_val(result_path+"//"+xlsx_file, args, solver.best_val, solver.best_val_FDE, solver.best_val_epoch)

# Note results of testing for Socialforce Experiment
if args.socialforce:
    if args.phase == "test":
        print("Writing test losses for social force experiment to xlsx file...")
        result_path = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "Analyse_Results", "Results")
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        file_name = "socialforce_" + str(args.model_type) + "_results.xlsx"
        note_test_results_for_socialforce(result_path + "//" + file_name, args, solver.loss_history_all[args.dataset_name][args.phase]["G_ADE"][-1], solver.loss_history_all[args.dataset_name][args.phase]["G_FDE"][-1])

        if "G_ADE_nl_regions" in model.losses:
            print("Writing loss in nonlinear Regions to xlsx file...")
            result_path = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "Analyse_Results", "Nonlinear_loss")
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            file_name = "socialforce_nl_loss.xlsx"
            filepath = result_path+"/"+file_name
            note_nonlinear_loss(filepath, args, solver.loss_history_all[args.dataset_name][args.phase]["G_ADE"][-1],  solver.loss_history_all[args.dataset_name][args.phase]["G_ADE_nl_regions"][-1])


# Show plot via Visdom module
if args.visdom:
    first_plot = True
    for dataset in solver.loss_history_all:
        for phase, phase_dict in solver.loss_history_all[dataset].items():
            for loss_type, loss_list in phase_dict.items():
                if loss_type == "G_MSE" or loss_type == "MSE":
                    continue
                if phase.lower() == "train" or phase.lower() == "test":
                    first_plot = True
                if loss_list:
                    if phase.lower() == "train" or phase.lower() == "val":
                        identifier = str(dataset) + " - " + str(args.model_type) + " - " + str(loss_type)
                        if first_plot:
                            viz.line(env=str(args.plot_name), name=str(loss_type) + "_" + str(phase),
                                     win=identifier,Y=np.asarray(loss_list), X=np.arange(len(loss_list)) + 1,
                                     opts=dict(showlegend=True, env=dataset,title=identifier, xlabel="Epochs", ylabel="Loss", layoutopts=dict(plotly=dict(yaxis=dict(range=[0, max(loss_list) * 1.5 + 1])))))
                            first_plot = False
                        else:
                            viz.line(env=str(args.plot_name), name=str(loss_type) + "_" + str(phase), update="append",
                                     win=identifier,Y=np.asarray(loss_list), X=np.arange(len(loss_list)) + 1, opts=dict(showlegend=True))
                    else:
                        print(str(dataset) + "-" + str(phase) + "-" + str(loss_type) + ": " + str(loss_list[-1]))


print("Configs: \nPlot num: " + str(args.plot_name) + " - Dropout: " + str(args.dropout) + " - Batch_size: " + str(args.batch_size) + "\nlr: " + str(args.lr) + " - wd: " + str(args.wd) + " - emb_dim: " + str(args.emb_dim) + " - enc_h_dim: " + str(args.encoder_h_dim) + " - dec_h_dim: " + str(args.decoder_h_dim) +
      " - padding: " + str(args.padding) + " - final_position: " + str(args.final_position))