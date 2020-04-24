# ==Imports==
import matplotlib
matplotlib.use('Agg')
import argparse
import os
import logging
import torch
import visdom
import numpy as np
import matplotlib.pyplot as plt
import socket

from data.loader import data_loader
from configs.config_dataprep import *
from helper.bool_flag import bool_flag
from helper.move_opt_stat_to_GPU import move_opt_state_to_GPU
from solver.solver import  Solver
from model.modules import LINEAR, LSTM
#=============

logger = logging.getLogger(__name__)

# Get input arguments from shell
parser = argparse.ArgumentParser("Grid Search for TrajectoryPrediction")

# General Configs
parser.add_argument("--model_type", default="lstm", type=str, help="Define model type. Either Linear or LSTM Model")
parser.add_argument("--dataset_name", default="zara1", type=str, help="Specify dataset")
parser.add_argument("--plot_number", default=1, type=int, help="Specify number for plot")
parser.add_argument("--obs_len", default=8, type=int, help="Specify observed length")
parser.add_argument("--pred_len", default=12, type=int, help="specify predicted length")
parser.add_argument("--data_augmentation", default=0, type=bool_flag, help="Specify whether or not you want to use data augmentation")
parser.add_argument("--optim", default="Adam", type=str, help="specify optimizer")
parser.add_argument("--num_epochs", default=1000, type=int, help="specify number of epochs")
parser.add_argument("--num_searches", default=30, type=int, help="specify number of searches")
parser.add_argument("--skip", default=20, type=int, help="specify skipping rate")
parser.add_argument("--max_num", default=10000, type=int, help="specify maximum number of ids")
parser.add_argument("--viz_port", default=8097, type=int, help="specify port for visdom")
parser.add_argument("--show_traj", default=False, type=bool_flag, help="specify whether to plot trajectories or not")

# Config which Hyperparameters should be searched for
parser.add_argument("--bat_size", default=False, type=bool_flag, help="search for batch size")
parser.add_argument("--dropout", default=False, type=bool_flag, help="search for dropout rate")
parser.add_argument("--num_layers", default=False, type=bool_flag, help="search for number of layers for LSTM")
parser.add_argument("--lr", default=False, type=bool_flag, help="search for lr")
parser.add_argument("--weight_decay", default=False, type=bool_flag, help="search for weight decay")
parser.add_argument("--decoder_h_dim", default=False, type=bool_flag, help="search for decoder_h_dim")
parser.add_argument("--encoder_h_dim", default=False, type=bool_flag, help="search for encoder_h_dim")
parser.add_argument("--emb_dim", default=False, type=bool_flag, help="search for embedding dimension")

# Get arguments
args = parser.parse_args()
args.phase = "train"

# Get Config
config = config()

# Set Parameters for Random Search
N = args.num_searches
current_ADE = 100
parameter_dict = {}

if not args.bat_size:
    args.batch_size = 32
    dset, loader = data_loader(args, config, phase="train", logger=logger)
    dset_val, val_loader = data_loader(args, config, phase="val", logger=logger)

print("Search for " + str(N) + " iterations...")
for i in range(N):
    # Search randomly for Hyperparameters
    if args.bat_size:
        batch_size_exponent = int(np.random.uniform(1,9))
        args.batch_size = 2**batch_size_exponent
    else:
        args.batch_size = 64

    if args.dropout:
        dropout = np.random.uniform(0,1)
    else:
        dropout = 0.0

    if args.num_layers:
        num_layers = int(np.random.uniform(0,4))
    else:
        num_layers = 1

    if args.lr:
        log_learning_rate = int(np.random.uniform(-1,-5))
        lr = 10**log_learning_rate
    else:
        lr = 10**(-4)

    if args.weight_decay:
        wd_factor = np.random.uniform(0,10)
        wd_log = int(np.random.uniform(-3,-6))
        wd = wd_factor*(10**wd_log)
    else:
        wd = 0.03

    if args.decoder_h_dim:
        decoder_h_dim = 2*int(np.random.uniform(1,17))
    else:
        decoder_h_dim = 16

    if args.encoder_h_dim:
        encoder_h_dim = 2*int(np.random.uniform(1,17))
    else:
        encoder_h_dim = 16

    if args.emb_dim:
        embedding_dim = 2*int(np.random.uniform(1,8))
    else:
        embedding_dim = 8

    # Get data
    if args.bat_size:
        dset, loader = data_loader(args, config, phase="train", logger=logger)
        dset_val, val_loader = data_loader(args, config, phase="val", logger=logger)

    # Define model
    if args.model_type.lower() == "linear":
        model = LINEAR(dropoutrate=dropout)
    elif args.model_type.lower() == "lstm":
        model = LSTM(dropout=dropout, num_layers=num_layers, decoder_h_dim=decoder_h_dim, encoder_h_dim=encoder_h_dim, embedding_dim=embedding_dim)
    else:
        raise ValueError("Please choose either Linear or LSTM model_type!")

    # Define optimizer
    if args.optim.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(),**{"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8,"weight_decay": wd})
    elif args.optim.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(),**{"lr": lr, "alpha": 0.99, "eps": 1e-8,"weight_decay": wd})
    else:
        raise ValueError("Please Enter a correct optimizer!")

    # Define things for recording
    loss_history_all = {args.dataset_name: {"train": {}, "val": {}, "test": {}}}
    epochs = 0

    # Define solver
    solver = Solver(optim=optimizer, loss_all=loss_history_all, epochs=epochs, args=args)

    # Train/eval Model
    solver.train(model, args.model_type.lower(), loader, val_loader, phase=args.phase, num_epochs=args.num_epochs)

    print(str(i+1)+" / " +str(N)+ " iterations completed.")

    if current_ADE > np.mean(solver.loss_history_all[args.dataset_name]["val"]["G_ADE"][-int(0.03*args.num_epochs):]):
        current_ADE = np.mean(solver.loss_history_all[args.dataset_name]["val"]["G_ADE"][-int(0.03*args.num_epochs):])
        loss_history_train = solver.loss_history_all[args.dataset_name]["train"]["G_ADE"]
        loss_history_val = solver.loss_history_all[args.dataset_name]["val"]["G_ADE"]
        parameter_dict["lr"] = lr
        parameter_dict["batch_size"] = args.batch_size
        parameter_dict["dropout"] = dropout
        parameter_dict["num_layers"] = num_layers
        parameter_dict["weight_decay"] = wd
        parameter_dict["decoder_h_dim"] = decoder_h_dim
        parameter_dict["encoder_h_dim"] = encoder_h_dim
        parameter_dict["emb_dim"] = embedding_dim
        parameter_dict["val_loss"] = current_ADE

print("====Random Search finished===\n")
print("\nParameters of best training: \n")
for key, value in parameter_dict.items():
    print(str(key)+": " + str(value))
print("plot_number: " + str(args.plot_number))

# plot ADE Loss
servername = socket.gethostname()
if "node" in servername:
    server = "http://atcremers9"
else:
    server = 'http://localhost'

viz = visdom.Visdom(server=server, port=args.viz_port)
viz.line(env="Random Search", name="train_ADE", win=str(args.model_type) + "_" + "_" + str(args.dataset_name) + "_" + str(args.plot_number),
         Y=np.asarray(loss_history_train), X=np.arange(len(loss_history_train)) + 1,
         opts=dict(showlegend=True, title=str(args.dataset_name) + " - " + str(args.model_type) + " - " + str(args.plot_number) + " - Random Search", xlabel="Epochs", ylabel="Loss", layoutopts=dict(
         plotly=dict(yaxis=dict(range=[0, max(loss_history_val) * 1.5 + 1])))))
viz.line(env="Random Search", update="append", name="val_ADE", win=str(args.model_type) + "_" + "_" + str(args.dataset_name) + "_" + str(args.plot_number),
         Y=np.asarray(loss_history_val), X=np.arange(len(loss_history_val)) + 1,
         opts=dict(showlegend=True, title=str(args.dataset_name) + " - " + str(args.model_type) + " - " + str(args.plot_number), xlabel="Epochs", ylabel="Loss", layoutopts=dict(
         plotly=dict(yaxis=dict(range=[0, max(loss_history_val) * 1.5 + 1])))))






