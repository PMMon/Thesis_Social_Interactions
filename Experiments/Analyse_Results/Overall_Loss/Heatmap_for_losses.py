import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
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
# Script to create a heatmap for the ADE and FDE of the model --model_type for various synthetic datasets.
#
# Input:
# - input-file: socialforce_model_type_results.xlsx -> captures ADE and FDE of model for different datasets
#
# Output:
# - heatmap for ADE: model_type/socialforce_results_model_type_ADE.jpg
# - heatmap for FDE: model_type/socialforce_results_model_type_FDE.jpg
# =========================================================================================================

# Get input arguments from shell
parser = argparse.ArgumentParser("Heatmap_for_results")

# Model specification
parser.add_argument("--model_type", default="", type=str, help="Specify model type")
parser.add_argument("--ns", default="", type=str, help="Specify neighborhood size")
parser.add_argument("--gs", default="", type=str, help="Specify grid size")

# Visdom configurations
parser.add_argument("--visdom", default=False, type=bool_flag, help="Specify whether to plot via visdom")
parser.add_argument("--viz_port", default=8090, type=int, help="Specify port for visdom")
parser.add_argument("--viz_server", default="", type=str, help="Specify server for visdom")
parser.add_argument("--viz_env", default="EvalMetrics_OverallLoss", type=str, help="Specify environment for visdom")

# Get arguments
args = parser.parse_args()

# Show plot via Visdom module
servername = socket.gethostname()
if "node" in servername:
    server = args.viz_server
else:
    server = 'http://localhost'

viz = visdom.Visdom(server=server, port=args.viz_port, env=args.viz_env)

# Get model type
if args.model_type == "":
    raise ValueError("please insert valid model_type!")
else:
    if args.model_type == "social-lstm":
        if args.ns == "":
            raise ValueError("please insert valid neighborhood size!")
        else:
            pass
        if args.gs == "":
            raise ValueError("please insert valid grid size!")
        else:
            pass

# Specify file paths
root_path = os.path.abspath(os.path.dirname(__file__))
root_path_losses = root_path
root_path_heatmap = os.path.join(root_path, args.model_type)


# Specify file name
if (args.ns != "" or args.gs != "") and args.model_type == "social-lstm":
    file_name_losses = "socialforce_" + str(args.model_type) + "_results_ns_"+str(args.ns)+"_gs_"+str(args.gs)+".xlsx"
    ADE_jpg_name = "socialforce_results_" + str(args.model_type) + "_ADE_ns_"+str(args.ns)+"_gs_"+str(args.gs)+".jpg"
    FDE_jpg_name =  "socialforce_results_" + str(args.model_type)+ "_FDE_ns_"+str(args.ns)+"_gs_"+str(args.gs)+".jpg"
else:
    file_name_losses = "socialforce_" + str(args.model_type) + "_results.xlsx"
    ADE_jpg_name = "socialforce_results_" + str(args.model_type) + "_ADE.jpg"
    FDE_jpg_name = "socialforce_results_" + str(args.model_type) + "_FDE.jpg"

# Read in losses of model
df = pd.read_excel(os.path.join(root_path_losses, file_name_losses))

ADE = np.zeros((len(df["V0"].unique()), len(df["sigma"].unique())))
FDE = np.zeros((len(df["V0"].unique()), len(df["sigma"].unique())))

# Prepare grid
V0_values = np.sort(np.array([(df["V0"].unique())]), axis=-1)
sigma_values = np.sort(np.array([(df["sigma"].unique())]), axis=-1)
V0_values = np.repeat(V0_values, sigma_values.shape[-1], axis=0).transpose()
sigma_values = np.repeat(sigma_values, V0_values.shape[0], axis=0)

# Note losses for values
for i in range(V0_values.shape[0]):
    for j in range(V0_values.shape[1]):
        ADE[i,j] = df["ADE"][(df["V0"] == V0_values[i,j]) & (df["sigma"] == sigma_values[i,j])]
        FDE[i,j] = df["FDE"][(df["V0"] == V0_values[i,j]) & (df["sigma"] == sigma_values[i,j])]

# Plot Results
min_ADE = np.min(ADE)
max_ADE = np.max(ADE) * 1.1

min_FDE = np.min(FDE)
max_FDE = np.max(FDE) * 1.1

labels_ADE = np.around(np.linspace(min_ADE, max_ADE, 10), decimals=2)
labels_FDE = np.around(np.linspace(min_FDE, max_FDE, 10), decimals=2)

# Plot ADE
fig, ax = plt.subplots()
CS2 = plt.contourf(sigma_values, V0_values, ADE, 100, cmap=cm.jet, vmin=min_ADE, vmax=max_ADE)
cl2 = plt.clabel(CS2, inline=0, fontsize=0, linewidth=0)
cbar = plt.colorbar(ticks=labels_ADE)
if (args.ns != "" or args.gs != "") and args.model_type == "social-lstm":
    plt.title("ADE of " + str(args.model_type) + " - Socialforces - ns " + str(args.ns) + " - gs " + str(args.gs))
else:
    plt.title("ADE of " + str(args.model_type) + " - Socialforces")
plt.xlabel(r'$\sigma$') # r [m] for V(r)= 10% of V0
plt.ylabel(r'$V^{0}$')

if args.visdom:
    print("Plot image for ADE in visdom...")
    viz.matplot(fig)
else:
    print("Saving image for ADE...")
    if not os.path.exists(root_path_heatmap):
        os.makedirs(root_path_heatmap)
    fig.savefig(os.path.join(root_path_heatmap, ADE_jpg_name))

plt.close(fig)

# Plot FDE
fig2, ax2 = plt.subplots()
CS1 = plt.contourf(sigma_values, V0_values, FDE, 100, cmap=cm.jet, vmin=min_FDE, vmax=max_FDE)
cl1 = plt.clabel(CS1, inline=0, fontsize=0, linewidth=0)
plt.colorbar(ticks=labels_FDE)
if (args.ns != "" or args.gs != "") and args.model_type == "social-lstm":
    plt.title("FDE of " + str(args.model_type) + " - Socialforces - ns " + str(args.ns) + " - gs " + str(args.gs))
else:
    plt.title("FDE of " + str(args.model_type) + " - Socialforces")
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$V^{0}$')

if args.visdom:
    print("Plot image for FDE in visdom...")
    viz.matplot(fig2)
else:
    print("Saving image for FDE...")
    fig2.savefig(os.path.join(root_path_heatmap, FDE_jpg_name))

plt.close(fig2)