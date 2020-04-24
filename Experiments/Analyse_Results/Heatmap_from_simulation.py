import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pandas as pd
import numpy as np
import socket
import visdom
import argparse

# Show plot via Visdom module
servername = socket.gethostname()
if "node" in servername:
    server = "http://atcremers10"
else:
    server = 'http://localhost'

viz = visdom.Visdom(server=server, port=8098, env="Results_Simulated")

# Get input arguments from shell
parser = argparse.ArgumentParser("Heatmap_for_results")
parser.add_argument("--model_type", default="", type=str, help="Specify model_type")
parser.add_argument("--ns", default="", type=str, help="Specify neighborhodd size")
parser.add_argument("--gs", default="", type=str, help="Specify grid size")
# Get arguments
args = parser.parse_args()

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

root = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "Analyse_Results", "Results")
if (args.ns != "" or args.gs != "") and args.model_type == "social-lstm":
    file_name = "socialforce_" + str(args.model_type) + "_results_ns_"+str(args.ns)+"_gs_"+str(args.gs)+".xlsx"
    ADE_jpg_name = "socialforce_results_" + str(args.model_type) + "_ADE_ns_"+str(args.ns)+"_gs_"+str(args.gs)+".jpg"
    FDE_jpg_name =  "socialforce_results_" + str(args.model_type)+ "_FDE_ns_"+str(args.ns)+"_gs_"+str(args.gs)+".jpg"
else:
    file_name = "socialforce_" + str(args.model_type) + "_results.xlsx"
    ADE_jpg_name = "socialforce_results_" + str(args.model_type) + "_ADE.jpg"
    FDE_jpg_name = "socialforce_results_" + str(args.model_type) + "_FDE.jpg"

df = pd.read_excel(root + "//" + file_name)

ADE = np.zeros((len(df["V0"].unique()), len(df["sigma"].unique())))
FDE = np.zeros((len(df["V0"].unique()), len(df["sigma"].unique())))

V0_values = np.sort(np.array([(df["V0"].unique())]), axis=-1)
sigma_values = np.sort(np.array([(df["sigma"].unique())]), axis=-1)
V0_values = np.repeat(V0_values, sigma_values.shape[-1], axis=0).transpose()
sigma_values = np.repeat(sigma_values, V0_values.shape[0], axis=0)

for i in range(V0_values.shape[0]):
    for j in range(V0_values.shape[1]):
        ADE[i,j] = df["ADE"][(df["V0"] == V0_values[i,j]) & (df["sigma"] == sigma_values[i,j])]
        FDE[i,j] = df["FDE"][(df["V0"] == V0_values[i,j]) & (df["sigma"] == sigma_values[i,j])]


# Plot Results
min_ADE = 0.0
max_ADE = 0.35

min_FDE = 0.0
max_FDE = 0.75

labels_ADE = np.around(np.linspace(min_ADE, max_ADE, 10), decimals=2)
labels_FDE = np.around(np.linspace(min_FDE, max_FDE, 10), decimals=2)

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
viz.matplot(fig)
print("saving image for ADE...")
fig.savefig(root+"//"+ADE_jpg_name)
plt.close(fig)

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
viz.matplot(fig2)
print("saving image for FDE...")
fig2.savefig(root+"//"+FDE_jpg_name)
plt.close(fig2)