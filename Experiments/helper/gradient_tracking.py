import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import visdom


def plot_grad_flow(named_parameters, server, args, epoch):
    '''
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(model.named_parameters(), self.server, self.args, epoch)" to visualize the gradient flow
    '''
    vis = visdom.Visdom(server=server, port=args.viz_port, env="gradient_flow")

    result_path = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "Results", args.dataset_name, "Gradient_Tracking")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    jpg_file = result_path + "/" + args.dataset_name + "_" + args.model_type + "_gradtrack_"+str(args.train_loss)+"_"+str(epoch)+".jpg"

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig = plt.figure(figsize=(6,7))
    ax = plt.axes()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.6, lw=1, facecolor="green", edgecolor="g") #c
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.6, lw=1, facecolor="blue", edgecolor="r")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.tight_layout()
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow " + str(args.train_loss) + " - " + str(epoch) + " - " + str(args.model_type) + " - V0: " + str(args.V0) + ";sigma: " + str(args.sigma))
    ax.yaxis.grid(True)
    plt.legend([Line2D([0], [0], color="green", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    vis.matplot(fig,win="gradient_tracking_"+str(epoch)+"_"+str(args.train_loss)+"_"+str(args.model_type) + "_" + str(args.V0) + "_" + str(args.sigma))
    fig.savefig(jpg_file)
    plt.close(fig)

