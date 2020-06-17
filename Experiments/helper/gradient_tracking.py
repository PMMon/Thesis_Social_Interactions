import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import visdom


def plot_grad_flow(named_parameters, server, args, epoch):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    """
    if args.visdom:
        vis = visdom.Visdom(server=server, port=args.viz_port, env="Gradient_flow")
    else:
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        image_path = os.path.join(root_path, "Stats", "Gradient_tracking", args.dataset_name)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        image_file = os.path.join(image_path, args.dataset_name + "_" + args.model_type + "_gradtrack_"+ str(args.train_loss)+"_"+str(epoch)+".jpg")

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
    plt.title("Gradient flow " + str(args.train_loss) + " - " + str(epoch) + " - " + str(args.model_type) + " - V0: " + str(args.V0) + "; sigma: " + str(args.sigma))
    ax.yaxis.grid(True)
    plt.legend([Line2D([0], [0], color="green", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    if args.visdom:
        vis.matplot(fig, win="Gradient_tracking_"+str(epoch)+"_"+str(args.train_loss)+"_"+str(args.model_type) + "_" + str(args.V0) + "_" + str(args.sigma))
    else:
        fig.savefig(image_file)

    plt.close(fig)

