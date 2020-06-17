import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import os
import numpy as np
import visdom
import torch

def plot_predictions_background_img(batch, outputs, phase, epoch, batch_number, model_type, server, args):
    """
    Plots observations, predictions and ground truth trajectories for experiments on real datasets for which a suitable
    background image of the scenario is available.
    Note: this is currently only implemented for data that was loaded with ./data/trajectories.py
    :param batch: Batch-dictionary containing information about observations and ground truth data
    :param outputs: Outputs-dictionary containing information about predicted trajectories
    :param phase: Either train, val or test
    :param epoch: Number of epoch
    :param batch_number: Number of batch
    :param model_type: Model type. At current state of implementation either linear, lstm or social-lstm
    :param server: Visdom server specification. Only relevant if flag --visdom is set to True
    :param args: Command line input-arguments
    """
    print("Visualize trajectories for scenario: %s..." % (args.dataset_name))
    if args.visdom:
        vis = visdom.Visdom(server=server, port=args.viz_port, env="Trajectories_RealDset")
    else:
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        saving_directory = os.path.join(root_path, "Stats", "Trajectories", str(args.dataset_name), str(model_type), str(phase))
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)

    # Get underlying image of scenario
    scene = batch["scene_img"][0]["scene"]
    img = batch["scene_img"][0]["scaled_image"]

    # Specify scaling factor to map background scene image
    scale = 1 / 0.05

    # Specify for how many trajectories you want to visualize in the batch (note that since data was loaded with ./data/trajectories.py, each trajectory is
    # processed independently to the respective scene)
    nr_traj = 10

    if nr_traj > batch["in_xy"].shape[1]:
        print("Specified number of trajectories exceeds available number of trajectories in batch!\nSet number of trajectories to maximal number of trajectories available.")
        nr_traj = batch["in_xy"].shape[1]

    # Create Plot
    for nr in range(nr_traj):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(str(model_type) + "_" + str(phase) + "_batch_" + str(batch_number) + "_" + str(nr) + "_" + str(args.dataset_name))
        fig.set_tight_layout(True)
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)

        # Show image behind scene
        ax.imshow(img)

        for m in ["in_xy", "gt_xy", "out_xy"]:
            if m == "in_xy":
                marker = 'P'
                linestyle = 'solid'
                traj = batch["{}".format(m)][:, nr, :] * scale

            elif m == "gt_xy":
                marker = 'o'
                linestyle = 'solid'
                traj = batch["{}".format(m)][:, nr, :] * scale

            elif m == "out_xy":
                marker = 'x'
                linestyle = 'dashed'
                traj = outputs["out_xy"][:, nr, :] * scale

            else:
                raise ValueError("Invalid input for marker!")

            traj = traj.cpu().detach().numpy()

            ax.plot((traj[:, 0]), (traj[:, 1]), linestyle=linestyle, marker=marker, linewidth=int(2), markersize=5)

        star = mlines.Line2D([], [], color='black', marker='P', linestyle='None', markersize=8, label='Obs')
        square = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='GT')
        triangle = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=8, label='Pred')

        plt.legend(handles=[star, square, triangle])

        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        if args.visdom:
            vis.matplot(fig, win= str(nr) + "_" + str(scene) + "_" + str(model_type) + "_" + str(phase) + "_" + str(epoch) + "_" + str(batch_number))
        else:
            image_file = os.path.join(saving_directory, "traj_" + str(nr) + "_" + str(model_type) + "_" + str(phase) + "_epoch_" + str(epoch) + "_batch_" + str(batch_number) + ".jpg")
            fig.savefig(image_file)

        plt.close(fig)


def plot_predictions(batch, outputs, phase, epoch, batch_number, model_type, server, args):
    """
    Plots observations, predictions and ground truth trajectories for experiments on real and synthetic datasets.
    One plot for each sequence is generated. Specify number of sequences you want to plot with variable nr_sequences.
    :param batch: Batch-dictionary containing information about observations and ground truth data
    :param outputs: Outputs-dictionary containing information about predicted trajectories
    :param phase: Either train, val or test
    :param epoch: Number of epoch
    :param batch_number: Number of batch
    :param model_type: Model type. At current state of implementation either linear, lstm or social-lstm
    :param server: Visdom server specification. Only relevant if flag --visdom is set to True
    :param args: Command line input-arguments
    """
    print("Visualize trajectories for dataset: %s..." % (args.dataset_name))

    if args.visdom:
        vis = visdom.Visdom(server=server, port=args.viz_port, env="Trajectories")
    else:
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        saving_directory = os.path.join(root_path, "Stats", "Trajectories", str(args.dataset_name), str(model_type), str(phase))
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)

    pred_check = torch.cat(batch["pred_check"], 0)
    pad_mask = torch.cat(batch["pad_mask"], 0)

    # Specify for how many sequences you want to visualize in the batch
    nr_sequences = 10

    if nr_sequences > len(batch["seq_start_end"]):
        print("Specified number of sequences exceeds available number of sequences!\nSet number of sequences to maximal number of sequences available.")
        nr_sequences = len(batch["seq_start_end"])

    for seq in range(nr_sequences):
        try:
            ped_start = batch["seq_start_end"][seq][0].item()
            last_ped = batch["seq_start_end"][seq][1].item()
        except:
            return

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(str(model_type) + "_" + str(phase) + "_batch_" + str(batch_number) + "_" + str(seq) + "_" + str(args.dataset_name))
        fig.set_tight_layout(True)
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)

        for ped in range(ped_start, last_ped):
            # Skip padded trajectories
            if pred_check[ped, :] == 0:
                continue

            # Specify color of trajectory
            c = np.random.rand(1,3).squeeze()

            # Specify marker and linestyle of trajectory-part and plot trajectory
            for m in ["in_xy", "gt_xy", "out_xy"]:
                if m == "in_xy":
                    marker = 'P'
                    linestyle = 'solid'
                    traj = batch["{}".format(m)][:, ped, :]
                    if pred_check[ped, :] == 0:
                        indices = pad_mask[ped,:8]
                        traj = traj[indices>0]

                elif m == "gt_xy":
                    marker = 'o'
                    linestyle = 'solid'
                    traj = batch["{}".format(m)][:, ped, :]
                    if pred_check[ped, :] == 0:
                        indices = pad_mask[ped,8:]
                        traj = traj[indices>0]

                elif m == "out_xy":
                    marker = 'x'
                    linestyle = 'dashed'
                    traj = outputs["out_xy_all"][:, ped, :]
                    if pred_check[ped, :] == 0:
                        indices = pad_mask[ped,8:]
                        traj = traj[indices>0]
                else:
                    raise ValueError("Invalid input for marker!")

                traj = traj.cpu().detach().numpy()

                if m == "gt_xy":
                    ax.plot((traj[:, 0]), (traj[:, 1]), linestyle=linestyle, marker=marker, color=c, linewidth=int(2), markersize=5)
                else:
                    ax.plot((traj[:, 0]), (traj[:, 1]), linestyle=linestyle, marker=marker, color=c, linewidth=int(2), markersize=5)

        star = mlines.Line2D([], [], color='black', marker='P', linestyle='None',markersize=8, label='Obs')
        square = mlines.Line2D([], [], color='black', marker='o', linestyle='None',markersize=8, label='GT')
        triangle = mlines.Line2D([], [], color='black', marker='x', linestyle='None',markersize=8, label='Pred')

        plt.legend(handles=[star, square, triangle])

        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        if args.visdom:
            vis.matplot(fig, win=str(seq) + "_" + str(model_type) + "_" + str(phase) + "_" + str(epoch) + "_" + str(batch_number))
        else:
            image_file = os.path.join(saving_directory, "seq_" + str(seq) + "_" + str(model_type) + "_epoch_" + str(epoch) + "_batch_" + str(batch_number) + ".jpg")
            fig.savefig(image_file)

        plt.close(fig)



def visualize_classified_trajectories(target, curv_values, args, max_nr_traj, group, sum_mask, N=0):
    """
    Plot trajectories that are assigned to specific trajectory-class
    :param target: GT trajectories of shape: pred_length x num_peds_in_batch x 2
    """
    print("Visualize trajectories for group: %s..." % (group))

    if args.visdom:
        vis = visdom.Visdom(server=args.viz_server, port=args.viz_port, env="EvalMetrics_ClassLoss_Traj_" + str(group))
    else:
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        saving_directory = os.path.join(root_path, "Analyse_Results", "ClassLoss", args.dataset_name, "Trajectories", str(group))
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)

    max_nr_traj = min(max_nr_traj, target.shape[1])

    marker = 'o'
    linestyle = 'solid'

    # Create plot
    for i in range(0, max_nr_traj):
        traj = target[:,i,:].cpu().detach().numpy()
        curv_value = curv_values[i,:].cpu().detach().numpy()

        c = np.random.rand(1, 3).squeeze()

        fig = plt.figure(figsize=(12,5))
        fig.set_tight_layout(True)
        ax = fig.add_subplot(121)

        title = "Group: " + group +  " - V0: " + str(args.V0) + " - sigma: " + str(args.sigma)
        ax.set_title(title)
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)
        ax.plot((traj[:, 1]), (traj[:, 0]), linestyle=linestyle, marker=marker, color=c, linewidth=int(2), markersize=5)
        for m, n, k in zip(traj[1:-1, 1], traj[1:-1, 0], curv_value):
            ax.annotate(str("%.2f"%(k)), xy=(m, n))
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.ylim([traj[:, 0].min()-0.5, traj[:, 0].max()+0.5])
        plt.xlim([traj[:, 1].min()-0.5, traj[:, 1].max()+0.5])

        ax = fig.add_subplot(122)
        ax.set_title(title)
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)
        ax.plot((traj[:, 1]), (traj[:, 0]), linestyle=linestyle, marker=marker, color=c, linewidth=int(2), markersize=5)
        ax.set_autoscaley_on(False)
        # Below could need adjustment - currently manually adjusted according to scene-size (square)
        a = 20
        ax.set_xlim([-a/2 -1, a/2 + 1])
        ax.set_ylim([-a/2 -1, a/2 + 1])

        plt.xlabel("x [m]")

        if args.visdom:
            vis.matplot(fig, win= "traj_" + str(i) + "_group_" + str(group))
        else:
            plot_file = os.path.join(saving_directory, "traj_" + str(i) + "_group_" + str(group) + ".jpg")
            fig.savefig(plot_file)

        plt.close(fig)