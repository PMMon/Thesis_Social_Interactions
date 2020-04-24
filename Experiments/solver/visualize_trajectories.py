import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import os
import numpy as np
import visdom
import torch

def create_visualization(batch, outputs, phase, epoch, batch_number, model_type, server):
    vis = visdom.Visdom(server=server, port=8097, env="trajectories")
    scene = batch["scene_img"][0]["scene"]

    saving_directory = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "Saved_Plots","trajectories", str(scene))
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    scale = 1 / 0.05
    # Get underlying image
    img = batch["scene_img"][0]["scaled_image"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(str(model_type) + "_" + str(phase) + "_" + str(epoch) + "_" + str(scene))
    #if phase == "train":
    #    ax.imshow(img)
    #else:
    fig.set_tight_layout(True)
    #ax.grid(linestyle='dotted')
    ax.set_axisbelow(True)
    #ax.set_xlim([-12*scale, 12*scale])
    #ax.set_ylim([-12*scale, 12*scale])

    for m in ["in_xy", "gt_xy", "out_xy"]:
        if m == "gt_xy" or m == "in_xy":
            marker = '-'
            traj = batch["{}".format(m)][:, 0, :] * scale
        elif m == "out_xy":
            marker = "--"
            traj = outputs["out_xy"][:, 0, :] * scale
        else:
            raise ValueError("Invalid input for marker!")

        traj = traj.cpu().detach().numpy()

        if m == "in_xy":
            writemode = "w"
        else:
            writemode = "a"

        with open(saving_directory+"/" + str(phase) + "_" + str(epoch) + "_" + str(batch_number) + ".txt", writemode) as coordinate_file:
            coordinate_file.write(str(m)+": \n")
            for entry in traj:
                coordinate_file.write(str(entry) + " / ")
            coordinate_file.write("\n")

        ax.plot((traj[:, 0]), (traj[:, 1]), linestyle=marker, linewidth=int(3))

    vis.matplot(fig, win=str(scene) + "_" + str(model_type) + "_" + str(phase) + "_" + str(epoch) + "_" + str(batch_number))
    fig.savefig(saving_directory + "/" + str(scene) + "_" + str(model_type) + "_" + str(phase) + "_" + str(epoch) + "_" + str(batch_number) + ".png")
    print("figure for phase " +  str(phase) + " epoch: " + str(epoch) + " batch: " + str(batch_number) + " saved.")
    plt.close(fig)


def plot_traj_for_social_exp(batch, outputs, phase, epoch, batch_number, model_type, server, args):

    vis = visdom.Visdom(server=server, port=args.viz_port, env="trajectories")

    print("visualize trajectories for Socialforce Experiment...")
    saving_directory = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "Saved_Plots", str(args.dataset_name), str(phase))
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    pred_check = torch.cat(batch["pred_check"], 0)
    pad_mask = torch.cat(batch["pad_mask"], 0)

    nr_sequences = 10

    for seq in range(nr_sequences):
        try:
            ped_start = batch["seq_start_end"][seq][0].item()
            last_ped = batch["seq_start_end"][seq][1].item()
        except:
            return

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(str(model_type) + "_" + str(phase) + "_" + str(seq) + "_" + str(args.dataset_name))
        fig.set_tight_layout(True)
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)

        for ped in range(ped_start, last_ped):
            if pred_check[ped, :] == 0:
                continue
            #c = np.random.rand(1,3).squeeze()
            if ped % 14 == 0:
                c = np.array([0.48424939, 0.63034156, 0.01339784])
            elif ped % 14 == 1:
                c = np.array([0.54037804, 0.23526515, 0.97111372])
            elif ped % 14 == 2:
                c = np.array([0.82024261, 0.7249857 , 0.59582993])
            elif ped % 14 == 3:
                c = np.array([0.57919682, 0.84879728, 0.71940149])
            elif ped % 14 == 4:
                c = np.array([0.39238921, 0.06529021, 0.11736424])
            elif ped % 14 == 5:
                c = np.array([0.59680548, 0.30685758, 0.21610199])
            elif ped % 14 == 6:
                c = np.array([0.08826138, 0.05514283, 0.98096983])
            elif ped % 14 == 7:
                c = np.array([0.21589303, 0.89076467, 0.65568039])
            elif ped % 14 == 8:
                c = np.array([0.40384945, 0.95009665, 0.47855528])
            elif ped % 14 == 9:
                c = np.array([0.82396847, 0.7295973 , 0.36564151])
            elif ped % 14 == 10:
                c = np.array([0.77490801, 0.29139385, 0.31267138])
            elif ped % 14 == 11:
                c = np.array([0.94585025, 0.19175765, 0.73930174])
            elif ped % 14 == 12:
                c = np.array([0.79213388, 0.43784464, 0.41200807])
            elif ped % 14 == 13:
                c = np.array([0.85533775, 0.40813272, 0.87100982])
            else:
                c = np.array([0.85533775, 0.40813272, 0.87100982])


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
                    label = 'ped %s' % str(ped+1)
                    ax.plot((traj[:, 1]), (traj[:, 0]), linestyle=linestyle, marker=marker, color=c, linewidth=int(2), markersize=5) # label=label
                else:
                    ax.plot((traj[:, 1]), (traj[:, 0]), linestyle=linestyle, marker=marker, color=c, linewidth=int(2), markersize=5)

        star = mlines.Line2D([], [], color='black', marker='P', linestyle='None',markersize=8, label='Obs')
        square = mlines.Line2D([], [], color='black', marker='o', linestyle='None',markersize=8, label='GT')
        triangle = mlines.Line2D([], [], color='black', marker='x', linestyle='None',markersize=8, label='Pred')

        plt.legend(handles=[star, square, triangle])

        #plt.legend(loc='best')
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        vis.matplot(fig, win=str(seq) + "_" + str(model_type) + "_" + str(phase) + "_" + str(epoch) + "_" + str(batch_number))
        plt.close(fig)

def visualize_specific_trajectories(target, curv_values, args, N, max_nr_traj):
    """
    Plot trajectories that are assigned to specific group of
    :param target: GT trajectories of shape: pred_length x num_peds_in_batch x 2
    """
    vis = visdom.Visdom(server=args.viz_server, port=args.viz_port, env="Nonlinear_Trajectories")

    max_nr_traj = min(max_nr_traj, target.shape[1])

    marker = 'o'
    linestyle = 'solid'

    for i in range(0, max_nr_traj):
        traj = target[:,i,:].cpu().detach().numpy()
        curv_value = curv_values[i,:].cpu().detach().numpy()

        c = np.random.rand(1, 3).squeeze()

        fig = plt.figure(figsize=(12,5))
        fig.set_tight_layout(True)
        ax = fig.add_subplot(121)
        ax.set_title("Threshold: " + str(args.threshold_nl) + " - N: " + str(N) + " - V0: " + str(args.V0) + " - sigma: " + str(args.sigma))
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)
        ax.plot((traj[:, 1]), (traj[:, 0]), linestyle=linestyle, marker=marker, color=c, linewidth=int(2), markersize=5)
        for i, j, k in zip(traj[1:-1, 1], traj[1:-1, 0], curv_value):
            ax.annotate(str("%.2f"%(k)), xy=(i, j))
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.ylim([traj[:, 0].min()-0.5, traj[:, 0].max()+0.5])
        plt.xlim([traj[:, 1].min()-0.5, traj[:, 1].max()+0.5])

        ax = fig.add_subplot(122)
        ax.set_title("Threshold: " + str(args.threshold_nl) + " - N: " + str(N) + " - V0: " + str(args.V0) + " - sigma: " + str(args.sigma))
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)
        ax.plot((traj[:, 1]), (traj[:, 0]), linestyle=linestyle, marker=marker, color=c, linewidth=int(2), markersize=5)
        ax.set_autoscaley_on(False)
        ax.set_xlim([-11, 11])
        ax.set_ylim([-11, 11])

        plt.xlabel("x [m]")

        vis.matplot(fig, win=str(args.threshold_nl) + "_" + str(N) + "_" + str(args.V0) + "_" + str(args.sigma) + str(i))
        plt.close(fig)

def visualize_grouped_trajectories(target, curv_values, args, max_nr_traj, group, sum_mask, N=0):
    """
    Plot trajectories that are assigned to specific group of
    :param target: GT trajectories of shape: pred_length x num_peds_in_batch x 2
    """
    vis = visdom.Visdom(server=args.viz_server, port=args.viz_port, env="Nonlinear_Trajectories")

    max_nr_traj = min(max_nr_traj, target.shape[1])

    marker = 'o'
    linestyle = 'solid'

    for i in range(0, max_nr_traj):
        traj = target[:,i,:].cpu().detach().numpy()
        curv_value = curv_values[i,:].cpu().detach().numpy()

        c = np.random.rand(1, 3).squeeze()

        fig = plt.figure(figsize=(12,5))
        fig.set_tight_layout(True)
        ax = fig.add_subplot(121)
        if group == "strictly_linear":
            title_curvature = " k < 0.11"
        elif group == "linear":
            title_curvature = " k < 0.11"
        elif group == "med_nonlinear":
            title_curvature = " 0.2 <= k < 0.7"
        elif group == "highly_nonlinear":
            title_curvature = " 1.0 <= k"
        elif group == "other":
            title_curvature = ""
        else:
            title_curvature = " undefined"

        title = "Group: " + group + title_curvature + " - V0: " + str(args.V0) + " - sigma: " + str(args.sigma) #  " - N: " + str(int(sum_mask[i].item())) +
        ax.set_title(title)
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)
        ax.plot((traj[:, 1]), (traj[:, 0]), linestyle=linestyle, marker=marker, color=c, linewidth=int(2), markersize=5)
        for i, j, k in zip(traj[1:-1, 1], traj[1:-1, 0], curv_value):
            ax.annotate(str("%.2f"%(k)), xy=(i, j))
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
        ax.set_xlim([-11, 11])
        ax.set_ylim([-11, 11])

        plt.xlabel("x [m]")

        vis.matplot(fig, win=str(args.threshold_nl) + "_" + str(N) + "_" + str(args.V0) + "_" + str(args.sigma) + str(i))
        plt.close(fig)