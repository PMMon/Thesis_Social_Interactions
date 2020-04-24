import torch
import platform
import numpy as np
import math

from solver.losses import L2, FDE, Scene_distances, ADE_nl_regions, nonlinear_loss, own_MSE, ADE_nl_Trajectories_fine, ADE_nl_Trajectories_coarse
from solver.visualize_trajectories import create_visualization, plot_traj_for_social_exp
from helper.calc_distance import calculate_distance
from helper.gradient_tracking import plot_grad_flow
from helper.histogram import create_histogram
from helper.results_to_xlxs import loss_nonlinear_trajectories, loss_nonlinear_trajectories_coarse

class Solver(object):
    def __init__(self, optim,  loss_all, epochs, args, server, loss_func=torch.nn.MSELoss()):
        self.optim = optim
        self.loss_func = loss_func
        self.loss_history_all = loss_all
        self.last_epoch = epochs
        self.dataset_name = args.dataset_name
        self.show_traj = args.show_traj
        self.args = args
        self.server = server
        self.best_val = 100
        self.best_val_FDE = 0
        self.best_val_epoch = 0

        self.distances_targets = np.array([])
        self.distances_outputs = np.array([])
        self.histo_distances_targets = np.array([])
        self.histo_distances_outputs = np.array([])
        self.histo_curvatues = np.array([])


    def reset_histories(self):
        self.loss_history = []

    def train(self, model, model_type, train_loader, val_loader, phase, num_epochs, log_nth=0, visualize=False, train_epoch = 0):
        """
                Train a given model with the provided data.
                Inputs:
                - model: model object initialized from a torch.nn.Module
                - train_loader: train data in torch.utils.data.DataLoader
                - val_loader: val data in torch.utils.data.DataLoader
                - num_epochs: total number of training epochs
                - log_nth: log training accuracy and loss every nth iteration
        """

        self.reset_histories()
        iter_per_epoch = len(train_loader)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if phase != "val":
            print('Using python: ', platform.python_version())
            print('Using device: ', device)

        model.to(device)

        if phase != "val":
            print("==START "+str(phase).upper()+"==")

        for epoch in range(num_epochs):
            # Initialize general losses
            L2_loss = L2()
            FDE_loss = FDE()

            #Inititalize Additional losses
            if "G_distance_loss" in model.losses:
                distance_loss = Scene_distances()
            if "G_ADE_nl_regions" in model.losses:
                ADE_nl = ADE_nl_regions(threshold=self.args.threshold_nl, approx_scheme=self.args.approx_scheme, p=2)
            if "G_NL" in model.losses:
                non_linear_loss = nonlinear_loss()
            if "G_MSE" in model.losses:
                customized_MSE = own_MSE(S=2)
            # Create new loss for fine grouped nonlinear trajectories
            if self.args.nl_fine:
                if phase == "test":
                    print("Analyze trajectories in classify them in fine groups...")
                    nonlinear_trajectories_dict = {}
                    for N in range(0,11):
                        nonlinear_trajectories_dict[N] = ADE_nl_Trajectories_fine(args=self.args, p=2)

            # Create new loss for coarse grouped nonlinear trajectories
            if self.args.nl_coarse:
                if phase == "test":
                    print("Analyze trajectories in classify them in coarse groups...")
                    groups = ["strictly_linear", "linear", "med_nonlinear", "highly_nonlinear", "other"]
                    nonlinear_trajectories_dict_coarse = {}
                    for group in groups:
                        nonlinear_trajectories_dict_coarse[group] = ADE_nl_Trajectories_coarse(args=self.args, p=2, group=group)



            if phase == "train":
                model.train()
            elif phase == "test" or phase == "val":
                model.eval()
            else:
                raise ValueError("Specify phase either train or test!" )

            for i, batch in enumerate(train_loader, 1):
                in_xy, gt_xy = batch["in_xy"].to(device), batch["gt_xy"].to(device)
                in_dxdy = batch["in_dxdy"].to(device)

                if "seq_start_end" in batch.keys():
                    seq_start_end = batch["seq_start_end"].to(device)
                    pred_check = torch.cat(batch["pred_check"], 0)
                    inputs = {"in_xy": in_xy, "in_dxdy": in_dxdy, "seq_start_end": seq_start_end, "pred_check": pred_check}

                    # Account for Padding in ground truth
                    if pred_check.shape[0] <= 1:
                        if pred_check[0] > 0:
                            gt_xy = gt_xy[:, 0:1,:]
                        else:
                            continue
                    else:
                        gt_xy = gt_xy[:, pred_check.squeeze() > 0,:]
                else:
                    inputs = {"in_xy": in_xy, "in_dxdy": in_dxdy}

                if self.args.final_position:
                    inputs["gt"] = batch["gt_xy"].to(device)

                targets = {"gt_xy": gt_xy}

                self.optim.zero_grad()
                outputs = model(inputs)

                # Define losses for batch
                if "G_L2" in model.losses:
                    if "G_L2" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_L2"] = []
                    L2_batch_loss = L2_loss.loss_batch(outputs["out_xy"], targets["gt_xy"])

                if "G_ADE" in model.losses:
                    if "G_ADE" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_ADE"] = []
                    # Already calculated for batch (see L2)

                if "G_FDE" in model.losses:
                    if "G_FDE" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_FDE"] = []
                    FDE_batch_loss = FDE_loss.loss_batch(outputs["out_xy"][-1], targets["gt_xy"][-1])

                if "G_AV" in model.losses:
                    if "G_AV" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_AV"] = []

                if "G_MSE" in model.losses:
                    if "G_MSE" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_MSE"] = []
                    #loss_MSE = self.loss_func(outputs["out_xy"], targets["gt_xy"])
                    loss_MSE = customized_MSE(outputs["out_xy"], targets["gt_xy"])

                # additional loss based distances between peds in scene
                if "G_distance_loss" in model.losses:
                    if "G_distance_loss" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_distance_loss"] = []
                    loss_dist = distance_loss.calc_distance_loss(outputs["out_xy_all"].to(device), batch["gt_xy"].to(device), batch["seq_start_end"], batch["pred_check"], batch["pad_mask"])

                if "G_ADE_nl_regions" in model.losses:
                    if "G_ADE_nl_regions" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_ADE_nl_regions"] = []
                    loss_ADE_nl = ADE_nl(outputs["out_xy"], targets["gt_xy"])

                if "G_NL" in model.losses:
                    if "G_NL" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_NL"] = []
                    nl_loss = non_linear_loss(outputs["out_xy"], targets["gt_xy"])

                # Create new loss for grouped nonlinear trajectories
                if self.args.nl_fine:
                    if phase == "test":
                        for N in nonlinear_trajectories_dict:
                            nonlinear_ADE_trajectories_fine = nonlinear_trajectories_dict[N](outputs["out_xy"], targets["gt_xy"], N)

                if self.args.nl_coarse:
                    if phase == "test":
                        for group in groups:
                            nonlinear_ADE_trajectories_coarse = nonlinear_trajectories_dict_coarse[group](outputs["out_xy"], targets["gt_xy"])


                # Backprop for training with L2 loss
                if phase == "train":
                    if self.args.train_loss == "ADE":
                        L2_batch_loss.backward()
                    elif self.args.train_loss == "nl_loss":
                        nl_loss.backward()
                    elif self.args.train_loss == "MSE":
                        loss_MSE.backward()
                    elif self.args.train_loss == "ADE_nl":
                        loss_ADE_nl.backward()
                    elif self.args.train_loss == "mixed":
                        (L2_batch_loss+loss_ADE_nl).backward()
                    else:
                        print("Specify train_loss correctly!")
                        quit()

                    self.optim.step()

                    if self.args.plot_gradient_flow:
                        if epoch%50 == 0 or epoch == num_epochs-1:
                            plot_grad_flow(model.named_parameters(), self.server, self.args, epoch)

                # Calculate distances to surrounding pedestrians
                if phase == "test":
                    if self.args.histo:
                        if "seq_start_end" in batch.keys():
                            print("calculating distances...")
                            self.distances_targets = np.concatenate((self.distances_targets, calculate_distance(batch["gt_xy"], batch["seq_start_end"], batch["pred_check"], batch["pad_mask"], self.args)), axis=0)
                            self.distances_outputs = np.concatenate((self.distances_outputs, calculate_distance(outputs["out_xy_all"].detach().cpu().numpy(), batch["seq_start_end"], batch["pred_check"], batch["pad_mask"], self.args)), axis=0)

                # Note MSE for averaging over batch later
                #self.loss_history.append(loss_MSE.detach().cpu().numpy())

                # Plot last 5 epochs
                if self.show_traj:
                    if self.args.socialforce:
                        if phase=="test":
                            plot_traj_for_social_exp(batch, outputs, phase, train_epoch, i, model_type, self.server, self.args)
                    else:
                        if epoch >= num_epochs-5 and (phase=="train" or phase=="test"):
                            visualize = True
                            train_epoch = epoch
                        if visualize:
                            if phase == "train" or phase == "test":
                                create_visualization(batch, outputs, phase, train_epoch, i, model_type, self.server)

            # Calculate losses for Epoch
            if "G_ADE" in model.losses:
                self.loss_history_all[self.dataset_name][str(phase)]["G_ADE"].append(L2_loss.loss_epoch().detach().cpu().numpy())
            if "G_FDE" in model.losses:
                self.loss_history_all[self.dataset_name][str(phase)]["G_FDE"].append(FDE_loss.loss_epoch().detach().cpu().numpy())
            if "G_AV" in model.losses:
                self.loss_history_all[self.dataset_name][str(phase)]["G_AV"].append((self.loss_history_all[self.dataset_name][str(phase)]["G_ADE"][-1]+ self.loss_history_all[self.dataset_name][str(phase)]["G_FDE"][-1])/2)
            if "G_MSE" in model.losses:
                self.loss_history_all[self.dataset_name][str(phase)]["G_MSE"].append(customized_MSE.loss_epoch().detach().cpu().numpy())
            if "G_distance_loss" in model.losses:
                self.loss_history_all[self.dataset_name][str(phase)]["G_distance_loss"].append(distance_loss.loss_epoch().detach().cpu().numpy())
            if "G_NL" in model.losses:
                self.loss_history_all[self.dataset_name][str(phase)]["G_NL"].append(non_linear_loss.loss_epoch().detach().cpu().numpy())
            if "G_ADE_nl_regions" in model.losses:
                self.loss_history_all[self.dataset_name][str(phase)]["G_ADE_nl_regions"].append(ADE_nl.loss_epoch().detach().cpu().numpy())

            # Create new loss for fine grouped nonlinear trajectories
            if self.args.nl_fine:
                if phase == "test":
                    loss_dict = {}
                    for N in nonlinear_trajectories_dict:
                        loss_dict[N] = {}
                        loss_dict[N]["ADE_nonlinear"] = nonlinear_trajectories_dict[N].loss_epoch().detach().cpu().numpy()
                        loss_dict[N]["nr_traj"] = nonlinear_trajectories_dict[N].num_trajectories


                    path = "Analyse_Results//Nonlinear_loss//Classified_Trajectories//Fine//" + self.args.dataset_name
                    loss_nonlinear_trajectories(path, self.args, loss_dict)

            # Create new loss for fine grouped nonlinear trajectories
            if self.args.nl_coarse:
                if phase == "test":
                    loss_dict_coarse = {}
                    for group in nonlinear_trajectories_dict_coarse:
                        loss_dict_coarse[group] = {}
                        loss_dict_coarse[group]["ADE_nonlinear"] = nonlinear_trajectories_dict_coarse[group].loss_epoch().detach().cpu().numpy()
                        loss_dict_coarse[group]["FDE_nonlinear"] = nonlinear_trajectories_dict_coarse[group].FDE_loss_epoch().detach().cpu().numpy()
                        loss_dict_coarse[group]["nr_traj"] = nonlinear_trajectories_dict_coarse[group].num_trajectories
                        loss_dict_coarse[group]["total_nr_traj"] = nonlinear_trajectories_dict_coarse[group].total_num_trajectories

                    path = "Analyse_Results//Nonlinear_loss//Classified_Trajectories//Coarse//" + self.args.dataset_name
                    loss_nonlinear_trajectories_coarse(path, self.args, loss_dict_coarse)


            # Add MSE
            #batch_size_losses = self.loss_history[-len(train_loader):]
            #MSE_loss = np.mean(batch_size_losses)
            #self.loss_history_all[self.dataset_name][str(phase)]["MSE"].append(MSE_loss)

            # adjust threshold according to mean curvature for training
            if phase == "train" and (self.args.train_loss == "mixed" or self.args.train_loss == "ADE_nl") and "G_ADE_nl_regions" in model.losses and epoch == 0:
                self.histo_curvatues = ADE_nl.get_curvatures()
                self.args.threshold = self.histo_curvatues.mean()

            # Check for validation loss to be lowest
            if phase == "val":
                L2_loss = L2_loss.loss_epoch().detach().cpu().numpy()
                if L2_loss < self.best_val:
                    self.best_val = L2_loss
                    self.best_val_FDE = FDE_loss.loss_epoch().detach().cpu().numpy()
                    self.best_val_epoch = train_epoch+1

            # Create histogram for curvatures
            if phase == "test" and "G_ADE_nl_regions" in model.losses and epoch == 0:
                print("Create histogram for curvature values in dataset.")
                self.histo_curvatues = ADE_nl.get_curvatures()
                bins = np.arange(self.histo_curvatues.min(), self.histo_curvatues.max(), 0.1)
                hist = create_histogram(args=self.args, server=self.server)
                hist.plot_histogram_curvatures(self.histo_curvatues, bins)

            # Print Epoch loss
            if log_nth:
                print('[Epoch %d/%d] ' %(epoch + 1, num_epochs) + str(phase) + ' loss: %.3f' % (self.loss_history_all[self.dataset_name][str(phase)]["G_ADE"][-1]))
                #print('[Epoch %d/%d] nonlinear_loss: ' % (epoch + 1, num_epochs) + str(phase) + ' loss: %.3f' % (self.loss_history_all[self.dataset_name][str(phase)]["G_NL"][-1]))


            # Validation
            if phase == "train":
                self.train(model, model_type, val_loader, val_loader, log_nth = log_nth, num_epochs=1, phase="val", visualize=visualize, train_epoch = epoch)

        if phase == "test":
            if self.args.histo:
                self.histo_distances_targets = np.histogram(self.distances_targets, np.arange(0, 30.1, 0.1))
                self.histo_distances_outputs = np.histogram(self.distances_outputs, np.arange(0, 30.1, 0.1))

        if phase != "val":
            self.last_epoch += num_epochs
            print("finish")


