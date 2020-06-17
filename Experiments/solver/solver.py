import torch
import platform
import numpy as np
import os

from solver.losses import L2, FDE, Scene_distances, ADE_nl_regions, nonlinear_loss, own_MSE, nl_loss_classified_Trajectories
from solver.visualize_trajectories import plot_predictions, plot_predictions_background_img
from helper.calc_distance import calculate_distance
from helper.gradient_tracking import plot_grad_flow
from helper.histogram import create_histogram
from helper.results_to_xlxs import loss_on_traj_class

class Solver(object):
    """
    Class for training/evaluating the model
    """
    def __init__(self, optim,  loss_all, epochs, args, server):
        self.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.optim = optim
        self.loss_history_all = loss_all
        self.last_epoch = epochs
        self.dataset_name = args.dataset_name
        self.show_traj = args.show_traj
        self.args = args
        self.server = server
        self.best_val = 100
        self.best_val_FDE = 100
        self.best_val_epoch = 0

        self.distances_targets = np.array([])
        self.distances_outputs = np.array([])
        self.histo_distances_targets = np.array([])
        self.histo_distances_outputs = np.array([])
        self.histo_curvatues = np.array([])


    def reset_histories(self):
        """
        Reset loss history of model
        """
        self.loss_history = []


    def train(self, model, model_type, train_loader, val_loader, phase, num_epochs, log_nth=0, visualize=False, train_epoch = 0):
        """
        Train a given model with the provided data.
        :param model: Model object
        :param model_type: Type of model (for current implementation either linear, lstm or social-lstm)
        :param train_loader: Train data in torch.utils.data.DataLoader
        :param val_loader: Val data in torch.utils.data.DataLoader
        :param phase: Phase of model, either train, val or test
        :param num_epochs: Total number of training epochs
        :param log_nth: Log training accuracy and loss every nth iteration
        :param visualize: Specifies whether to visualize prediction or not (boolean)
        :param train_epoch: Current training epoch
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
                # Calculates the error with respect to the distances between all pedestrians in a scene
                distance_loss = Scene_distances()
            if "G_ADE_nl_regions" in model.losses:
                # Measures the average nonlinear displacement error
                ADE_nl = ADE_nl_regions(threshold=self.args.threshold_nl, approx_scheme=self.args.approx_scheme, p=2)
            if "G_NL" in model.losses:
                # Calculates the error in nonlinearity of each trajectory between prediction and ground truth
                non_linear_loss = nonlinear_loss()
            if "G_MSE" in model.losses:
                # Customized Mean S error
                customized_MSE = own_MSE(S=2)

            # Create new ADE and FDE for trajectories that are classified with respect to their degree of nonlinearity
            if self.args.nl_classified:
                if phase == "test":
                    print("Analyze trajectories in classify them in coarse groups...")
                    groups = ["strictly_linear", "linear", "gradually_nonlinear", "highly_nonlinear", "other"]
                    classified_trajectories_dict = {}
                    for group in groups:
                        classified_trajectories_dict[group] = nl_loss_classified_Trajectories(args=self.args, p=2, group=group)

            if phase == "train":
                model.train()
            elif phase == "test" or phase == "val":
                model.eval()
            else:
                raise ValueError("Specify phase either train or test!" )

            for i, batch in enumerate(train_loader, 1):
                # Get data from TrainLoader
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

                # Get prediction of model
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
                    loss_MSE = customized_MSE(outputs["out_xy"], targets["gt_xy"])

                # Additional losses
                if "G_distance_loss" in model.losses:
                    if "G_distance_loss" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_distance_loss"] = []
                    loss_dist = distance_loss.calc_distance_loss(outputs["out_xy_all"].to(device), batch["gt_xy"].to(device), batch["seq_start_end"], batch["pad_mask"], self.args)

                if "G_ADE_nl_regions" in model.losses:
                    if "G_ADE_nl_regions" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_ADE_nl_regions"] = []
                    loss_ADE_nl = ADE_nl(outputs["out_xy"], targets["gt_xy"])

                if "G_NL" in model.losses:
                    if "G_NL" not in self.loss_history_all[self.dataset_name][phase].keys():
                        self.loss_history_all[self.dataset_name][phase]["G_NL"] = []
                    nl_loss = non_linear_loss(outputs["out_xy"], targets["gt_xy"])

                # Loss for classified trajectories
                if self.args.nl_classified:
                    if phase == "test":
                        for group in groups:
                            loss_classified_trajectories = classified_trajectories_dict[group](outputs["out_xy"], targets["gt_xy"])


                # Backprop for training according to train loss
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
                        # Plot flow of gradients in order to detect vanishing gradients
                        if epoch%50 == 0 or epoch == num_epochs-1:
                            plot_grad_flow(model.named_parameters(), self.server, self.args, epoch)

                # Calculate distances to surrounding pedestrians for collision avoidance behavior
                if phase == "test":
                    if self.args.analyse_coll_avoidance:
                        if "seq_start_end" in batch.keys():
                            print("Calculating distances between pedestrians for each scene...")
                            self.distances_targets = np.concatenate((self.distances_targets, calculate_distance(batch["gt_xy"], batch["seq_start_end"], batch["pad_mask"], self.args)), axis=0)
                            self.distances_outputs = np.concatenate((self.distances_outputs, calculate_distance(outputs["out_xy_all"].detach().cpu().numpy(), batch["seq_start_end"], batch["pad_mask"], self.args)), axis=0)

                # Plot predictions
                if self.show_traj:
                    if self.args.socialforce:
                        if phase=="test":
                            plot_predictions(batch, outputs, phase, train_epoch, i, model_type, self.server, self.args)
                    else:
                        if phase == "test":
                            plot_predictions(batch, outputs, phase, train_epoch, i, model_type, self.server, self.args)


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


            # Calculate loss for trajectory-groups that are classified according to their degree of nonlinearity
            if self.args.nl_classified:
                if phase == "test":
                    loss_dict_classified = {}
                    for group in classified_trajectories_dict:
                        loss_dict_classified[group] = {}
                        loss_dict_classified[group]["ADE_nonlinear"] = classified_trajectories_dict[group].loss_epoch().detach().cpu().numpy()
                        loss_dict_classified[group]["FDE_nonlinear"] = classified_trajectories_dict[group].FDE_loss_epoch().detach().cpu().numpy()
                        loss_dict_classified[group]["nr_traj"] = classified_trajectories_dict[group].num_trajectories
                        loss_dict_classified[group]["total_nr_traj"] = classified_trajectories_dict[group].total_num_trajectories

                    # Write calculated losses for model and dataset in .xlsx-file for further analysis
                    path = os.path.join(self.root_path, "Analyse_Results", "ClassLoss", self.args.dataset_name)
                    loss_on_traj_class(path, self.args, loss_dict_classified)


            # Adjust threshold according to mean curvature for training on loss ADE_nl
            if phase == "train" and (self.args.train_loss == "mixed" or self.args.train_loss == "ADE_nl") and "G_ADE_nl_regions" in model.losses and epoch == 0:
                self.histo_curvatues = ADE_nl.get_curvatures()
                self.args.threshold = self.histo_curvatues.mean()

            # Create histogram for curvatures-values of trajectories
            if phase == "test" and "G_ADE_nl_regions" in model.losses and epoch == 0:
                print("Create histogram for curvature values in dataset.")
                self.histo_curvatues = ADE_nl.get_curvatures()
                bins = np.arange(self.histo_curvatues.min(), self.histo_curvatues.max(), 0.1)
                hist = create_histogram(args=self.args, server=self.server)
                hist.plot_histogram_curvatures(self.histo_curvatues, bins)

            # Note lowest validation error
            if phase == "val":
                L2_loss = L2_loss.loss_epoch().detach().cpu().numpy()
                if L2_loss < self.best_val:
                    self.best_val = L2_loss
                    self.best_val_FDE = FDE_loss.loss_epoch().detach().cpu().numpy()
                    self.best_val_epoch = train_epoch+1

            # Print Epoch loss
            if log_nth:
                print('[Epoch %d/%d] ' %(epoch + 1, num_epochs) + str(phase) + ' loss: %.3f' % (self.loss_history_all[self.dataset_name][str(phase)]["G_ADE"][-1]))

            # Validation
            if phase == "train":
                self.train(model, model_type, val_loader, val_loader, log_nth = log_nth, num_epochs=1, phase="val", visualize=visualize, train_epoch = epoch)

        if phase == "test":
            if self.args.analyse_coll_avoidance:
                self.histo_distances_targets = np.histogram(self.distances_targets, np.arange(0, 30.1, 0.1))
                self.histo_distances_outputs = np.histogram(self.distances_outputs, np.arange(0, 30.1, 0.1))

        if phase == "train":
            self.last_epoch += num_epochs
            print("finish")


