import torch

from .visualize_trajectories import visualize_classified_trajectories

# ======================================= Description ===========================================
# Below you will find classes for various customized losses:
# * L2: Classical Average Displacement Error (ADE) between predicted and target trajectories
# * FDE: Classical Final Displacement Error (FDE) between predicted and target trajectories
# * own_MSE: For S=2 classical MSE. The variable S defines the exponent.
# * nl_loss: Base class for all losses that analyze nonlinearities of the trajectories
# * ADE_nl_classified_Trajectories: ADE and FDE on trajectory-classes, where the trajectories are
# classified with respect to their degree of nonlinearity
# * ADE_nl_regions: Nonlinear Average Displacement Error, i.e. the ADE in specific nonlinear regions
# of the trajectory
# * nonlinear_loss: Error in nonlinearity between predictions and ground truth
# * Scene_distances: Error in distances between all pedestrians in a scene for predicted and ground
# truth positions
# ===============================================================================================


class L2:
    """
    Classical Average Displacement Error (ADE) between predicted and target trajectories
    """
    def __init__(self, norm=2):
        self.norm = norm
        self.losses = torch.Tensor()

    def loss_batch(self, input, target):
        """
        Calculate loss for batch
        :param input: pred_length x num_peds_in_batch x 2 -> input-batch with predicted coordinates for trajectories
        :param target: pred_length x num_peds_in_batch x 2 -> target-batch with ground truth data
        :return: ADE
        """
        loss = torch.norm(target-input, p=self.norm, dim=2).mean(0)
        self.losses = torch.cat((self.losses.to(loss), loss), 0)
        loss_batch = loss.mean(0)

        return loss_batch

    def loss_epoch(self):
        """
        Calculate loss for epoch
        """
        loss_epoch = self.losses.mean(0)
        return loss_epoch


class FDE(L2):
    """
    Classical Final Displacement Error (FDE). Calculated as difference between last point of predicted and target trajectory
    """
    def __init__(self, norm=2):
        super().__init__(norm)

    def loss_batch(self, input, target):
        """
        Calculate loss for batch
        :param input: num_peds_in_batch x 2 -> input-batch with last predicted coordinate of trajectories
        :param target: num_peds_in_batch x 2 -> target-batch with ground truth data
        :return: FDE
        """
        loss = torch.norm(target - input, p=self.norm, dim=1).detach().cpu()
        loss_batch = loss.mean(0)
        self.losses = torch.cat((self.losses, loss), 0)
        return loss_batch


class own_MSE(torch.nn.Module):
    """
    Mean S Error. For S=2 this becomes the classical MSE. We can modify S such that the distance between input and target is taken to the power of S
    """
    def __init__(self, S=2, norm=2):
        self.S = S
        self.norm = norm
        self.losses = torch.Tensor()
        super(own_MSE, self).__init__()

    def forward(self, input, target):
        """
        Calculate error for batch
        :param input: pred_length x num_peds_in_batch x 2 -> input-batch with predicted coordinates for trajectories
        :param target: pred_length x num_peds_in_batch x 2 -> target-batch with ground truth data
        :return: Error
        """
        loss = (torch.norm(target - input, p=self.norm, dim=2)**self.S).mean(0)
        self.losses = torch.cat((self.losses.to(loss), loss), 0)
        loss_batch = loss.mean(0)

        return loss_batch

    def loss_epoch(self):
        """
        Calculate error for epoch
        """
        loss_epoch = self.losses.mean(0)
        return loss_epoch


class nl_loss(torch.nn.Module):
    """
    Base class for all losses that analyze nonlinearities of the trajectories
    """
    def __init__(self, p, approx_scheme):
        self.p = p
        self.approx_scheme = approx_scheme
        self.losses = torch.Tensor()
        self.FDE_losses = torch.Tensor()
        self.curvatures = torch.Tensor()
        super(nl_loss, self).__init__()

    def forward(self, *args):
        pass

    def loss_epoch(self):
        """
        Classical Average Displacement Error (ADE). Calculated as difference between predicted and target trajectory
        :return: ADE over epoch
        """
        if self.losses.shape[0] == 0:
            print("No loss measured in nonlinear ADE for Epoch!")

        loss_epoch = self.losses.mean(0)

        return loss_epoch

    def FDE_loss_epoch(self):
        """"
        Classical Final Displacement Error (FDE). Calculated as difference between last point of predicted and target trajectory
        :return: FDE over epoch
        """
        FDE_epoch = self.FDE_losses.mean(0)
        return FDE_epoch

    def get_curvatures(self):
        return self.curvatures.detach().cpu().numpy()

    def circle_approx(self, points):
        """
        This function will approximate the curvature at the middle of three following points, using the Menger curvature. The formula is as follows:
        k = 1/R = 4*area/(|P2-P1|*|P3-P2|*|P3-P1|)

        Use for area calculation:
        2*area = x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)
        :param points: tensor containing points
        :return: tensor containing values for curvature
        """
        area_twice = torch.abs(points[0, :, 0] * (points[1, :, 1] - points[2, :, 1]) + points[1, :, 0] * (points[2, :, 1] - points[0, :, 1]) + points[2, :, 0] * (points[0, :, 1] - points[1, :, 1]))
        distance = (torch.norm(points[1, :, :] - points[0, :, :], p=2, dim=1) * torch.norm(points[2, :, :] - points[1, :, :], p=2, dim=1) * torch.norm(points[2, :, :] - points[0, :, :], p=2, dim=1))

        k = 2 * area_twice / distance
        k[k != k] = 0

        return k


class nl_loss_classified_Trajectories(nl_loss):
    """
    Class that calculate the ADE and FDE of the models on trajectory classes that are classified with respect to their
    degree of nonlinearity/their influence by social interactions between pedestrians
    """
    def __init__(self, args, group, p=2):
        self.threshold = args.threshold_nl
        self.group = group
        self.args = args
        self.num_trajectories = 0
        self.total_num_trajectories = 0
        super(nl_loss_classified_Trajectories, self).__init__(p, args.approx_scheme)

    def forward(self, input, target):
        """
        Calculate ADE and FDE on trajectory-class
        1) Classify GT trajectories according to their degree of nonlinearity
        => This will be done by calculating the curvature of the trajectory at pred_len - 2 positions and classifying
        the trajectory using a heuristically defined set of rules
        2) Calculate the Average Displacement Error (ADE) and Final Displacement Error (FDE) for the respective classes
        :param input: pred_length x num_peds_in_batch x 2 -> input-batch with predicted coordinates for trajectories
        :param target: pred_length x num_peds_in_batch x 2 -> target-batch with coordinate for GT trajectories
        :return: ADE for groups of nonlinear trajectories
        """
        # 1) Classify GT trajectories
        if self.approx_scheme == "menger_curvature":
            for i in range(0,len(target)-2):
                if i == 0:
                    curv_target = self.circle_approx(target[i:(i+3)]).unsqueeze(1)
                else:
                    curv_target = torch.cat([curv_target, self.circle_approx(target[i:(i+3)]).unsqueeze(1)], dim=1)

        # ======== Criterias for classification =========
        # Defining k as curvature value at a position and N as number of positions
        # Groups: 1) strictly linear - 0.0 <= k < 0.11 for all N
        #         2) linear - k <= 0.4 for all N. If at one position the curvature is above 0.11: k > 0.11, the
        #            curvature value at the next position has to be k <= 0.11
        #         3) gradually nonlinear - k  < 0.7 for all N. No more than three consecutive points can have
        #            curvature values above k >= 0.2.
        #         4) highly nonlinear - at least three consecutive points have curvature values k >= 1.0
        #         5) other - remaining trajectories


        nonlin_mask = torch.zeros(curv_target.shape)

        if self.group == "strictly_linear":
            threshold_lower = 0.0
            threshold_upper = 0.11
            nonlin_mask[(threshold_lower <= torch.abs(curv_target)) & (torch.abs(curv_target) < threshold_upper)] = 1

            sum_mask = nonlin_mask.sum(dim=1)
            criteria = (sum_mask == 10)

        elif self.group == "linear":
            nonlin_mask_additional_cases = torch.zeros(curv_target.shape)

            threshold_lower = 0.0
            threshold_upper = 0.11
            threshold_max = 0.4
            nonlin_mask[(threshold_lower <= torch.abs(curv_target)) & (torch.abs(curv_target) < threshold_upper)] = 1
            nonlin_mask_additional_cases[(threshold_upper <= torch.abs(curv_target)) & (torch.abs(curv_target) <= threshold_max)] = 1
            nonlin_mask[(threshold_max < torch.abs(curv_target))] = -10

            num_following_curv = torch.zeros(nonlin_mask.shape[0])
            max_entries = torch.zeros(nonlin_mask.shape[0])

            for index in range(nonlin_mask_additional_cases.shape[1]):
                num_following_curv[nonlin_mask_additional_cases[:, index] == 1] += 1
                max_entries[nonlin_mask_additional_cases[:, index] != 1] = torch.max(max_entries[nonlin_mask_additional_cases[:, index] != 1],num_following_curv[nonlin_mask_additional_cases[:, index] != 1])
                max_entries[nonlin_mask_additional_cases[:, index] == 1] = torch.max(max_entries[nonlin_mask_additional_cases[:, index] == 1],num_following_curv[nonlin_mask_additional_cases[:, index] == 1])
                num_following_curv[nonlin_mask_additional_cases[:, index] != 1] = 0

            sum_mask = nonlin_mask.sum(dim=1)

            criteria = ((sum_mask == 10) | (max_entries == 1)) & (sum_mask > 0)

        elif self.group == "gradually_nonlinear":
            nonlin_mask_highly_nl = torch.zeros(curv_target.shape)

            threshold_lower = 0.2
            threshold_upper = 0.7
            nonlin_mask[(threshold_lower <= torch.abs(curv_target)) & (torch.abs(curv_target) < threshold_upper)] = 1
            nonlin_mask_highly_nl[(threshold_upper <= torch.abs(curv_target))] = 1

            num_following_curv = torch.zeros(nonlin_mask.shape[0])
            max_entries = torch.zeros(nonlin_mask.shape[0])

            num_following_curv_highly_nl = torch.zeros(nonlin_mask_highly_nl.shape[0])
            max_entries_highly_nl = torch.zeros(nonlin_mask_highly_nl.shape[0])

            for index in range(nonlin_mask.shape[1]):
                num_following_curv[nonlin_mask[:,index] == 1] += 1
                max_entries[nonlin_mask[:, index] != 1] = torch.max(max_entries[nonlin_mask[:, index] != 1], num_following_curv[nonlin_mask[:, index] != 1])
                max_entries[nonlin_mask[:, index] == 1] = torch.max(max_entries[nonlin_mask[:, index] == 1], num_following_curv[nonlin_mask[:, index] == 1])
                num_following_curv[nonlin_mask[:, index] != 1] = 0

                num_following_curv_highly_nl[nonlin_mask_highly_nl[:, index] == 1] += 1
                max_entries_highly_nl[nonlin_mask_highly_nl[:, index] != 1] = torch.max(max_entries_highly_nl[nonlin_mask_highly_nl[:, index] != 1],num_following_curv_highly_nl[nonlin_mask_highly_nl[:, index] != 1])
                max_entries_highly_nl[nonlin_mask_highly_nl[:, index] == 1] = torch.max(max_entries_highly_nl[nonlin_mask_highly_nl[:, index] == 1],num_following_curv_highly_nl[nonlin_mask_highly_nl[:, index] == 1])
                num_following_curv_highly_nl[nonlin_mask_highly_nl[:, index] != 1] = 0

            sum_mask = nonlin_mask.sum(dim=1)
            sum_mask_highly_nl = nonlin_mask_highly_nl.sum(dim=1)
            criteria = (3 <= max_entries) & (max_entries < 11) & (sum_mask_highly_nl <= 2)

        elif self.group == "highly_nonlinear":
            threshold_lower = 1.0
            nonlin_mask[(threshold_lower <= torch.abs(curv_target))] = 1

            num_following_curv = torch.zeros(nonlin_mask.shape[0])
            max_entries = torch.zeros(nonlin_mask.shape[0])

            for index in range(nonlin_mask.shape[1]):
                num_following_curv[nonlin_mask[:, index] == 1] += 1
                max_entries[nonlin_mask[:, index] != 1] = torch.max(max_entries[nonlin_mask[:, index] != 1],
                                                                    num_following_curv[nonlin_mask[:, index] != 1])
                max_entries[nonlin_mask[:, index] == 1] = torch.max(max_entries[nonlin_mask[:, index] == 1],
                                                                    num_following_curv[nonlin_mask[:, index] == 1])
                num_following_curv[nonlin_mask[:, index] != 1] = 0

            sum_mask = nonlin_mask.sum(dim=1)
            criteria = (3 <= max_entries) & (max_entries < 11) & (sum_mask > 0)

        elif self.group == "other":
            threshold_lower = 0.0
            threshold_upper = 0.11
            nonlin_mask[(threshold_lower <= torch.abs(curv_target)) & (torch.abs(curv_target) < threshold_upper)] = 1

            sum_mask = nonlin_mask.sum(dim=1)
            criteria = (sum_mask == 10)

            nonlin_mask = torch.zeros(curv_target.shape)
            nonlin_mask_additional_cases = torch.zeros(curv_target.shape)

            threshold_lower = 0.0
            threshold_upper = 0.11
            threshold_max = 0.4
            nonlin_mask[(threshold_lower <= torch.abs(curv_target)) & (torch.abs(curv_target) < threshold_upper)] = 1
            nonlin_mask_additional_cases[(threshold_upper <= torch.abs(curv_target)) & (torch.abs(curv_target) <= threshold_max)] = 1
            nonlin_mask[(threshold_max < torch.abs(curv_target))] = -10

            num_following_curv = torch.zeros(nonlin_mask.shape[0])
            max_entries = torch.zeros(nonlin_mask.shape[0])

            for index in range(nonlin_mask_additional_cases.shape[1]):
                num_following_curv[nonlin_mask_additional_cases[:, index] == 1] += 1
                max_entries[nonlin_mask_additional_cases[:, index] != 1] = torch.max(
                    max_entries[nonlin_mask_additional_cases[:, index] != 1],
                    num_following_curv[nonlin_mask_additional_cases[:, index] != 1])
                max_entries[nonlin_mask_additional_cases[:, index] == 1] = torch.max(
                    max_entries[nonlin_mask_additional_cases[:, index] == 1],
                    num_following_curv[nonlin_mask_additional_cases[:, index] == 1])
                num_following_curv[nonlin_mask_additional_cases[:, index] != 1] = 0

            sum_mask = nonlin_mask.sum(dim=1)

            criteria = criteria | (((sum_mask == 10) | (max_entries == 1)) & (sum_mask > 0))

            nonlin_mask = torch.zeros(curv_target.shape)
            nonlin_mask_highly_nl = torch.zeros(curv_target.shape)

            threshold_lower = 0.2
            threshold_upper = 0.7
            nonlin_mask[(threshold_lower <= torch.abs(curv_target)) & (torch.abs(curv_target) < threshold_upper)] = 1
            nonlin_mask_highly_nl[(threshold_upper <= torch.abs(curv_target))] = 1

            num_following_curv = torch.zeros(nonlin_mask.shape[0])
            max_entries = torch.zeros(nonlin_mask.shape[0])

            num_following_curv_highly_nl = torch.zeros(nonlin_mask_highly_nl.shape[0])
            max_entries_highly_nl = torch.zeros(nonlin_mask_highly_nl.shape[0])

            for index in range(nonlin_mask.shape[1]):
                num_following_curv[nonlin_mask[:, index] == 1] += 1
                max_entries[nonlin_mask[:, index] != 1] = torch.max(max_entries[nonlin_mask[:, index] != 1],
                                                                    num_following_curv[nonlin_mask[:, index] != 1])
                max_entries[nonlin_mask[:, index] == 1] = torch.max(max_entries[nonlin_mask[:, index] == 1],
                                                                    num_following_curv[nonlin_mask[:, index] == 1])
                num_following_curv[nonlin_mask[:, index] != 1] = 0

                num_following_curv_highly_nl[nonlin_mask_highly_nl[:, index] == 1] += 1
                max_entries_highly_nl[nonlin_mask_highly_nl[:, index] != 1] = torch.max(
                    max_entries_highly_nl[nonlin_mask_highly_nl[:, index] != 1],
                    num_following_curv_highly_nl[nonlin_mask_highly_nl[:, index] != 1])
                max_entries_highly_nl[nonlin_mask_highly_nl[:, index] == 1] = torch.max(
                    max_entries_highly_nl[nonlin_mask_highly_nl[:, index] == 1],
                    num_following_curv_highly_nl[nonlin_mask_highly_nl[:, index] == 1])
                num_following_curv_highly_nl[nonlin_mask_highly_nl[:, index] != 1] = 0

            sum_mask = nonlin_mask.sum(dim=1)
            sum_mask_highly_nl = nonlin_mask_highly_nl.sum(dim=1)
            criteria = criteria | ((3 <= max_entries) & (max_entries < 11) & (sum_mask_highly_nl <= 2))

            threshold_lower = 1.0
            nonlin_mask = torch.zeros(curv_target.shape)
            nonlin_mask[(threshold_lower <= torch.abs(curv_target))] = 1

            num_following_curv = torch.zeros(nonlin_mask.shape[0])
            max_entries = torch.zeros(nonlin_mask.shape[0])

            for index in range(nonlin_mask.shape[1]):
                num_following_curv[nonlin_mask[:, index] == 1] += 1
                max_entries[nonlin_mask[:, index] != 1] = torch.max(max_entries[nonlin_mask[:, index] != 1],
                                                                    num_following_curv[nonlin_mask[:, index] != 1])
                max_entries[nonlin_mask[:, index] == 1] = torch.max(max_entries[nonlin_mask[:, index] == 1],
                                                                    num_following_curv[nonlin_mask[:, index] == 1])
                num_following_curv[nonlin_mask[:, index] != 1] = 0

            sum_mask = nonlin_mask.sum(dim=1)
            criteria = criteria | ((3 <= max_entries) & (max_entries < 11) & (sum_mask > 0))

            criteria = ~criteria

        else:
            raise ValueError("Invalid group: " + str(self.group))

        # =============================================================
        # 2) Calculate ADE and FDE of class

        if self.args.visualize_classified and self.group != "other":
            # visualized max_nr_traj trajectories of the respective class
            visualize_classified_trajectories(target[:, criteria , :], curv_target[criteria, :], self.args, group=self.group, sum_mask=sum_mask[criteria], max_nr_traj=10)

        # Calculate ADE of class
        loss = torch.norm(target - input, p=self.p, dim=2)
        self.total_num_trajectories += loss.shape[1]

        if self.group == "other":
            print("ADE and FDE for trajectory-classes calculated.")

        loss = loss[: , criteria]
        # Calculate FDE of class
        self.FDE_losses = torch.cat((self.FDE_losses.to(loss), loss[-1,:]), 0)

        self.num_trajectories += loss.shape[1]

        loss = loss.mean(0)

        self.losses = torch.cat((self.losses.to(loss), loss), 0)
        loss_batch = loss.mean(0)

        return loss_batch


class ADE_nl_regions(nl_loss):
    """
    Nonlinear average displacement error
    """
    def __init__(self, p=2, approx_scheme="menger_curvature", threshold=0.5):
        self.threshold = threshold
        super(ADE_nl_regions, self).__init__(p, approx_scheme)

    def forward(self, input, target):
        """
        In order to analyse the behavior in nonlinear regions, this forward-function uses (choosable) approximation-techniques in order
        to calculate the curvature of the predicted trajectories at discrete points.
        It then defines nonlinear regions by a threshold and calculates the ADE loss exclusively at these specific regions.
        :param input: pred_length x num_peds_in_batch x 2 -> input-batch with predicted coordinates for trajectories
        :param target: pred_length x num_peds_in_batch x 2 -> target-batch with ground truth data
        :return: ADE at specific nonlinear regions
        """
        # Calculate curvature values of trajectories
        if self.approx_scheme == "menger_curvature":
            for i in range(0,len(target)-2):
                if i == 0:
                    curv_target = self.circle_approx(target[i:(i+3)]).unsqueeze(1)
                else:
                    curv_target = torch.cat([curv_target, self.circle_approx(target[i:(i+3)]).unsqueeze(1)], dim=1)

        self.curvatures = torch.cat((self.curvatures.to(curv_target), curv_target.clone()), 0)

        # Exclude positions with curvature values below threshold-value
        nonlin_mask = torch.zeros(curv_target.shape)
        nonlin_mask[(torch.abs(curv_target) >= self.threshold)] = 1
        append_zeros = torch.zeros(nonlin_mask.shape[0],1)
        nonlin_mask = (torch.cat([append_zeros, nonlin_mask, append_zeros], dim=1)).transpose_(0,1).to(input)

        # Calculate loss
        loss = torch.norm(target-input, p=self.p, dim=2)
        negative_mask = torch.zeros(loss.shape).to(loss)
        negative_mask[nonlin_mask == 0] = loss[nonlin_mask == 0].clone()
        loss = loss - negative_mask
        loss = loss.sum(dim=0)

        nonlin_mask = nonlin_mask.sum(dim=0)
        nonlin_mask[nonlin_mask == 0] = 1

        loss = loss/nonlin_mask
        loss = loss[loss != 0]

        self.losses = torch.cat((self.losses.to(loss), loss), 0)
        if len(loss) == 0:
            print("No loss measured in nonlinear ADE for batch!")

        loss_batch = loss.mean(0)

        return loss_batch


class nonlinear_loss(torch.nn.Module):
    """
    Error in nonlinearity between predicted and target trajectories
    """
    def __init__(self, p=2, approx_scheme="menger_curvature", threshold=0.5):
        self.p = p
        self.approx_scheme = approx_scheme
        self.threshold = threshold
        self.losses = torch.Tensor()
        super(nonlinear_loss, self).__init__()

    def forward(self,input, target):
        """
        Calculates the error in nonlinearity between predicted and target trajectories.
        :param input: pred_length x num_peds_in_batch x 2 -> input-batch with predicted coordinates for trajectories
        :param target: pred_length x num_peds_in_batch x 2 -> target-batch with ground truth data
        :return: averaged difference between the curvature of the predicted trajectories and the curv. of the ground truth data at discrete points
        """
        # Calculate curvature values of trajectories
        if self.approx_scheme == "menger_curvature":
            for i in range(0,len(input)-2):
                if i == 0:
                    curv_input = self.circle_approx(input[i:(i+3)]).unsqueeze(1)
                    curv_target = self.circle_approx(target[i:(i+3)]).unsqueeze(1)
                else:
                    curv_input = torch.cat([curv_input, self.circle_approx(input[i:(i+3)]).unsqueeze(1)], dim=1)
                    curv_target = torch.cat([curv_target, self.circle_approx(target[i:(i+3)]).unsqueeze(1)], dim=1)

        loss = (torch.abs(curv_input-curv_target)).mean(1)
        self.losses = torch.cat((self.losses.to(loss), loss), 0)
        loss_batch = loss.mean(0)

        return loss_batch

    def loss_epoch(self):
        loss_epoch = self.losses.mean(0)
        return loss_epoch

    def circle_approx(self, points):
        """
        This function will approximate the curvature at the middle of three following points, using the Menger curvature. The formula is as follows:
        k = 1/R = 4*area/(|P2-P1|*|P3-P2|*|P3-P1|)

        Use for area calulation:
        2*area = x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)
        :param points: tensor containing points
        :return: tensor containing values for curvature
        """
        area_twice = torch.abs(points[0,:,0]*(points[1,:,1] - points[2,:,1]) + points[1,:,0]*(points[2,:,1] - points[0,:,1]) + points[2,:,0]*(points[0,:,1] - points[1,:,1]))
        distance = (torch.norm(points[1,:,:]-points[0,:,:], p=2, dim=1)*torch.norm(points[2,:,:]-points[1,:,:], p=2, dim=1)*torch.norm(points[2,:,:]-points[0,:,:], p=2, dim=1))

        k = 2*area_twice/distance
        k[k != k] = 0

        return k


class Scene_distances:
    """
    Class that calculates for each scene the euclidean distance between all pedestrian in the scene. This information is used as a
    loss function between real distance-values and predicted distance-values
    """
    def __init__(self):
        self.losses = torch.Tensor()

    def calc_distances(self, tens):
        """
        calculate distance between pedestrians in a scene
        :param tens: Tensor with positions of pedestrians in a scene
        :return: Tensor with distance between pedestrians in a scene
        """
        r_a = tens.unsqueeze(1)
        r_b = tens.unsqueeze(0)
        distance_matrix = torch.norm(r_a - r_b, dim=2)
        return distance_matrix

    def calc_distance_loss(self, output, target, seq_start_end, pad_mask, args):
        """
        Calculates the error between the distances between pedestrians in a scene for the prediction and ground truth data
        :param output: Predictions of model
        :param target: Ground truth data
        :param seq_start_end: A list of tuples which delimit sequences within batch
        :param pad_mask: A tuple of NxS matrices that indicate which positions are padded, with N: Number of Pedestrians that have been in the sequence & S: Length of Sequence
        :param args: Command line arguments
        :return: Loss
        """
        loss = torch.tensor([]).to(output)

        for seq_nr, seq in enumerate(seq_start_end):
            # Get information about pedestrians in scene
            first_ped = seq[0].item()
            last_ped = seq[1].item()
            for scene_nr, scene in enumerate(output):
                if scene[first_ped:last_ped, :].shape[0] == 1:
                    scene_output = scene[first_ped:last_ped, :][:, :]
                    scene_target = target[scene_nr, first_ped:last_ped, :][:, :]
                else:
                    scene_output = scene[first_ped:last_ped, :][pad_mask[seq_nr][:, scene_nr + args.obs_len] > 0, :]
                    scene_target = target[scene_nr, first_ped:last_ped, :][pad_mask[seq_nr][:, scene_nr + args.obs_len] > 0, :]
                dist_matrix_output = self.calc_distances(scene_output)
                dist_matrix_target = self.calc_distances(scene_target)

                # average out
                scene_loss = ((torch.abs(dist_matrix_output-dist_matrix_target)).sum(dim=0)).mean()
                loss = torch.cat((loss,scene_loss.unsqueeze(0)),0)

        loss = loss.mean().unsqueeze(0)
        self.losses = torch.cat((self.losses.to(loss), loss), 0)

        return loss

    def loss_epoch(self):
        """
        Loss of epoch
        """
        loss_epoch = self.losses.mean(0)
        return loss_epoch