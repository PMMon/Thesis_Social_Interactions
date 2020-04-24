import torch
import numpy as np
from solver.visualize_trajectories import visualize_specific_trajectories, visualize_grouped_trajectories

class L2:
    def __init__(self, norm=2):
        self.norm = norm
        self.losses = torch.Tensor()

    def loss_batch(self, input, target):
        """"
        Classical Average Displacement Error (ADE) between predicted and target trajectories
        :param input: pred_length x num_peds_in_batch x 2 -> input-batch with predicted coordinates for trajectories
        :param target: pred_length x num_peds_in_batch x 2 -> target-batch with ground truth data
        :return: ADE
        """
        loss = torch.norm(target-input, p=self.norm, dim=2).mean(0)
        self.losses = torch.cat((self.losses.to(loss), loss), 0)
        loss_batch = loss.mean(0)

        return loss_batch

    def loss_epoch(self):
        loss_epoch = self.losses.mean(0)
        return loss_epoch


class FDE(L2):
    def __init__(self, norm=2):
        super().__init__(norm)

    def loss_batch(self, input, target):
        """"
        Classical Final Displacement Error (FDE). Calculated as difference between last point of predicted and target trajectory
        :param input: num_peds_in_batch x 2 -> input-batch with last predicted coordinate of trajectories
        :param target: num_peds_in_batch x 2 -> target-batch with ground truth data
        :return: FDE
        """
        loss = torch.norm(target - input, p=self.norm, dim=1).detach().cpu()
        loss_batch = loss.mean(0)
        self.losses = torch.cat((self.losses, loss), 0)
        return loss_batch


class own_MSE(torch.nn.Module):
    def __init__(self, S=2, norm=2):
        self.S = S
        self.norm = norm
        self.losses = torch.Tensor()
        super(own_MSE, self).__init__()

    def forward(self, input, target):
        """
        Mean Squared Error. We can modify S such that the distance between input and target is taken to the power of S
        :param input: pred_length x num_peds_in_batch x 2 -> input-batch with predicted coordinates for trajectories
        :param target: pred_length x num_peds_in_batch x 2 -> target-batch with ground truth data
        :return: Error
        """
        loss = (torch.norm(target - input, p=self.norm, dim=2)**self.S).mean(0)
        self.losses = torch.cat((self.losses.to(loss), loss), 0)
        loss_batch = loss.mean(0)

        return loss_batch

    def loss_epoch(self):
        loss_epoch = self.losses.mean(0)
        return loss_epoch

class ADE_nl_loss(torch.nn.Module):
    def __init__(self, p, approx_scheme):
        self.p = p
        self.approx_scheme = approx_scheme
        self.losses = torch.Tensor()
        self.FDE_losses = torch.Tensor()
        self.curvatures = torch.Tensor()
        super(ADE_nl_loss, self).__init__()

    def forward(self, *args):
        pass

    def loss_epoch(self):
        if self.losses.shape[0] == 0:
            print("No loss measured in nonlinear ADE for Epoch!")

        loss_epoch = self.losses.mean(0)

        return loss_epoch

    def FDE_loss_epoch(self):
        """"
        Classical Final Displacement Error (FDE). Calculated as difference between last point of predicted and target trajectory
        :return: FDE loss over epoch
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

class ADE_nl_Trajectories_fine(ADE_nl_loss):
    def __init__(self, args, p=2):
        self.threshold = args.threshold_nl
        self.args = args
        self.num_trajectories = 0
        super(ADE_nl_Trajectories_fine, self).__init__(p, args.approx_scheme)

    def forward(self, input, target, N):
        """
        This function will:
        1) Classify GT trajectories according to their amount of nonlinearity
        => This will be done by a number N of points of the trajectories for which the trajectory has a curvature >= self.threshold
        2) Calculate the Average Displacement Error (ADE) for these respective trajectories
        :param N: number N of points of the trajectories for which the trajectory has a curvature >= self.threshold
        :param input: pred_length x num_peds_in_batch x 2 -> input-batch with predicted coordinates for trajectories
        :param target: pred_length x num_peds_in_batch x 2 -> target-batch with coordinate for GT trajectories
        :return: ADE for groups of nonlinear trajectories
        """
        if self.approx_scheme == "menger_curvature":
            for i in range(0,len(target)-2):
                if i == 0:
                    curv_target = self.circle_approx(target[i:(i+3)]).unsqueeze(1)
                else:
                    curv_target = torch.cat([curv_target, self.circle_approx(target[i:(i+3)]).unsqueeze(1)], dim=1)

        # ======== Create criteria for classification =========
        # 1) curvature values equal or bigger than threshold:   curv_target >= t
        # nonlin_mask = torch.zeros(curv_target.shape)
        # nonlin_mask[torch.abs(curv_target) >= self.threshold] = 1

        nonlin_mask = torch.zeros(curv_target.shape)
        nonlin_mask[torch.abs(curv_target) >= self.threshold] = 1

        print(nonlin_mask)

        sum_mask = nonlin_mask.sum(dim=1)

        # =====================================================

        if self.args.visualize_classified and N == 6:
            visualize_specific_trajectories(target[:, sum_mask == N, :], curv_target[sum_mask == N, :], self.args, N, max_nr_traj=20)

        loss = torch.norm(target - input, p=self.p, dim=2)
        loss = loss[: , sum_mask == N]

        self.num_trajectories += loss.shape[1]

        loss = loss.mean(0)

        self.losses = torch.cat((self.losses.to(loss), loss), 0)
        loss_batch = loss.mean(0)

        return loss_batch

class ADE_nl_Trajectories_coarse(ADE_nl_loss):
    def __init__(self, args, group, p=2):
        self.threshold = args.threshold_nl
        self.group = group
        self.args = args
        self.num_trajectories = 0
        self.total_num_trajectories = 0
        super(ADE_nl_Trajectories_coarse, self).__init__(p, args.approx_scheme)

    def forward(self, input, target):
        """
        This function will:
        1) Classify GT trajectories according to their amount of nonlinearity
        => This will be done by a number N of points of the trajectories for which the trajectory has a curvature >= self.threshold
        2) Calculate the Average Displacement Error (ADE) for these respective trajectories
        :param N: number N of points of the trajectories for which the trajectory has a curvature >= k
        :param input: pred_length x num_peds_in_batch x 2 -> input-batch with predicted coordinates for trajectories
        :param target: pred_length x num_peds_in_batch x 2 -> target-batch with coordinate for GT trajectories
        :return: ADE for groups of nonlinear trajectories
        """
        if self.approx_scheme == "menger_curvature":
            for i in range(0,len(target)-2):
                if i == 0:
                    curv_target = self.circle_approx(target[i:(i+3)]).unsqueeze(1)
                else:
                    curv_target = torch.cat([curv_target, self.circle_approx(target[i:(i+3)]).unsqueeze(1)], dim=1)

        # ======== Create criteria for coarse classification =========
        # Groups: 1) strictly linear - 0.0 <= k < 0.11; N = 10
        #         2) linear - k
        #         3) slightly nonlinear over complete trajectory N > 5; 0.2 < k <= 0.5
        #         4) highly nonlinear at one region - N <= 4; 0.5 < k
        #         5) highly nonlinear over complete trajectory - N > 4; 0.5 < k

        nonlin_mask = torch.zeros(curv_target.shape)
        threshold_baseline = 0.1

        if self.group == "strictly_linear":
            threshold_lower = 0.0
            threshold_upper = 0.11
            nonlin_mask[(threshold_lower <= torch.abs(curv_target)) & (torch.abs(curv_target) < threshold_upper)] = 1

            sum_mask = nonlin_mask.sum(dim=1)
            criteria = (sum_mask == 10)

            print("strictly linear: ")

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

            print("linear: ")



        elif self.group == "med_nonlinear":
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

            print("medium nonlinear: ")



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

            print("highly nonlinear:")

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

            print("other:")



        else:
            raise ValueError("Invalid group: " + str(self.group))

        print("number of trajectories: " + str(target[:, criteria, :].shape[1]))

        if self.args.visualize_classified:
            visualize_grouped_trajectories(target[:, criteria , :], curv_target[criteria, :], self.args, group=self.group, sum_mask=sum_mask[criteria], max_nr_traj=10)

        # =============================================================

        loss = torch.norm(target - input, p=self.p, dim=2)
        self.total_num_trajectories += loss.shape[1]

        if self.group == "other":
            print("Total nr. of trajectories: " + str(self.total_num_trajectories))


        loss = loss[: , criteria]
        self.FDE_losses = torch.cat((self.FDE_losses.to(loss), loss[-1,:]), 0)

        self.num_trajectories += loss.shape[1]

        loss = loss.mean(0)

        self.losses = torch.cat((self.losses.to(loss), loss), 0)
        loss_batch = loss.mean(0)

        return loss_batch


class ADE_nl_regions(ADE_nl_loss):
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
        if self.approx_scheme == "menger_curvature":
            for i in range(0,len(target)-2):
                if i == 0:
                    curv_target = self.circle_approx(target[i:(i+3)]).unsqueeze(1)
                else:
                    curv_target = torch.cat([curv_target, self.circle_approx(target[i:(i+3)]).unsqueeze(1)], dim=1)

        self.curvatures = torch.cat((self.curvatures.to(curv_target), curv_target.clone()), 0)

        nonlin_mask = torch.zeros(curv_target.shape)
        nonlin_mask[(torch.abs(curv_target) >= self.threshold)] = 1 # Note changed for thesis work from (torch.abs(curv_target) >= self.threshold) to current state
        append_zeros = torch.zeros(nonlin_mask.shape[0],1)
        nonlin_mask = (torch.cat([append_zeros, nonlin_mask, append_zeros], dim=1)).transpose_(0,1).to(input)

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


class Scene_distances:
    def __init__(self):
        self.losses = torch.Tensor()

    def calc_distances(self, tens):
        r_a = tens.unsqueeze(1)
        r_b = tens.unsqueeze(0)
        distance_matrix = torch.norm(r_a - r_b, dim=2)
        return distance_matrix

    def calc_distance_loss(self, output, target, seq_start_end, pred_check, pad_mask):
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
                    scene_output = scene[first_ped:last_ped, :][pad_mask[seq_nr][:, scene_nr] > 0, :]
                    scene_target = target[scene_nr, first_ped:last_ped, :][pad_mask[seq_nr][:, scene_nr] > 0, :]
                dist_matrix_output = self.calc_distances(scene_output)
                dist_matrix_target = self.calc_distances(scene_target)

                # average out
                scene_loss = ((torch.abs(dist_matrix_output-dist_matrix_target)).sum(dim=0)).mean()
                loss = torch.cat((loss,scene_loss.unsqueeze(0)),0)

        loss = loss.mean().unsqueeze(0)
        self.losses = torch.cat((self.losses.to(loss), loss), 0)

        return loss

    def loss_epoch(self):
        loss_epoch = self.losses.mean(0)
        return loss_epoch


class nonlinear_loss(torch.nn.Module):
    def __init__(self, p=2, approx_scheme="menger_curvature", threshold=0.5):
        self.p = p
        self.approx_scheme = approx_scheme
        self.threshold = threshold
        self.losses = torch.Tensor()
        super(nonlinear_loss, self).__init__()

    def forward(self,input, target):
        """
        In order to analyse behavior in nonlinear regions, this forward-function uses (choosable) approximation-techniques in order
        to calculate the curvature of the predicted trajectories at discrete points.
        :param input: pred_length x num_peds_in_batch x 2 -> input-batch with predicted coordinates for trajectories
        :param target: pred_length x num_peds_in_batch x 2 -> target-batch with ground truth data
        :return: averaged difference between the curvature of the predicted trajectories and the curv. of the ground truth data at discrete points
        """
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




if __name__ == "__main__":
    import socket
    from visualize_trajectories import plot_traj_for_social_exp
    # Connect to visdom server
    servername = socket.gethostname()
    if "node" in servername:
        server = "http://atcremers11"
    else:
        server = 'http://localhost'

    threshold = 0.1

    output = torch.tensor([[[1,0],[-3,9]],
                           [[1,1],[-2,4]],
                           [[1,2],[-1,1]],
                           [[1,3],[0,0]],
                           [[1,4],[1,1]],
                           [[1,5],[2,4]],
                           [[1,6],[3,9]],
                           [[1,7],[4,9]],
                           [[1,8],[5,9]],
                           [[1,9],[6,7]],
                           [[1,10],[7,5]],
                           [[1,11],[8,3]]
                           ]).float()
    in_xy = torch.tensor([[[1, -1],[-4,16]]]).float()
    target = output.clone()
    modifier = torch.ones(target.shape)
    modifier[:,:,0] = 0
    modifier[:,1,:] = 1/0.3
    modifier = modifier * 0.3

    target = target*modifier
    target[:,1,:] += 0.5
    print("output: " + str(output))
    print("out dim: " + str(output.shape))
    print("target: " + str(target))
    print("target dim: " + str(target.shape))


    print("visualizing trajectories...")
    class args:
        def __init__(self, viz_port, dataset_name):
            self.viz_port = viz_port
            self.dataset_name = dataset_name

    phase = "test"
    train_epoch = 0
    i = 0
    model_type = "customized"
    batch = {}
    outputs = {}
    seq_start_end = torch.tensor([[0,2]])
    batch["gt_xy"] = target
    batch["out_xy"] = output
    batch["seq_start_end"] = seq_start_end
    batch["pred_check"] = (torch.ones(output.shape[1],1)*20,)
    batch["pad_mask"] = (torch.ones(output.shape[0:2]),)
    batch["in_xy"] = in_xy
    outputs["out_xy_all"] = output
    args = args(viz_port=8097, dataset_name="squaresimulated_V06b2u6058")
    plot_traj_for_social_exp(batch, outputs, phase, train_epoch, i, model_type, server, args)

    print("calculating ADE loss")
    ADE_loss = L2()
    loss_batch_ADE = ADE_loss.loss_batch(output, target)
    print("loss_batch_ADE: " + str(loss_batch_ADE))

    print("calculating loss for ADE_nl regions")
    ADE_loss_nl = ADE_nl_regions(threshold=threshold)
    loss_batch_ADE_nl = ADE_loss_nl(output,target)
    print("loss_batch_ADE_nl: " + str(loss_batch_ADE_nl))