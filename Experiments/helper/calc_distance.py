import numpy as np

# ============================================== Description ==============================================
# Script designed to calculate distances between each pedestrian in a given scene.
# Trajectories are only predicted for pedestrians that stay in the scene the whole sequence length, therefore,
# only these pedestrians are taken into account.

# update: padding has been implemented. Additional arguments that might be passed to the function:
#   - pad_mask: an NxS matrix with N: Number of Pedestrians that have been in the sequence
#                                  S: Length of Sequence
#   -> indicating which scene_data for each pedestrian in the sequence is padded (0) or given by the scene (1)
# ==========================================================================================================


def calculate_distance(input, seq_start_end, pad_mask, args):
    """
    Calculate distances between each pedestrian in a given scene.
    :param input: Torch tensor with future trajectories of pedestrians. Has shape (pred_len, batch_size, 2)
    :param seq_start_end: A list of tuples which delimit sequences within batch
    :param pad_mask: A tuple of NxS matrices that indicate which positions are padded, with N: Number of Pedestrians that have been in the sequence & S: Length of Sequence
    :param args: Command line arguments
    :return: Tensor with distances between pedestrians
    """
    # Get max distance of interest according to value of sigma/range of social force
    sigma_r0_rel = {0.2171: 0.5, 0.4343: 1, 0.8686: 2, 1.303: 3, 1.7371: 4, 2.171: 5, 2.6058: 6}
    max_r0 = sigma_r0_rel[args.sigma]

    # iterate trough all sequences
    distance_vector = np.array([])
    for seq_nr, seq in enumerate(seq_start_end):
        # Get information about pedestrians in each sequence
        first_ped = seq[0].item()
        last_ped = seq[1].item()
        for scene_nr, scene in enumerate(input):
            if scene[first_ped:last_ped,:].shape[0] == 1:
                given_chords = scene[first_ped:last_ped, :][:,:]
            else:
                given_chords = scene[first_ped:last_ped,:][pad_mask[seq_nr][:, scene_nr + args.obs_len]>0,:]
            r_a = np.expand_dims(given_chords, 1)
            r_b = np.expand_dims(given_chords, 0)
            distance_matrix = np.linalg.norm(r_a-r_b, axis=2)
            distance_vector = np.concatenate((distance_vector, distance_matrix[np.triu_indices(distance_matrix.shape[0], k = 1)]), axis=0)
            distance_vector = distance_vector[distance_vector<=max_r0]


    return distance_vector


