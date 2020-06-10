import numpy as np

# ============================================= Description =============================================
# Computation of the desired direction of each pedestrian and the speed of a pedestrian,
# according to the Social Force model.
#
# For a detailed description of the Social Force Model see:
# D. Helbing and P. Monlar. “Social Force Model for Pedestrian Dynamics”. In: Physical Review 51.5 (1995).
#
# Link to respective work: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.51.4282
# =======================================================================================================

def desired_directions(state):
    """
    Given the current state and destination, compute desired direction
    """
    destination_vectors = state[:, 4:6] - state[:, 0:2]
    norm_factors = np.linalg.norm(destination_vectors, axis=-1)
    return destination_vectors / np.expand_dims(norm_factors, -1)


def speeds(state):
    """
    Return the speeds corresponding to a given state
    """
    return np.linalg.norm(state[:, 2:4], axis=-1)
