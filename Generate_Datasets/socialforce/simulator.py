# coding=utf-8

"""Synthetic pedestrian behavior according to the Social Force model.

See Helbing and Molnár 1998.
"""

import numpy as np
import math

from .potentials import PedPedPotential
from .fieldofview import FieldOfView
from . import stateutils

MAX_SPEED_MULTIPLIER = 1.3  # with respect to initial speed


class Simulator(object):
    """Simulate social force model.

    Main interface is the state. Every pedestrian is an entry in the state and
    represented by a vector (x, y, v_x, v_y, d_x, d_y, [tau]).
    tau is optional in this vector.

    ped_space is an instance of PedSpacePotential.

    delta_t in seconds.
    tau in seconds: either float or numpy array of shape[n_ped].
    """
    def __init__(self, initial_state, v0, sigma, delta_t=0.4, tau=0.5, ped_space=None, beta=1):
        self.state = initial_state
        self.initial_speeds = stateutils.speeds(initial_state)
        self.max_speeds = np.maximum(MAX_SPEED_MULTIPLIER * self.initial_speeds, 3*np.ones(self.initial_speeds.shape))      # added in case of vel = 0 -> needs validation

        self.delta_t = delta_t
        self.beta = beta

        if self.state.shape[1] < 7:
            if not hasattr(tau, 'shape'):
                tau = tau * np.ones(self.state.shape[0])
            self.state = np.concatenate((self.state, np.expand_dims(tau, -1)), axis=-1)

        # potentials
        self.V = PedPedPotential(self.delta_t, v0=v0, sigma=sigma)
        self.U = ped_space

        # field of view
        self.w = FieldOfView()

    def f_ab(self):
        """Compute f_ab."""
        return -1.0 * self.V.grad_r_ab(self.state)

    def f_aB(self):
        """Compute f_aB."""
        if self.U is None:
            return np.zeros((self.state.shape[0], 0, 2))
        return -1.0 * self.U.grad_r_aB(self.state)

    def capped_velocity(self, desired_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, self.max_speeds / desired_speeds)
        return desired_velocity * np.expand_dims(factor, -1)

    def step(self):
        """Do one step in the simulation and update the state in place."""
        # accelerate to desired velocity
        e = stateutils.desired_directions(self.state)
        vel = self.state[:, 2:4]
        tau = self.state[:, 6:7]
        F0 = 1.0 / tau * (np.expand_dims(self.initial_speeds, -1) * e - vel)            # Acceleration term towards final destination

        # repulsive terms between pedestrians
        f_ab = self.f_ab()
        w = np.expand_dims(self.w(e, -f_ab), -1)
        F_ab = w * f_ab

        rotation = np.array([[0, -1], [1, 0]])
        F_ab_total = np.sum(F_ab, axis=1)
        F_ab_orth = np.transpose(np.dot(rotation, np.transpose(F_ab_total)))

        for j in range(F_ab_orth.shape[0]):
            if math.acos((np.dot(F_ab_orth[j], F0[j]))/(np.linalg.norm(F_ab_orth[j])*np.linalg.norm(F0[j]))) <= 1/2*math.pi:
                F_ab_orth[j] = np.transpose(np.dot(-rotation, np.transpose(F_ab_total[j])))

        F_ab = self.beta*F_ab_total + (1-self.beta)*F_ab_orth

        # repulsive terms between pedestrians and boundaries
        F_aB = self.f_aB()

        # social force
        F = F0 + F_ab + np.sum(F_aB, axis=1)
        # desired velocity
        w = self.state[:, 2:4] + self.delta_t * F
        # velocity
        v = self.capped_velocity(w)

        # update state
        self.state[:, 0:2] += v * self.delta_t
        self.state[:, 2:4] = v

        return self

    def calc_force(self, new_state):
        """Do one step in the simulation and update the state in place."""
        # accelerate to desired velocity
        e = stateutils.desired_directions(new_state)

        # repulsive terms between pedestrians
        f_ab =  -1.0 * self.V.grad_r_ab(new_state)
        w = np.expand_dims(self.w(e, -f_ab), -1)
        F_ab = w * f_ab

        # social force
        F = np.sum(F_ab, axis=1)

        return F

    def calc_potential(self, new_state):
        r_ab = self.V.r_ab(new_state)
        speeds = stateutils.speeds(new_state)
        desired_directions = stateutils.desired_directions(new_state)

        v = self.V.value_r_ab(r_ab, speeds, desired_directions)

        return v