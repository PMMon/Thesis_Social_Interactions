import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from Implementations.initialization import random_initialization, meeting


class generate_scene:
    """
    Base class covering fundamental methods to generate datasets with the Social Force model
    """
    def __init__(self, scenario):
        self.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.scenario = scenario

    def initialize_agents(self, v_max, v_min, nr_agents, agent_state, a, b, setting="random"):
        """
        Initialize the states of the agents in the dataset that is being generated
        :param v_max: Maximal initial velocity of agents
        :param v_min: Minimal initial velocity of agents
        :param nr_agents: Number of agents in the scene
        :param agent_state: Numpy array of shape N x 6 for N already initialized agent-states in the dataset. Defines hand-crafted inital states of N agents.
        The row of the i-th agent needs to be of the form: [x, y, v_x, v_y, d_x, d_y].
        :param a: Specifies width of square/rectangle - only important if dataset == square or dataset == rectangle
        :param b: Specifies length of rectangle - only important if dataset == square or dataset == rectangle
        :param scenario: Either meeting or random
        :return: Initial states of agents
        """
        initial_state = agent_state

        if setting == "meeting":
            initialize = meeting(self.scenario, v_max, v_min, a, b, tau=0.5)
        else:
            initialize = random_initialization(self.scenario, v_max, v_min, a, b, tau=0.5)

        for i in range(nr_agents-len(initial_state)):
            if i == 0 and not initial_state:
                initial_state = initialize.initialize_state(i+1)
            else:
                initial_state = np.append(initial_state, initialize.initialize_state(i+1), axis=0)

        return initial_state


    def initialize_space(self, input_image, create_background, scaling):
        """
        Incorporates the obstacle-setting of the desired input-scene by reading in a segmented image of the scene.
        The obstacle-class that should be incorporated in the simulated data needs to be represented in black. 
        :param Input_image: Path to input image of scene
        :param Create_background: if True, create background image with approximated obstacles
        :return: Numpy array that determines boundaries of obstacles
        """
        space = []

        # Specify outputfile location
        if create_background:
            self.background_path =  os.path.abspath(os.path.join(self.root_path, "docs", "simulated_backgrounds"))
            self.background_image_simulated = os.path.join(self.background_path, self.scenario + "_simulated.jpg")

            if not os.path.exists(self.background_path):
                os.makedirs(self.background_path)

        # Read in image
        print("Read in segmented input-image to approximate boundaries of obstacles")
        img = cv2.imread(input_image, 0)
        print("dimensions of input-image: (%d, %d)" % (img.shape[0], img.shape[1]))
        img = img[::-1, :]
        img1 = imutils.resize(img)

        # Define threshold: (input-image, threshold, max_value, method) set all values <= threshold to 0 all other to max_value
        ret, thresh = cv2.threshold(img1, 0, 255, 1)
        # Find contours
        image_not_used, contours, hierarchy = cv2.findContours(thresh, 0, 2)

        if create_background:
            fig = plt.figure()
            ax = plt.axes()
            ax.grid(linestyle='dotted')
            ax.set_axisbelow(True)
            ax.set_title(self.scenario)
            ax.set_xlim(img.shape[1]*scaling)
            ax.set_ylim(img.shape[0]*scaling)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        for element in contours:
            epsilon = 0.02 * cv2.arcLength(element,True)  # approximated curve for epsilon = 2% of arc length empirically fitted best
            approx = cv2.approxPolyDP(element, epsilon, True)
            approx = np.append(approx, approx[0, :, :].reshape(1, 1, 2), axis=0)
            approx = (approx.squeeze()) * 0.05

            for i in range(len(approx) - 1):
                space.append(np.linspace(approx[i, :], approx[i + 1, :], 100))

            if create_background:
                for s in space:
                    ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

        if create_background:
            fig.savefig(self.background_path)
            plt.close()

        print("Obstacles initialized.")

        return space


    def resulting_potential(self, state_at_scene, X, Y, s):
        """
        Calculate repulsive potential between pedestrians for current scene
        :param state_at_scene: State of the pedestrians in the current scene
        :param X: X-Component of the grid
        :param Y: Y-Component of the grid
        :return: Dictionary with repulsive potential, it's x-components and y-components
        """
        v, v_x, v_y = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)
        for ped in range(len(state_at_scene)):
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    a = np.expand_dims(state_at_scene[ped,:].copy(), axis=0)
                    a = np.concatenate((a,np.array([[X[i,j], Y[i,j], 0, 0, 0, 0, 0.5]])))
                    pot = s.calc_potential(a)
                    v_x[i, j] = pot[0, 0]
                    v_y[i, j] = pot[0, 1]
                    v[i,j] = v[i,j] + v_y[i, j]

        return {"v": v, "v_x": v_x, "v_y": v_y}


    def contours(self, X, Y, state, s):
        """
        Calculates the forces between pedestrians in a scene
        :param X: X-Component of the grid
        :param Y: Y-Component of the grid
        :param state: State of the pedestrians in the current scene
        :return: Dictionary with social forces between pedestrians, it's x-components and y-components
        """
        F, F_x, F_y = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                a = state.copy()
                a[0, 0:2] = [X[i, j], Y[i, j]]
                Force = s.calc_force_between_pedestrians(a)
                F_x[i, j] = Force[0, 0]
                F_y[i, j] = Force[0, 1]
                F[i, j] = np.linalg.norm([F_x[i, j], F_y[i, j]], axis=-1)

        return {"F": F, "F_x": F_x, "F_y": F_y}