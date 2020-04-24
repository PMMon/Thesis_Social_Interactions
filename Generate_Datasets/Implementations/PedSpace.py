# ==Imports==
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import contextmanager
import numpy as np
import socialforce
from initialization import random_initialization
from bool_flag import bool_flag
import argparse
import os
import cv2
import socket
import visdom
import imutils

# Realistic Configs for various datasets:
# ETH: V0: 4; sigma: 1.303; U0: 6; r: 2

class PedSpace_Interactions:
    """
    Class that creates Sequences of scenes with Agent-Agent interactions and Agent-Obstacle interactions
    """
    def __init__(self, args):
        self.nr_agents = args.nr_agents
        self.nr_scenes = args.nr_scenes

        self.scene = args.scene
        self.scaling = args.scaling
        self.v_max = args.v_max
        self.v_min = args.v_min

        self.threshold = args.threshold

        if args.V0 == -1 or args.sigma == -1.0:
            raise ValueError("Please specify V0 and sigma!")
        else:
            self.V0 = args.V0
            self.sigma = args.sigma

        self.U0 = args.U0
        self.r = args.r
        self.a = args.a
        self.b = args.b

        self.create_background = args.create_background
        self.create_video = args.create_video
        self.plot_potentials = args.plot_potentials

        self.image_root = os.path.join("docs", "real_scene_images")
        self.image_path = self.image_root + "/" + args.scene + "-op.jpg"

        self.dataset_root = os.path.join("docs", "simulated_data")
        self.dataset_path = self.dataset_root + "/train_" + args.scene + "simulated.txt"

        self.background_root = os.path.join("docs", "simulated_background")
        self.background_path = self.background_root + "/" + args.scene + "simulated.jpg"

        self.video_root = os.path.join("docs", "simulated_videos")
        self.video_path = self.video_root + "/" + args.scene + "-simulated.mp4"

        if args.visdom:
            self.viz = self.connect2visdom(args.viz_server, args.viz_port)
        else:
            self.viz = None


    def random_crossings(self):
        """
        Generate Sequence of Scene with obstacle-setting of the desired Input-Scene. For this, the input-scene-op.jpg image is read in an the obstacles are approximated accordingly.
        :param scene: Name of input-scene
        :param nr_agents: Number of agents that will be constantly in scene. Whenever an agent is leaving the scene, a new agent will spawn
        :param nr_scenes: Number of Frames/Scenes in Sequence
        :param image_path: Path to input-image for obstacle approximation
        :param dataset_path: Path to output-txt-file (generated dataset)
        :param create_background: Boolean - if True, create background image for sequence
        :param create_video: Boolean - if True, create .mp4-file which visualizes the sequence
        :return: Dataset (txt-file) and if chosen video and background image
        """

        # agents have state vectors of the form (x, y, v_x, v_y, d_x, d_y, [tau])

        print("start generating scene for %s..." % (self.scene))

        initialize = random_initialization(self.scene, self.v_max, self.v_min, a=self.a, b=self.b, tau=0.5)
        initial_state = initialize.initialize_state(0)

        self.total_y_min = initialize.total_y_min
        self.total_y_max = initialize.total_y_max
        self.total_x_min = initialize.total_x_min
        self.total_x_max = initialize.total_x_max

        # Initialize state for each agent
        for i in range(1, self.nr_agents):
            initial_state = np.append(initial_state, initialize.initialize_state(0), axis=0)

        # Initialize space according to input image
        space = self.conv_image_to_space(self.image_path, self.create_background, self.scaling)

        # Get Socialforce Simulator
        s = socialforce.Simulator(initial_state, ped_space=socialforce.PedSpacePotential(space, u0=self.U0, r=self.r), v0=self.V0, sigma=self.sigma)

        # Get Array for scene numbers and agents
        scene = np.zeros(self.nr_agents)
        agents = np.arange(self.nr_agents) + 1

        # Concatenate arrays with respective x-y-coordinates
        dataset = np.concatenate((scene.reshape(self.nr_agents, 1), agents.reshape(self.nr_agents, 1), s.state[:, 0:2]), axis=1)

        # Write Result to txt-file
        print("create dataset as txt-file for scene %s" % (self.scene))
        file = open(self.dataset_path, "w")
        np.savetxt(file, dataset, fmt="%.2f", delimiter="\t")
        file.close()

        file = open(self.dataset_path, "a")

        # Repeat for each scene and update according steps of agents
        states = []
        agent_list = []
        states.append(s.state.copy())

        for _ in range(self.nr_scenes):
            state = s.step().state
            agent_list.append(agents.copy())
            scene += 1

            # avoid oscillations of pedestrians
            if _ >= 10:
                dist = np.linalg.norm(states[_][:, 0:2] - states[_ - 10][:, 0:2], axis=1)
                agent_cond = agent_list[_] == agent_list[_ - 10]
                add_cond = (dist[:] < 0.8) & agent_cond
            else:
                add_cond = np.ones((state.shape[0])) > 1

            # periodic boundary conditions
            condition = (state[:, 0] > self.total_x_max + self.threshold) | (state[:, 0] < self.total_x_min - self.threshold) | (state[:, 1] > self.total_y_max + self.threshold) | (state[:, 1] < self.total_y_min - self.threshold) | add_cond

            agents[np.argwhere(condition)] = (max(agents) + np.arange(len(np.argwhere(condition))) + 1).reshape(-1, 1)

            # Generate new Agents when old Agents left the scene
            if len(state[condition]) != 0:
                state[condition] = (np.stack(initialize.initialize_state(agents[np.argwhere(condition)][i, :]) for i in range(len(state[condition])))).squeeze()

            dataset = np.concatenate((scene.reshape(self.nr_agents, 1), agents.reshape(self.nr_agents, 1), state[:, 0:2]), axis=1)
            np.savetxt(file, dataset, fmt="%.2f", delimiter="\t")

            states.append(state.copy())

        file.close()
        states = np.stack(states)
        print("dataset created.")

        # Show animation of simulation
        if self.create_video:
            print("start creating video for visualization of simulated scene " + str(self.scene) + "...")
            with self.visualize(states=states, space=space, sim=s) as _:
                pass

            print("video created.")


    def conv_image_to_space(self, input_image, create_background, scaling):
        """
        Read in image of scene and approximate obstacles in scene accordingly.
        :param input_image: Path to input image of scene
        :param create_background: if True, create background image with approximated obstacles
        :return: numpy array that determines boundaries of obstacles
        """
        space = []

        # Read in image
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
            ax.set_title(self.scene)
            ax.set_xlim(img.shape[1]*scaling)
            ax.set_ylim(img.shape[0]*scaling)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        # Create obstacles out of contour-lines
        for element in contours:
            epsilon = 0.02 * cv2.arcLength(element, True)  # approximated curve for epsilon = 2% of arc length empirically fitted best
            approx = cv2.approxPolyDP(element, epsilon, True)
            approx = np.append(approx, approx[0, :, :].reshape(1, 1, 2), axis=0)
            approx = (approx.squeeze()) * scaling

            for i in range(len(approx) - 1):
                space.append(np.linspace(approx[i, :], approx[i + 1, :], 100))

            if create_background:
                for s in space:
                    ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

        if create_background:
            fig.savefig(self.background_path)
            plt.close()

        return space

    # Calculate Resulting Potential
    def resulting_potential(self, state_at_scene, X, Y, s):
        """
        Calculate potential of Agent-Agent interaction-forces for visualization
        :param state_at_scene: state of agents at scene: nr_agents x 6
        :param s: socialforce
        :return: potential
        """
        v, v_x, v_y = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)
        for ped in range(len(state_at_scene)):
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    a = np.expand_dims(state_at_scene[ped, :].copy(), axis=0)
                    a = np.concatenate((a, np.array([[X[i, j], Y[i, j], 0, 0, 0, 0, 0.5]])))
                    pot = s.calc_potential(a)
                    v_x[i, j] = pot[0, 0]
                    v_y[i, j] = pot[0, 1]
                    v[i, j] = v[i, j] + np.linalg.norm([v_x[i, j], v_y[i, j]], axis=-1)

        return {"v": v, "v_x": v_x, "v_y": v_y}

    @contextmanager
    def visualize(self, states, sim, space=[]):
        """
        Visualize states of agents for each scene
        """
        with socialforce.show.animation(len(states), self.scene, self.video_path, writer='imagemagick', viz=self.viz) as context:
            ax = context['ax']
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')

            yield ax

            for s in space:
                ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

            actors = []
            for ped in range(states.shape[1]):
                radius = 0.2
                p = plt.Circle(states[0, ped, 0:2], radius=radius, facecolor='white', edgecolor='black')
                actors.append(p)
                ax.add_patch(p)

            def update(i):
                ax.clear()
                ax.grid(linestyle='dotted')
                ax.set_aspect(1.2, 'datalim')
                ax.set_axisbelow(True)
                ax.set_title(self.scene.upper() + " - V0: " + str(self.V0) + " - sigma: " + str(self.sigma) + " - U0: " + str(self.U0))
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_ylim([self.total_y_min - 2, self.total_y_max + 2])
                ax.set_xlim([self.total_x_min - 2, self.total_x_max + 2])
                for ped, p in enumerate(actors):
                    ax.add_patch(p)
                    p.center = states[i, ped, 0:2]
                    p.set_radius(radius)
                for s in space:
                    ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

                if self.plot_potentials:
                    x = np.arange(self.total_x_min, self.total_x_max, 0.3)
                    y = np.arange(self.total_y_min, self.total_y_max, 0.3)
                    X, Y = np.meshgrid(x, y)
                    v = self.resulting_potential(states[i, :, :], X, Y, sim)
                    CS2 = plt.contour(X, Y, v["v"])
                    cl2 = plt.clabel(CS2, inline=1, fontsize=10)

                print("scene nr. " + str(i) + " completed.")

            context['update_function'] = update


    def connect2visdom(self, viz_server, viz_port):
        """
        Connect to Visdom server
        """
        servername = socket.gethostname()
        if "node" in servername:
            self.server = viz_server
        else:
            self.server = 'http://localhost'

        return visdom.Visdom(server=self.server, port=viz_port, env="Socialforce_PedSpace")


if __name__ == "__main__":
    # Get input arguments from shell
    parser = argparse.ArgumentParser("Trajectory Prediction Basics")

    # General Configs about Scene
    parser.add_argument("--nr_agents", default=15, type=int, help="Specify number of agents in scenes")
    parser.add_argument("--nr_scenes", default=100, type=int, help="Specify number of scenes")
    parser.add_argument("--v_max", default=1.2, type=float, help="Specify v_max of agents in scenes")
    parser.add_argument("--v_min", default=0.6, type=float, help="Specify v_min of agents in scenes")
    parser.add_argument("--threshold", default=0.05, type=float, help="Specify threshold for agents to which degree they are allowed to overshoot their destination")

    parser.add_argument("--V0", default=-1, type=int, help="Specify V0: agent-agent potential")
    parser.add_argument("--sigma", default=-1.0, type=float, help="Specify range of V0")
    parser.add_argument("--U0", default=4, type=int, help="Specify U0: agent-obstacle potential")
    parser.add_argument("--r", default=1.5, type=float, help="Specify range of U0")

    parser.add_argument("--a", default=20., type=float, help="Specify width of scene")
    parser.add_argument("--b", default=25., type=float, help="Specify height of scene")

    parser.add_argument("--scaling", default=0.05, type=float, help="Specify scaling factor to map pixel of input image to coordinates of scene")

    # Configs about Script behavior
    parser.add_argument("--scene", default="hotel", type=str, help="Determine which scene to analyze")
    parser.add_argument("--create_background", default=False, type=bool_flag, help="Determine whether to show and save created background space for simulation")
    parser.add_argument("--create_video", default=False, type=bool_flag, help="Determine whether to create an mp4-video of the created sequence")
    parser.add_argument("--plot_potentials", default=False, type=bool_flag, help="Specify whether to plot potentials")

    # Configs about Visdom
    parser.add_argument("--visdom", default=False, type=bool_flag, help="specify whether plot loss in visdom")
    parser.add_argument("--viz_port", default=8098, type=int, help="specify port for visdom")
    parser.add_argument("--viz_server", default="http://atcremers10", type=str, help="specify server for visdom")


    # Get arguments
    args = parser.parse_args()

    # Create Sequence of Scenes
    Sequence = PedSpace_Interactions(args)
    Sequence.random_crossings()

