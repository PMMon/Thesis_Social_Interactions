# ==Imports==
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import text
from contextlib import contextmanager
import numpy as np
import socialforce
import argparse
import os
from os import strerror
import errno
import socket
import visdom

from Implementations.Scenarios import generate_scene
from helper.bool_flag import bool_flag
from helper import show
from Implementations.initialization import random_initialization


class PedSpace_Interactions(generate_scene):
    """
    Class that creates Sequences of scenes with Agent-Agent interactions and Agent-Obstacle interactions
    """
    def __init__(self, args):
        super(PedSpace_Interactions, self).__init__(args.scenario)
        self.nr_agents = args.nr_agents
        self.nr_scenes = args.nr_scenes

        self.scenario = args.scenario
        self.phase = args.phase
        self.scaling = args.scaling
        self.agent_radius = args.agent_radius

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
        self.show_animation = args.show_animation
        self.show_potentials = args.show_potentials

        # Specify file paths
        self.root_path_experiments = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Experiments"))
        if not os.path.exists(self.root_path_experiments):
            raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), "Can not find Experiments folder!")

        self.input_image_root = os.path.join(self.root_path, "docs", "real_scene_images")
        self.image_path = os.path.join(self.input_image_root, args.scenario + "-op.jpg")

        if not os.path.exists(self.input_image_root):
            raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), self.input_image_root)

        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), self.image_path)

        self.dataset_root = os.path.abspath(os.path.join(self.root_path_experiments, "datasets", "PedSpaceScenes", self.scenario + "_simulated", str(self.phase)))
        self.dataset_path = os.path.join(self.dataset_root,  str(self.phase) + "_" + str(self.scenario) + "simulated.txt")

        if not os.path.exists(self.dataset_root):
            os.makedirs(self.dataset_root)

        self.background_root = os.path.join(self.root_path, "docs", "simulated_background")
        self.background_path = os.path.join(self.background_root, args.scenario + "_simulated.jpg")

        if not os.path.exists(self.background_root):
            os.makedirs(self.background_root)

        self.video_root = os.path.join(self.root_path, "docs", "videos", "PedSpaceScene", self.scenario + "_simulated")
        self.video_path = os.path.join(self.video_root, args.scenario + "_simulated_" + str(self.phase) + ".mp4")

        if not os.path.exists(self.video_root):
            os.makedirs(self.video_root)

        if args.visdom:
            self.viz = self.connect2visdom(args.viz_server, args.viz_port, args.viz_env)
        else:
            self.viz = None


    def get_scene(self, agent_states):
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

        print("Start generating scenes for dataset %s..." % (self.scenario))

        # Initialize state for each agent
        self.initial_state = self.initialize_agents(self.v_max, self.v_min, self.nr_agents, agent_states, self.a, self.b, setting="random")

        # Get boundaries
        initialize = random_initialization(self.scenario, self.v_max, self.v_min, a=self.a, b=self.b, tau=0.5)
        self.total_y_min = initialize.total_y_min
        self.total_y_max = initialize.total_y_max
        self.total_x_min = initialize.total_x_min
        self.total_x_max = initialize.total_x_max


        # Initialize space according to input image
        space = self.initialize_space(self.image_path, self.create_background, self.scaling)

        # Get Socialforce Simulator
        s = socialforce.Simulator(self.initial_state, ped_space=socialforce.PedSpacePotential(space, u0=self.U0, r=self.r), v0=self.V0, sigma=self.sigma)

        # Get Array for scene numbers and agents
        scene = np.zeros(self.nr_agents)
        agents = np.arange(self.nr_agents) + 1

        # Concatenate arrays with respective x-y-coordinates
        dataset = np.concatenate((scene.reshape(self.nr_agents, 1), agents.reshape(self.nr_agents, 1), s.state[:, 0:2]), axis=1)

        # Create txt-file for dataset
        print("Create dataset as txt-file for scenario: %s" % (self.scenario))
        file = open(self.dataset_path, "w")
        np.savetxt(file, dataset, fmt="%.2f", delimiter="\t")
        file.close()

        file = open(self.dataset_path, "a")

        # Calculate states of each agent for every scene in the dataset
        states = []
        agent_list = []
        states.append(s.state.copy())

        for _ in range(self.nr_scenes):
            state = s.step().state
            agent_list.append(agents.copy())
            scene += 1

            # Avoid oscillations of pedestrians -> Agents that have been at the same place for a longer period of time are exit the scene and new agents enter it
            oscillation_length = 20  # corresponds to: 20 * 0.4 = 8 seconds

            if _ >= oscillation_length:
                dist = np.linalg.norm(states[_][:, 0:2] - states[_ - oscillation_length][:, 0:2], axis=1)
                agent_cond = agent_list[_] == agent_list[_ - oscillation_length]
                add_cond = (dist[:] < 0.8) & agent_cond
            else:
                add_cond = np.ones((state.shape[0])) > 1

            # Boundary conditions - Determine when agents leave the scene
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
        print("Dataset created.")

        # Show animation of simulation
        if self.show_animation:
            print("Start creating video for visualization of simulated scenario " + str(self.scenario) + "...")
            with self.animate_scenes(states=states, space=space, sim=s) as _:
                pass

            print("Video created.")


    @contextmanager
    def animate_scenes(self, states, sim, space=[]):
        """
        Visualize states of agents for each scene
        """
        with show.animation(len(states), self.scenario, self.video_path, writer='imagemagick', viz=self.viz) as context:
            ax = context['ax']
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')

            yield ax

            for s in space:
                ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

            actors = []
            for ped in range(states.shape[1]):
                p = plt.Circle(states[0, ped, 0:2], radius=self.agent_radius, facecolor='white', edgecolor='black')
                actors.append(p)
                ax.add_patch(p)

            def update(i):
                current_time = round(0.4 * i, 1)
                ax.clear()
                ax.grid(linestyle='dotted')
                #ax.set_aspect(1.2, 'datalim')
                ax.set_axisbelow(True)
                ax.set_title(self.scenario.upper() + " - V0: " + str(self.V0) + " - sigma: " + str(self.sigma) + " - U0: " + str(self.U0) + " - r: " + str(self.r))
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_ylim([self.total_y_min - 4, self.total_y_max + 4])
                ax.set_xlim([self.total_x_min - 4, self.total_x_max + 4])
                text(0.95, 0.95, str(current_time) + " s", ha='right', va='top', transform=ax.transAxes, color="r", fontstyle="oblique")

                for ped, p in enumerate(actors):
                    ax.add_patch(p)
                    p.center = states[i, ped, 0:2]
                    p.set_radius(self.agent_radius)
                for s in space:
                    ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

                if self.show_potentials:
                    x = np.arange(self.total_x_min, self.total_x_max, 0.3)
                    y = np.arange(self.total_y_min, self.total_y_max, 0.3)
                    X, Y = np.meshgrid(x, y)
                    v = self.resulting_potential(states[i, :, :], X, Y, sim)
                    CS2 = plt.contour(X, Y, v["v"])
                    cl2 = plt.clabel(CS2, inline=1, fontsize=10)

                print("scene nr. " + str(i) + " completed.")

            context['update_function'] = update


    def connect2visdom(self, viz_server, viz_port, viz_env):
        """
        Connect to Visdom server
        """
        servername = socket.gethostname()
        if "node" in servername:
            self.server = viz_server
        else:
            self.server = 'http://localhost'

        return visdom.Visdom(server=self.server, port=viz_port, env=viz_env)


if __name__ == "__main__":
    # Get input arguments from shell
    parser = argparse.ArgumentParser("Generate datasets that include human-space interactions")

    # General Configs
    parser.add_argument("--scenario", default="hotel", type=str,help="Specify name of scenario that should be animated. Choose either zara1, zara2, univ, hotel or eth")
    parser.add_argument("--nr_scenes", default=100, type=int, help="Specify number of scenes")
    parser.add_argument("--nr_agents", default=15, type=int, help="Specify number of agents in scenes")
    parser.add_argument("--phase", default="train", type=str, help="Specify for which phase data should be created. Choose either train, val or test")
    parser.add_argument("--a", default=20., type=float, help="Specify width of scene")
    parser.add_argument("--b", default=25., type=float, help="Specify height of scene")
    parser.add_argument("--threshold", default=0.05, type=float,help="Specify threshold for agents to which degree they are allowed to overshoot their destination")

    # Configs for initial State
    parser.add_argument("--nr_agents", default=14, type=int, help="Specify number of agents in scene")
    parser.add_argument("--v_max", default=1.2, type=float, help="Specify v_max of agents in scenes")
    parser.add_argument("--v_min", default=0.4, type=float, help="Specify v_min of agents in scenes")

    # Configs for Potential and Forces
    parser.add_argument("--V0", default=-1, type=int, help="Specify V0: agent-agent potential")
    parser.add_argument("--sigma", default=-1.0, type=float, help="Specify range of V0")
    parser.add_argument("--U0", default=4, type=int, help="Specify U0: agent-obstacle potential")
    parser.add_argument("--r", default=1.5, type=float, help="Specify range of U0")

    # Configs for animation of dataset
    parser.add_argument("--create_background", default=False, type=bool_flag, help="Determine whether to show and save created background space for simulation")
    parser.add_argument("--create_video", default=False, type=bool_flag, help="Determine whether to create an mp4-video of the created sequence")
    parser.add_argument("--plot_potentials", default=False, type=bool_flag, help="Specify whether to plot potentials")
    parser.add_argument("--scaling", default=0.05, type=float, help="Specify scaling factor to map pixel of input image to coordinates of scene")

    # Configs about Visdom
    parser.add_argument("--visdom", default=False, type=bool_flag, help="specify whether plot loss in visdom")
    parser.add_argument("--viz_port", default=8098, type=int, help="specify port for visdom")
    parser.add_argument("--viz_server", default="http://atcremers10", type=str, help="specify server for visdom")
    parser.add_argument("--viz_env", default="Socialforce_PedSpaceScene", type=str, help="Specify environment name for visdom")


    # Get arguments
    args = parser.parse_args()

    # Create Sequence of Scenes
    Sequence = PedSpace_Interactions(args)

    initial_agent_states = []
    Sequence.get_scene(initial_agent_states)

