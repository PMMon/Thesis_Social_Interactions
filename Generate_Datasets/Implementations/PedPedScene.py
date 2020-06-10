import matplotlib
matplotlib.use('Agg')
from contextlib import contextmanager
import matplotlib.pyplot as plt
import argparse
from pylab import text
import numpy as np
import os
from os import strerror
import errno

import socialforce
from Implementations.Scenarios import generate_scene
from helper import show
from helper.bool_flag import bool_flag
from helper.connect2visdm import connect2visdom
from Implementations.initialization import random_initialization, meeting

class PedPed_Interactions(generate_scene):
    """
    Class for the generation of datasets that focus on human-human interactions and exclude obstacles
    """
    def __init__(self, V0, sigma, delta_t, twophi, c, v_max, v_min, scenario, phase, tau, a, b, beta):
        self.initial_state = []
        self.bar = True
        self.V0 = V0
        self.sigma = sigma
        self.tau = tau
        self.delta_t = delta_t
        self.twophi = twophi
        self.c = c
        self.v_max = v_max
        self.v_min = v_min
        self.a = a
        self.b = b
        self.beta = beta
        self.phase = phase

        # Create unique identifier of dataset according to values of V0 and sigma of the form: V0 valueofV0 b valueofsigma
        self.parameterinfo = "V0"
        for element_V0 in str(self.V0).split("."):
            self.parameterinfo += str(element_V0) + "u"
        self.parameterinfo = self.parameterinfo[:-1] + "b"
        for element_sigma in str(self.sigma).split("."):
            self.parameterinfo += str(element_sigma) + "u"
        self.parameterinfo = self.parameterinfo[:-1]

        super(PedPed_Interactions, self).__init__(scenario)


    def get_scene(self, nr_agents, nr_scenes, agent_states, threshold, setting="random"):
        """
        Function to generate each scene of the dataset.
        The agents enter the scenario at one side of the setting and leave it at another side. An implementation of the Social Force Model is used to
        describe the agents motion behavior including the interactions between the agents.
        :param nr_agents: Number of agents in each scene
        :param nr_scenes: Number of scenes of the dataset
        :param agent_states: Numpy array of shape N x 6 for N initial agents in the dataset. Determines the initial states of the agents in the dataset.
        The row of the i-th agent needs to be of the form: [x, y, v_x, v_y, d_x, d_y]. If the states should be initialized randomly, use agent_states = [] as input.
        :param threshold: Specifies threshold for agents getting out of the scene (in meters)
        :param setting: Either random or meeting
        """
        print("Start generating scenes for dataset %s..." % (self.scenario))

        # Initialize scene
        if setting == "meeting":
            self.initialize = meeting(self.scenario, self.v_max, self.v_min, self.a, self.b, self.tau)
        else:
            self.initialize = random_initialization(self.scenario, self.v_max, self.v_min,  self.a, self.b, self.tau)

        self.initial_state = self.initialize_agents(self.v_max, self.v_min, nr_agents, agent_states, self.a, self.b, setting)

        # Initialize simulator
        self.s = socialforce.Simulator(self.initial_state, self.V0, self.sigma, self.delta_t, self.tau, beta=self.beta)

        # Get Array for scene numbers and agents
        scene = np.zeros(nr_agents)
        agents = np.arange(nr_agents) + 1

        # Concatenate arrays with respective x-y-coordinates
        dataset = np.concatenate((scene.reshape(nr_agents, 1), agents.reshape(nr_agents, 1), self.s.state[:, 0:2]), axis=1)

        # Specify file location
        root_path_experiments = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Experiments"))
        if not os.path.exists(root_path_experiments):
            raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), "Can not find Experiments folder!")

        root_path_dataset = os.path.abspath(os.path.join(root_path_experiments, "datasets", str(self.scenario) + "simulated" + "_" + str(self.parameterinfo),  str(self.phase)))

        if not os.path.exists(root_path_dataset):
            os.makedirs(root_path_dataset)

        # Create background image
        fig, ax = plt.subplots()
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_axisbelow(True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_ylim([self.initialize.total_y_min - 2, self.initialize.total_y_max + 2])
        ax.set_xlim([self.initialize.total_x_min - 2, self.initialize.total_x_max + 2])
        ax.set_title(self.scenario + " V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
        fig.set_tight_layout(True)
        fig.savefig(root_path_dataset+"//"+str(self.scenario) + "simulated.jpg")
        plt.close()

        # Create txt-file for dataset
        txt_file = os.path.join(root_path_dataset, str(self.phase) + "_" + str(self.scenario) + "simulated.txt")
        print("Create dataset as txt-file for scenario: " + str(self.scenario))
        file = open(txt_file, "w")
        np.savetxt(file, dataset, fmt="%.2f", delimiter="\t")
        file.close()
        file = open(txt_file, "a")

        # Calculate states of each agent for every scene in the dataset
        states = []
        states.append(self.s.state.copy())
        agent_list = []

        for _ in range(nr_scenes):
            state = self.s.step().state
            agent_list.append(agents.copy())
            scene += 1

            # Avoid oscillations of pedestrians -> Agents that have been at the same place for a longer period of time are exit the scene and new agents enter it
            oscillation_length = 20   # corresponds to: 20 * 0.4 = 8 seconds

            if _ >= oscillation_length:
                dist = np.linalg.norm(states[_][:,0:2] - states[_-oscillation_length][:,0:2], axis=1)
                agent_cond = agent_list[_] == agent_list[_-oscillation_length]
                add_cond = (dist[:] < 0.8) & agent_cond
            else:
                add_cond =  np.ones((state.shape[0])) > 1

            # Boundary conditions - Determine when agents leave the scene
            condition = (state[:, 0] > self.initialize.total_x_max + threshold) | (state[:, 0] < self.initialize.total_x_min - threshold) | (state[:, 1] > self.initialize.total_y_max + threshold) | (state[:, 1] < self.initialize.total_y_min - threshold) | add_cond

            agents[np.argwhere(condition)] = (max(agents) + np.arange(len(np.argwhere(condition))) + 1).reshape(-1, 1)

            # Generate new Agents when old Agents left the scene
            if len(state[condition]) != 0:
                state[condition] = (np.stack(self.initialize.initialize_state(agents[np.argwhere(condition)][i,:]) for i in range(len(state[condition])))).squeeze()

            dataset = np.concatenate((scene.reshape(nr_agents, 1), agents.reshape(nr_agents, 1), state[:, 0:2]), axis=1)

            # Write states to txt-file
            np.savetxt(file, dataset, fmt="%.2f", delimiter="\t")

            states.append(state.copy())

        file.close()
        states = np.stack(states)

        return states

    @contextmanager
    def animate_scenes(self, states, show_potential, agent_radius, viz):
        """
        Takes the states of the agents for every scene in the dataset to generate a video that shows the respective movements of the agents.
        :param states: Numpy-array of states of each pedestrian for every scene of the dataset
        """
        print("Creating video for visualization of simulated scene " + str(self.scenario) + "...")

        # Specify file location
        video_path =  os.path.abspath(os.path.join(self.root_path, "docs", "videos", "PedPedScene", str(self.scenario) + "simulated" + "_" + str(self.parameterinfo)))

        if not os.path.exists(video_path):
            os.makedirs(video_path)

        video_file =  os.path.join(video_path, str(self.scenario) + "_" + str(self.parameterinfo) + "_" + str(self.phase) + ".mp4")

        mode = video_file.split(".")[-2].split("/")[-1]

        # Create animation
        with show.animation(len(states), mode, video_file, writer='imagemagick', viz=viz) as context:
            ax = context['ax']
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set(xlim=(self.initialize.total_x_min - 2, self.initialize.total_x_max + 2), ylim=(self.initialize.total_y_min - 2, self.initialize.total_y_max + 2))

            yield ax

            actors = []
            for ped in range(states.shape[1]):
                p = plt.Circle(states[0, ped, 0:2], radius=agent_radius, facecolor='white', edgecolor='black')
                actors.append(p)
                ax.add_patch(p)

            def update(i):
                current_time = round(0.4*i, 1)
                ax.clear()
                ax.grid(linestyle='dotted')
                ax.set_axisbelow(True)
                ax.set_title("V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set(xlim=(self.initialize.total_x_min-2, self.initialize.total_x_max+2), ylim=(self.initialize.total_y_min-2, self.initialize.total_y_max+2))
                text(0.95, 0.95, str(current_time) + " s", ha='right', va='top', transform=ax.transAxes, color="r", fontstyle="oblique")

                for ped, p in enumerate(actors):
                    ax.add_patch(p)
                    p.center = states[i, ped, 0:2]
                    p.set_radius(agent_radius)

                x = np.arange(self.initialize.total_x_min, self.initialize.total_x_max, 0.3)
                y = np.arange(self.initialize.total_y_min, self.initialize.total_y_max, 0.3)

                # Enable below for plotting repulsive potentials between pedestrians (high comp costs)
                if show_potential:
                    X, Y = np.meshgrid(x, y)
                    v = self.resulting_potential(states[i,:,:], X, Y, self.s)
                    CS2 = plt.contour(X, Y, v["v"])
                    cl2 = plt.clabel(CS2, inline=1, fontsize=10)
                    if self.bar:
                        col = plt.colorbar()
                        col.ax.set_ylabel('Vges')
                        self.bar = False

                print("scene nr. " + str(i) + " completed.")

            context['update_function'] = update

        print("animation created.")



class PedPedScene_constantly_present(PedPed_Interactions):
    """
    Subclass of PedPed_Interactions. Creates a dataset for which a number of agents is constantly present in a certain environment.
    After seq_len time steps, all agents are replaced by an equal number of new agents. Can be used e.g. to ensure that all agents
    are seq_len time steps in the scene.
    """
    def __init__(self, V0, sigma, delta_t, twophi, c, v_max, v_min, scenario, phase, tau, a, b, beta, seq_len=20):
        self.seq_len = seq_len
        super(PedPedScene_constantly_present, self).__init__(V0, sigma, delta_t, twophi, c, v_max, v_min, scenario, phase, tau, a, b, beta=beta)


    def get_scene(self, nr_agents, nr_scenes, agent_states, threshold, setting="random"):
        """
        Method overriding. In this case agents do not leave the scenario at a specific side of the scene, but remain seq_len time steps in the scene.
        After seq_len time steps all agents are replaced by an equal number of new agents. An implementation of the Social Force Model is used to
        describe the agents motion behavior including the interactions between the agents.
        :param nr_agents: Number of agents in each scene
        :param nr_scenes: Number of scenes of the dataset
        :param agent_states: Numpy array of shape N x 6 for N initial agents in the dataset. Determines the initial states of the agents in the dataset.
        The row of the i-th agent needs to be of the form: [x, y, v_x, v_y, d_x, d_y]. If the states should be initialized randomly, use agent_states = [] as input.
        :param threshold: Specifies the threshold for agents getting out of the scene (in meters)
        :param setting: Either random or meeting
        """

        if setting == "meeting":
            self.initialize = meeting(self.scenario, self.v_max, self.v_min, self.a, self.b, self.tau)
        else:
            self.initialize = random_initialization(self.scenario, self.v_max, self.v_min,  self.a, self.b, self.tau)

        self.initial_state = self.initialize_agents(self.v_max, self.v_min, nr_agents, agent_states, self.a, self.b, setting)

        self.s = socialforce.Simulator(self.initial_state, self.V0, self.sigma, self.delta_t, self.tau, beta=self.beta)

        # Get Array for scene numbers and agents
        scene = np.zeros(nr_agents)
        agents = np.arange(nr_agents) + 1

        # Concatenate arrays with respective x-y-coordinates
        dataset = np.concatenate((scene.reshape(nr_agents, 1), agents.reshape(nr_agents, 1), self.s.state[:, 0:2]), axis=1)

        # Specify file location
        root_path_experiments = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Experiments"))
        if not os.path.exists(root_path_experiments):
            raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), "Can not find Experiments folder!")

        root_path_dataset = os.path.abspath(os.path.join(root_path_experiments, "datasets", str(self.scenario) + "simulated" + "_" + str(self.parameterinfo),  str(self.phase)))

        if not os.path.exists(root_path_dataset):
            os.makedirs(root_path_dataset)

        # Create background image
        fig, ax = plt.subplots()
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_axisbelow(True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_ylim([self.initialize.total_y_min - 2, self.initialize.total_y_max + 2])
        ax.set_xlim([self.initialize.total_x_min - 2, self.initialize.total_x_max + 2])
        ax.set_title(self.scenario + " V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
        fig.set_tight_layout(True)
        fig.savefig(root_path_dataset+"//"+str(self.scenario) + "simulated.jpg")
        plt.close()

        txt_file = os.path.join(root_path_dataset, str(self.phase) + "_" + str(self.scenario) + "simulated.txt")
        print("Create dataset as txt-file for scene " + str(self.scenario))
        file = open(txt_file, "w")
        np.savetxt(file, dataset, fmt="%.2f", delimiter="\t")
        file.close()
        file = open(txt_file, "a")

        # Repeat for each scene and update according steps of agents
        states = []
        states.append(self.s.state.copy())

        for _ in range(nr_scenes):
            state = self.s.step().state
            scene += 1

            # Periodic Boundary condition
            if _ != 0 and (_ + 1) % self.seq_len == 0:
                # Generate new Agents after seq_len timesteps
                state[:] = np.stack(self.initialize.initialize_state(i) for i in range(len(state))).squeeze()
                # Update IDs
                agents += len(agents)
            dataset = np.concatenate((scene.reshape(nr_agents, 1), agents.reshape(nr_agents, 1), state[:, 0:2]), axis=1)
            np.savetxt(file, dataset, fmt="%.2f", delimiter="\t")

            states.append(state.copy())

        file.close()
        states = np.stack(states)

        return states


if __name__ == "__main__":
    # Get input arguments from shell
    parser = argparse.ArgumentParser("Generate Datasets with Social Force Model")

    # General Configs
    parser.add_argument("--nr_scenes", default=100, type=int, help="Specify number of scenes")
    parser.add_argument("--run_list", default=False, type=bool_flag, help="Determine whether to run simulation for various values of V0 and sigma defined in two lists")
    parser.add_argument("--dataset", default="", type=str, help="Specify name of scene that should be animated")
    parser.add_argument("--phase", default="train", type=str, help="Specify for which phase data should be created. Choose either train, val or test")
    parser.add_argument("--a", default=20, type=float, help="Specify width of square/rectangle")
    parser.add_argument("--b", default=30, type=float, help="Specify length of rectangle")
    parser.add_argument("--threshold", default=0.02, type=float, help="Specify threshold for agents getting out of scene")

    # Configs for animation of dataset
    parser.add_argument("--show_animation", default=False, type=bool_flag, help="Determine whether to show and save animation of dataset or not")
    parser.add_argument("--show_potential", default=False, type=bool_flag, help="Determine whether to visualize repulsive pedpedpotential in animation or not (high comp costs)")
    parser.add_argument("--agent_radius", default=0.2, type=float, help="Specify radius of circles that represent the agents in the dataset")

    # Configs for initial State
    parser.add_argument("--nr_agents", default=14, type=int, help="Specify number of agents in scene")
    parser.add_argument("--v_max", default=1.2, type=float, help="Specify v_max")
    parser.add_argument("--v_min", default=0.4, type=float, help="Specify v_min")

    # Configs for Potential and Forces
    parser.add_argument("--V0", default=2, type=int, help="Specify V0 of PedPedPotential")
    parser.add_argument("--sigma", default=1.303, type=float, help="Specify sigma of PedPedPotential")
    parser.add_argument("--delta_t", default=0.4, type=float, help="Specify time for step size")
    parser.add_argument("--twophi", default=200.0, type=float, help="Specify angle for visible range")
    parser.add_argument("--c", default=0.5, type=float, help="Specify out-of-view factor")
    parser.add_argument("--tau", default=0.5, type=float, help="Specify relaxation time")
    parser.add_argument("--U0", default=2.1, type=float, help="Specify U0 of PedSpacePotential")
    parser.add_argument("--r", default=1, type=float, help="Specify r of PedSpacePotential")
    parser.add_argument("--beta", default=1, type=float, help="Specify factor for orthogonal force ratio (1 for none)")

    # Configs about Visdom
    parser.add_argument("--visdom", default=False, type=bool_flag, help="Specify whether plot loss in visdom")
    parser.add_argument("--viz_port", default=8098, type=int, help="Specify port for visdom")
    parser.add_argument("--viz_server", default="http://atcremers10", type=str, help="Specify server for visdom")
    parser.add_argument("--viz_env", default="Socialforce_PedPedScene", type=str, help="Specify environment name for visdom")

    parser.add_argument("--scene", default="hotel", type=str, help="Determine which scene to analyze")

    # Get arguments
    args = parser.parse_args()

    if args.visdom:
        viz = connect2visdom(args.viz_server, args.viz_port, args.viz_env)
    else:
        viz = None

    # Create dataset with --V0 and --sigma
    PedPedScene = PedPedScene_constantly_present(args.V0, args.sigma, args.delta_t, args.twophi, args.c, args.v_max, args.v_min, args.dataset, args.phase, tau=args.tau, a=args.a, b=args.b, beta=args.beta)

    initial_agent_states = []
    states_of_Scene = PedPedScene.get_scene(args.nr_agents, args.nr_scenes, initial_agent_states, threshold=args.threshold)

    if args.show_animation:
        # Create animation of the dataset generated
        with PedPedScene.animate_scene(states_of_Scene, args.show_potential, args.agent_radius, viz) as _:
            pass
