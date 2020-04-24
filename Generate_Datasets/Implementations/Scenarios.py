# ==Imports==
import matplotlib
matplotlib.use('Agg')
from contextlib import contextmanager
import matplotlib.pyplot as plt
from pylab import text
from Implementations.initialization import random_initialization, meeting
import numpy as np
import cv2
import imutils
import socialforce
import os

class generate_scene:
    def __init__(self, dataset):
        self.dataset = dataset

    def initialize_agents(self, v_max, v_min, nr_agents, agent_state, a, b, scenario):
        initial_state = agent_state

        if scenario == "meeting":
            initialize = meeting(self.dataset, v_max, v_min, a, b, tau=0.5)
        else:
            initialize = random_initialization(self.dataset, v_max, v_min, a, b, tau=0.5)

        for i in range(nr_agents-len(initial_state)):
            if i == 0 and not initial_state:
                initial_state = initialize.initialize_state(i+1)
            else:
                initial_state = np.append(initial_state, initialize.initialize_state(i+1), axis=0)

        return initial_state

    def initialize_space(self, input_image):
        space = []
        scene = input_image.split("-op")[0].split("//")[-1]

        if scene != self.dataset:
            print("Input image displays scene " + str(scene) + ", but dataset is " + str(self.dataset) + "!")

        img = cv2.imread(input_image, 0)
        img = img[::-1, :]
        img1 = imutils.resize(img)
        ret, thresh = cv2.threshold(img1, 0, 255, 1)
        image_not_used, contours, hierarchy = cv2.findContours(thresh, 0, 2)

        for element in contours:
            epsilon = 0.02 * cv2.arcLength(element,True)  # approximated curve for epsilon = 2% of arc length empirically fitted best
            approx = cv2.approxPolyDP(element, epsilon, True)
            approx = np.append(approx, approx[0, :, :].reshape(1, 1, 2), axis=0)
            approx = (approx.squeeze()) * 0.05

            for i in range(len(approx) - 1):
                space.append(np.linspace(approx[i, :], approx[i + 1, :], 100))

        return space

class PedPedScene(generate_scene):
    def __init__(self, V0, sigma, delta_t, twophi, c, v_max, v_min, dataset, phase, tau, a, b, beta):
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
        self.initial_state = []
        self.bar = True
        self.beta = beta
        self.phase = phase
        self.parameterinfo = "V0"
        for element_V0 in str(self.V0).split("."):
            self.parameterinfo += str(element_V0) + "u"
        self.parameterinfo = self.parameterinfo[:-1] + "b"
        for element_sigma in str(self.sigma).split("."):
            self.parameterinfo += str(element_sigma) + "u"
        self.parameterinfo = self.parameterinfo[:-1]
        super(PedPedScene, self).__init__(dataset)

    def get_scene(self, nr_agents, nr_scenes, agent_states, threshold, scenario="random"):
        if scenario == "meeting":
            self.initialize = meeting(self.dataset, self.v_max, self.v_min, self.a, self.b, self.tau)
        else:
            self.initialize = random_initialization(self.dataset, self.v_max, self.v_min,  self.a, self.b, self.tau)

        self.initial_state = self.initialize_agents(self.v_max, self.v_min, nr_agents, agent_states, self.a, self.b, scenario)

        self.s = socialforce.Simulator(self.initial_state, self.V0, self.sigma, self.delta_t, self.tau, beta=self.beta)

        # Get Array for scene numbers and agents
        scene = np.zeros(nr_agents)
        agents = np.arange(nr_agents) + 1

        # Concatenate arrays with respective x-y-coordinates
        dataset = np.concatenate((scene.reshape(nr_agents, 1), agents.reshape(nr_agents, 1), self.s.state[:, 0:2]), axis=1)

        # Write Result to txt-file
        root_path = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "datasets", str(self.dataset) + "simulated" + "_" + str(self.parameterinfo),  str(self.phase))
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        fig, ax = plt.subplots()
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_axisbelow(True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_ylim([self.initialize.total_y_min - 2, self.initialize.total_y_max + 2])
        ax.set_xlim([self.initialize.total_x_min - 2, self.initialize.total_x_max + 2])
        ax.set_title(self.dataset + " V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
        fig.set_tight_layout(True)
        fig.savefig(root_path+"//"+str(self.dataset) + "simulated.jpg")
        plt.close()

        txt_file = root_path + "//" + str(self.phase) + "_" + str(self.dataset) + "simulated.txt"
        print("create dataset as txt-file for scene " + str(self.dataset) + " under " + str(txt_file))
        file = open(txt_file, "w")
        np.savetxt(file, dataset, fmt="%.2f", delimiter="\t")
        file.close()
        file = open(txt_file, "a")

        # Repeat for each scene and update according steps of agents
        states = []
        states.append(self.s.state.copy())
        agent_list = []

        for _ in range(nr_scenes):
            state = self.s.step().state
            agent_list.append(agents.copy())
            scene += 1

            # avoid oscillations of pedestrians
            if _ >= 10:
                dist = np.linalg.norm(states[_][:,0:2] - states[_-10][:,0:2], axis=1)
                agent_cond = agent_list[_] == agent_list[_-10]
                add_cond = (dist[:] < 0.8) & agent_cond
            else:
                add_cond =  np.ones((state.shape[0])) > 1

            # periodic boundary conditions
            condition = (state[:, 0] > self.initialize.total_x_max + threshold) | (state[:, 0] < self.initialize.total_x_min - threshold) | (state[:, 1] > self.initialize.total_y_max + threshold) | (state[:, 1] < self.initialize.total_y_min - threshold) | add_cond

            agents[np.argwhere(condition)] = (max(agents) + np.arange(len(np.argwhere(condition))) + 1).reshape(-1, 1)
            # Generate new Agents when old Agents left the scene
            if len(state[condition]) != 0:
                state[condition] = (np.stack(self.initialize.initialize_state(agents[np.argwhere(condition)][i,:]) for i in range(len(state[condition])))).squeeze()

            dataset = np.concatenate((scene.reshape(nr_agents, 1), agents.reshape(nr_agents, 1), state[:, 0:2]), axis=1)
            np.savetxt(file, dataset, fmt="%.2f", delimiter="\t")

            states.append(state.copy())

        file.close()
        states = np.stack(states)

        return states

    def contours(self, X, Y, state):
        F, F_x, F_y = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                a = state.copy()
                a[0, 0:2] = [X[i, j], Y[i, j]]
                Force = self.s.calc_force(a)
                F_x[i, j] = Force[0, 0]
                F_y[i, j] = Force[0, 1]
                F[i, j] = np.linalg.norm([F_x[i, j], F_y[i, j]], axis=-1)
        return {"F": F, "F_x": F_x, "F_y": F_y}

    # Calculate Resulting Potential
    def resulting_potential(self, state_at_scene, X, Y):
        v, v_x, v_y = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)
        for ped in range(len(state_at_scene)):
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    a = np.expand_dims(state_at_scene[ped,:].copy(), axis=0)
                    a = np.concatenate((a,np.array([[X[i,j], Y[i,j], 0, 0, 0, 0, 0.5]])))
                    pot = self.s.calc_potential(a)
                    v_x[i, j] = pot[0, 0]
                    v_y[i, j] = pot[0, 1]
                    v[i,j] = v[i,j] + v_y[i, j]

        return {"v": v, "v_x": v_x, "v_y": v_y}


    def pathways(self, states):
        # Visualize
        path = "./docs/simulated_images/train_" + str(self.dataset) + str(self.parameterinfo) + ".jpg"
        with socialforce.show.canvas(path) as ax:
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title(self.dataset + " V0: " + str(self.V0) + " - sigma: " + str(self.sigma))

            for ped in range(len(self.initial_state)):
                x = states[:, ped, 0]
                y = states[:, ped, 1]
                ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
            ax.legend()

        return

    @contextmanager
    def animate_scene(self, states):
        print("start creating video for visualization of simulated scene " + str(self.dataset) + "...")
        output_filename = 'docs/simulated_videos/' + str(self.dataset) + str(self.parameterinfo) + 'simulated.mp4'
        mode = output_filename.split(".")[-2].split("/")[-1]

        with socialforce.show.animation(len(states), mode, output_filename, writer='imagemagick') as context:
            ax = context['ax']
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set(xlim=(self.initialize.total_x_min - 2, self.initialize.total_x_max + 2),ylim=(self.initialize.total_y_min - 2, self.initialize.total_y_max + 2))

            yield ax

            actors = []
            for ped in range(states.shape[1]):
                radius = 0.2
                p = plt.Circle(states[0, ped, 0:2], radius=radius, facecolor='white', edgecolor='black')
                actors.append(p)
                ax.add_patch(p)

            def update(i):
                current_time = round(0.4*i,1)
                ax.clear()
                ax.grid(linestyle='dotted')
                #ax.set_aspect(1.0, 'datalim')
                ax.set_axisbelow(True)
                ax.set_title("V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set(xlim=(self.initialize.total_x_min-2, self.initialize.total_x_max+2), ylim=(self.initialize.total_y_min-2, self.initialize.total_y_max+2))
                text(0.95, 0.95,str(current_time) + " s", ha='right', va='top', transform=ax.transAxes, color="r", fontstyle="oblique")
                for ped, p in enumerate(actors):
                    ax.add_patch(p)
                    p.center = states[i, ped, 0:2]
                    p.set_radius(radius)
                x = np.arange(self.initialize.total_x_min, self.initialize.total_x_max, 0.3)
                y = np.arange(self.initialize.total_y_min, self.initialize.total_y_max, 0.3)
                X, Y = np.meshgrid(x, y)
                # Enable below for plotting potentials (high comp costs)
                #v = self.resulting_potential(states[i,:,:], X,Y)
                #CS2 = plt.contour(X, Y, v["v"])
                #cl2 = plt.clabel(CS2, inline=1, fontsize=10)
                if self.bar:
                    #col = plt.colorbar()
                    #col.ax.set_ylabel('Vges')
                    self.bar = False

                print("scene nr. " + str(i) + " completed.")
            context['update_function'] = update

        print("animation created.")


class PedPedScene_constantly_present(PedPedScene):
    def __init__(self, V0, sigma, delta_t, twophi, c, v_max, v_min, dataset, phase, tau, a, b, seq_len, beta):
        self.seq_len = seq_len
        super(PedPedScene_constantly_present, self).__init__(V0, sigma, delta_t, twophi, c, v_max, v_min, dataset, phase, tau, a, b, beta=beta)

    def get_scene(self, nr_agents, nr_scenes, agent_states, threshold, scenario="random"):
        if scenario == "meeting":
            self.initialize = meeting(self.dataset, self.v_max, self.v_min, self.a, self.b, self.tau)
        else:
            self.initialize = random_initialization(self.dataset, self.v_max, self.v_min,  self.a, self.b, self.tau)

        self.initial_state = self.initialize_agents(self.v_max, self.v_min, nr_agents, agent_states, self.a, self.b, scenario)

        self.s = socialforce.Simulator(self.initial_state, self.V0, self.sigma, self.delta_t, self.tau, beta=self.beta)

        # Get Array for scene numbers and agents
        scene = np.zeros(nr_agents)
        agents = np.arange(nr_agents) + 1

        # Concatenate arrays with respective x-y-coordinates
        dataset = np.concatenate((scene.reshape(nr_agents, 1), agents.reshape(nr_agents, 1), self.s.state[:, 0:2]), axis=1)

        # Write Result to txt-file
        root_path = os.path.join(os.path.expanduser("~"), "TrajectoryPredictionBasics", "datasets", str(self.dataset) + "simulated" + "_" + str(self.parameterinfo),  str(self.phase))
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        fig, ax = plt.subplots()
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_axisbelow(True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_ylim([self.initialize.total_y_min - 2, self.initialize.total_y_max + 2])
        ax.set_xlim([self.initialize.total_x_min - 2, self.initialize.total_x_max + 2])
        ax.set_title(self.dataset + " V0: " + str(self.V0) + " - sigma: " + str(self.sigma))
        fig.set_tight_layout(True)
        fig.savefig(root_path+"//"+str(self.dataset) + "simulated.jpg")
        plt.close()

        txt_file = root_path + "//" + str(self.phase) + "_" + str(self.dataset) + "simulated.txt"
        print("create dataset as txt-file for scene " + str(self.dataset))
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

            # periodic boundary conditions
            if _ != 0 and (_ + 1)%self.seq_len==0:
                # Generate new Agents after 20 timesteps
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
    print("main")
