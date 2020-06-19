import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from helper.bool_flag import bool_flag
from helper.connect2visdm import connect2visdom
from Implementations import PedPedScene

# ======================================================= Description =======================================================
# Main Script to generate synthetic datasets that focus on the interactions between pedestrians with the Social Force model.
# The dataset excludes Human-Space interactions, i.e. interactions between pedestrians and their physical environment (obstacles).
# In order to include Human-Space interactions run the Script GenPedSpaceScene.py in this folder.
#
# If the argument --run_list is set to true, datasets for different combinations of V0 and sigma, specified by
# V0_list and sigma_list, are created. Otherwise one single dataset with --V0 and --sigma is created.
#
# Used values for V0 and sigma in Thesis:
# V0 = [0, 1, 2, 4, 6]
# sigma = [0.2171, 0.4343, 0.8686, 1.303, 1.7371, 2.171, 2.6058]
#
# Use arguments listed below to specify additional configurations
# ===========================================================================================================================

# Get input arguments from shell
parser = argparse.ArgumentParser("Generate datasets that focus on social interactions")

# General Configs
parser.add_argument("--run_list", default=False, type=bool_flag, help="Determine whether to run simulation for various values of V0 and sigma defined in two lists")
parser.add_argument("--nr_scenes", default=100, type=int, help="Specify number of scenes")
parser.add_argument("--scenario", default="square", type=str, help="Specify name of scenario that should be animated. Choose either zara1, zara2, univ, hotel, eth, square or rectangle.")
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
parser.add_argument("--U0", default=2, type=int, help="Specify magnitude U0 of agent-obstacle potential")
parser.add_argument("--r", default=0.4343, type=float, help="Specify range of agent-obstacle potential")
parser.add_argument("--tau", default=0.5, type=float, help="Specify relaxation time")
parser.add_argument("--beta", default=1, type=float, help="Specify factor for orthogonal force ratio (1 for none)")
parser.add_argument("--delta_t", default=0.4, type=float, help="Specify time for step size")
parser.add_argument("--twophi", default=200.0, type=float, help="Specify angle for visible range")
parser.add_argument("--c", default=0.5, type=float, help="Specify out-of-view factor")

# Configs about Visdom
parser.add_argument("--visdom", default=False, type=bool_flag, help="Specify whether show animations/videos in visdom")
parser.add_argument("--viz_port", default=8090, type=int, help="Specify port for visdom")
parser.add_argument("--viz_server", default="", type=str, help="Specify server for visdom")
parser.add_argument("--viz_env", default="Socialforce_PedPedScene", type=str, help="Specify environment name for visdom")


# Get arguments
args = parser.parse_args()

if args.visdom:
    viz = connect2visdom(args.viz_server, args.viz_port, args.viz_env)
else:
    viz = None

if args.run_list:

    # For each unique combination of V0 and sigma a training, validation and testing set are created
    V0_list = [0, 1, 2, 4, 6]
    sigma_list = [0.2171, 0.4343, 0.8686, 1.303, 1.7371, 2.171, 2.6058]

    for V0 in V0_list:
        for sigma in sigma_list:
            args.V0 = V0
            args.sigma = sigma

            # validation:
            args.phase = "val"
            args.nr_scenes = 1800
            # Create Scene
            print("Create validation set...")
            PedPedScene_val = PedPedScene.PedPed_Interactions(args.V0, args.sigma, args.delta_t, args.twophi, args.c, args.v_max, args.v_min, args.scenario, args.phase, tau=args.tau, a=args.a, b=args.b, beta=args.beta)

            initial_agent_states = []
            states_of_Scene_val = PedPedScene_val.get_scene(args.nr_agents, args.nr_scenes, initial_agent_states, threshold=args.threshold)

            # test:
            args.phase = "test"
            args.nr_scenes = 1800
            # Create Scene
            print("Create test set...")
            PedPedScene_test = PedPedScene.PedPed_Interactions(args.V0, args.sigma, args.delta_t, args.twophi, args.c, args.v_max, args.v_min, args.scenario, args.phase, tau=args.tau, a=args.a, b=args.b, beta=args.beta)

            initial_agent_states = []
            states_of_Scene_test = PedPedScene_test.get_scene(args.nr_agents, args.nr_scenes, initial_agent_states, threshold=args.threshold)

            # train:
            args.phase = "train"
            args.nr_scenes = 9000
            # Create Scene
            print("Create training set...")
            PedPedScene_train = PedPedScene.PedPed_Interactions(args.V0, args.sigma, args.delta_t, args.twophi, args.c, args.v_max, args.v_min, args.scenario, args.phase, tau=args.tau, a=args.a, b=args.b, beta=args.beta)

            initial_agent_states = []
            states_of_Scene_train = PedPedScene_train.get_scene(args.nr_agents, args.nr_scenes, initial_agent_states, threshold=args.threshold)

            print("datasets for V0: " + str(args.V0) + " and sigma: " + str(args.sigma) + " created.")

            if args.show_animation:
                # Create animation of the datasets generated

                # validation
                with PedPedScene_val.animate_scenes(states_of_Scene_val, args.show_potential, args.agent_radius, viz) as _:
                    pass

                # testing
                with PedPedScene_test.animate_scenes(states_of_Scene_test, args.show_potential, args.agent_radius, viz) as _:
                    pass

                # training
                with PedPedScene_train.animate_scenes(states_of_Scene_train, args.show_potential, args.agent_radius, viz) as _:
                    pass

else:
    # Create dataset with --V0 and --sigma
    PedPedScene = PedPedScene.PedPed_Interactions(args.V0, args.sigma, args.delta_t, args.twophi, args.c, args.v_max, args.v_min, args.scenario, args.phase, tau=args.tau, a=args.a, b=args.b, beta=args.beta)

    initial_agent_states = []
    states_of_Scene = PedPedScene.get_scene(args.nr_agents, args.nr_scenes, initial_agent_states, threshold=args.threshold)

    if args.show_animation:
        # Create animation of the dataset generated
        with PedPedScene.animate_scenes(states_of_Scene, args.show_potential, args.agent_radius, viz) as _:
            pass