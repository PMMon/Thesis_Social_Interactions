import argparse

from helper.bool_flag import bool_flag
from Implementations.PedSpaceScene import PedSpace_Interactions

# ======================================================= Description =======================================================
# Main Script to generate synthetic datasets that incorporates interactions between pedestrians and obstacles, i.e. Human-Space interactions.
# In order to exclude Human-Space interactions and to particularly focus on Human-Human interactions (social interactions),
# run the Script GenPedPedScene.py in this folder.
#
# If the argument --run_list is set to true, datasets for different combinations of V0 and sigma, specified by
# V0_list and sigma_list, are created. Otherwise one single dataset with --V0 and --sigma is created.
#
# Use arguments listed below to specify additional configurations
#
# Realistic Configs for various datasets:
# ETH: V0: 2; sigma: 0.8686; U0: 4; r: 0.4343
# HOTEL: V0: 2; sigma: 0.8686; U0: 1; r: 0.4343; agent_radius: 0.25
# UNIV: V0: 2; sigma: 0.8686; U0: 2; r: 0.4343
# ZARA1: V0: 2; sigma: 0.8686; U0: 2; r: 0.4343; nr_agents: 6
# ZARA2: V0: 2; sigma: 0.8686; U0: 2; r: 0.4343; nr_agents: 6
#
# Note: the above recommendations depend on several other configurations, such as the number of agents in the scene,
# the scaling factor, etc. The recommendations refer to the default configurations below.
# ===========================================================================================================================

# Get input arguments from shell
parser = argparse.ArgumentParser("Generate datasets that include human-space interactions")

# General Configs
parser.add_argument("--run_list", default=False, type=bool_flag, help="Determine whether to run simulation for various values of V0 and sigma defined in two lists")
parser.add_argument("--nr_scenes", default=100, type=int, help="Specify number of scenes")
parser.add_argument("--scenario", default="zara1", type=str, help="Specify name of scenario that should be animated. Choose either zara1, zara2, univ, hotel or eth")
parser.add_argument("--phase", default="train", type=str, help="Specify for which phase data should be created. Choose either train, val or test")
parser.add_argument("--a", default=20., type=float, help="Specify width of square/rectangle")
parser.add_argument("--b", default=30., type=float, help="Specify length of rectangle")
parser.add_argument("--threshold", default=0.02, type=float, help="Specify threshold for agents getting out of scene")

# Configs for initial State
parser.add_argument("--nr_agents", default=8, type=int, help="Specify number of agents in scene")
parser.add_argument("--v_max", default=1.2, type=float, help="Specify v_max of agents in scenes")
parser.add_argument("--v_min", default=0.4, type=float, help="Specify v_min of agents in scenes")

# Configs for Potential and Forces
parser.add_argument("--V0", default=2, type=int, help="Specify magnitude V0 of agent-agent potential")
parser.add_argument("--sigma", default=0.8686, type=float, help="Specify range of agent-agent potential")
parser.add_argument("--U0", default=2, type=int, help="Specify magnitude U0 of agent-obstacle potential")
parser.add_argument("--r", default=0.4343, type=float, help="Specify range of agent-obstacle potential")
parser.add_argument("--tau", default=0.5, type=float, help="Specify relaxation time")
parser.add_argument("--beta", default=1, type=float, help="Specify factor for orthogonal force ratio (1 for none)")
parser.add_argument("--delta_t", default=0.4, type=float, help="Specify time for step size")
parser.add_argument("--twophi", default=200.0, type=float, help="Specify angle for visible range")
parser.add_argument("--c", default=0.5, type=float, help="Specify out-of-view factor")

# Configs for animation of dataset
parser.add_argument("--show_animation", default=False, type=bool_flag, help="Determine whether to create an mp4-video of the created sequence")
parser.add_argument("--show_potentials", default=False, type=bool_flag, help="Specify whether to plot potentials")
parser.add_argument("--agent_radius", default=0.35, type=float, help="Specify radius of circles that represent the agents in the dataset")
parser.add_argument("--create_background", default=False, type=bool_flag, help="Determine whether to show and save created background space for simulation")
parser.add_argument("--scaling", default=0.05, type=float, help="Specify scaling factor to map pixel of input image to coordinates of scene")

# Configs about Visdom
parser.add_argument("--visdom", default=False, type=bool_flag, help="specify whether show animations/images in visdom")
parser.add_argument("--viz_port", default=8098, type=int, help="specify port for visdom")
parser.add_argument("--viz_server", default="", type=str, help="specify server for visdom")
parser.add_argument("--viz_env", default="Socialforce_PedSpaceScene", type=str, help="Specify environment name for visdom")


# Get arguments
args = parser.parse_args()

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
            PedSpaceScene_val = PedSpace_Interactions(args)

            initial_agent_states = []
            states_of_Scene_val = PedSpaceScene_val.get_scene(initial_agent_states)

            # test:
            print("Create test set...")
            args.phase = "test"
            args.nr_scenes = 1800
            # Create Scene
            PedSpaceScene_test = PedSpace_Interactions(args)

            initial_agent_states = []
            states_of_Scene_test = PedSpaceScene_test.get_scene(initial_agent_states)

            # train:
            args.phase = "train"
            args.nr_scenes = 9000
            # Create Scene
            print("Create training set...")
            PedSpaceScene_train = PedSpace_Interactions(args)

            initial_agent_states = []
            states_of_Scene_train = PedSpaceScene_train.get_scene(initial_agent_states)

            print("datasets for V0: " + str(args.V0) + ", sigma: " + str(args.sigma) + ", U0: " + str(args.U0) + " and r: " + str(args.r) + " created.")

else:
    # Create dataset with --V0 and --sigma
    PedSpaceScene = PedSpace_Interactions(args)

    initial_agent_states = []
    states_of_Scene = PedSpaceScene.get_scene(initial_agent_states)