import argparse
from Implementations.bool_flag import bool_flag
from Implementations import Scenarios

# Get input arguments from shell
parser = argparse.ArgumentParser("Trajectory Prediction Basics")

# General Configs
parser.add_argument("--nr_scenes", default=100, type=int, help="Specify number of scenes")
parser.add_argument("--show_animation", default=False, type=bool_flag, help="Determine whether to show and save animation or not")
parser.add_argument("--run_list", default=False, type=bool_flag, help="Determine whether to run simulation for lists of V0 and sigma")
parser.add_argument("--dataset", default="", type=str, help="Specify which scene should be animated")
parser.add_argument("--phase", default="train", type=str, help="Specify for which phase data should be created")
parser.add_argument("--a", default=20, type=float, help="Specify width")
parser.add_argument("--b", default=30, type=float, help="Specify length")
parser.add_argument("--threshold", default=0.02, type=float, help="Specify threshold for agents getting out of scene")

# Configs for initial State
parser.add_argument("--nr_agents", default=8, type=int, help="Specify number of agents")
parser.add_argument("--v_max", default=1.2, type=float, help="Specify v_max")
parser.add_argument("--v_min", default=0.4, type=float, help="Specify v_min")

# Configs for Potential and Forces
parser.add_argument("--V0", default=2, type=int, help="Specify V0 of PedPedPotential")
parser.add_argument("--sigma", default=1.303, type=float, help="Specify sigma of PedPedPotential")
parser.add_argument("--delta_t", default=0.4, type=float, help="Specify time for step size")
parser.add_argument("--twophi", default=200.0, type=float, help="Specify angle for visible range")
parser.add_argument("--c", default=0.5, type=float, help="Specify out of view factor")
parser.add_argument("--tau", default=0.5, type=float, help="Specify relaxation time")
parser.add_argument("--U0", default=2.1, type=float, help="Specify U0 of PedSpacePotential")
parser.add_argument("--r", default=1, type=float, help="Specify r of PedSpacePotential")
parser.add_argument("--beta", default=1, type=float, help="Specify factor for orthogonal force ratio (1 for none)")

parser.add_argument("--scene", default="hotel", type=str, help="Determine which scene to analyze")

# Get arguments
args = parser.parse_args()

if args.run_list:

    #sigma_list = [0.2171, 0.4343, 0.8686, 1.303, 1.7371, 2.171, 2.6058]
    #V0_list = [0,1,2,4,6]
    sigma_list = [0.2171, 0.4343, 0.8686, 1.303, 1.7371, 2.171, 2.6058]
    V0_list = [0,1,2,4,6]

    for V0 in V0_list:
        for sigma in sigma_list:
            args.V0 = V0
            args.sigma = sigma

            # val:
            args.phase = "val"
            args.nr_scenes = 1800
            # Create Scene
            PedPedScene = Scenarios.PedPedScene(args.V0, args.sigma, args.delta_t, args.twophi, args.c, args.v_max, args.v_min, args.dataset, args.phase, tau=args.tau, a=args.a, b=args.b, beta=args.beta)

            initial_agent_states = []
            states_of_Scene = PedPedScene.get_scene(args.nr_agents, args.nr_scenes, initial_agent_states, threshold=args.threshold)

            #test:
            args.phase = "test"
            args.nr_scenes = 1800
            # Create Scene
            PedPedScene = Scenarios.PedPedScene(args.V0, args.sigma, args.delta_t, args.twophi, args.c, args.v_max,
                                                args.v_min, args.dataset, args.phase, tau=args.tau, a=args.a, b=args.b, beta=args.beta)

            initial_agent_states = []
            states_of_Scene = PedPedScene.get_scene(args.nr_agents, args.nr_scenes, initial_agent_states,
                                                    threshold=args.threshold)

            # train
            args.phase = "train"
            args.nr_scenes = 9000
            # Create Scene
            PedPedScene = Scenarios.PedPedScene(args.V0, args.sigma, args.delta_t, args.twophi, args.c, args.v_max,
                                                args.v_min, args.dataset, args.phase, tau=args.tau, a=args.a, b=args.b, beta=args.beta)

            initial_agent_states = []
            states_of_Scene = PedPedScene.get_scene(args.nr_agents, args.nr_scenes, initial_agent_states,
                                                    threshold=args.threshold)

            print("datasets for V0: " + str(args.V0) + " and sigma: " + str(args.sigma) + " created.")

else:
    PedPedScene = Scenarios.PedPedScene(args.V0, args.sigma, args.delta_t, args.twophi, args.c, args.v_max, args.v_min,
                                        args.dataset, args.phase, tau=args.tau, a=args.a, b=args.b, beta=args.beta)

    initial_agent_states = []
    states_of_Scene = PedPedScene.get_scene(args.nr_agents, args.nr_scenes, initial_agent_states,
                                            threshold=args.threshold)

if args.show_animation:
    with PedPedScene.animate_scene(states_of_Scene) as _:
        pass


