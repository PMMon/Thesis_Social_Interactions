#!/bin/bash
#SBATCH --job-name="TrajectoryPrediction_SocialForce"
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --time=20:00:00
#SBATCH --mail-type=all
#SBATCH --output=slurm/slurm-%j.out

# Example bash-script to run Experiment with Vanilla LSTM Model on real dataset zara1
# eval "python  $@";

eval "python main/Experiment_Control.py --dataset_name zara1 --dataset_type real --socialforce false --model_type lstm --lstm_pool False --args_set lstm --phase test --num_epochs 250 --padding True --final_position False --show_traj True --visdom false --analyse_coll_avoidance false --nl_classified False --visualize_classified False";

