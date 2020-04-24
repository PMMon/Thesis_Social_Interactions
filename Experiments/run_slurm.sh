#!/bin/bash
#SBATCH --job-name="TrajectoryPrediction_SocialForce"
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --time=20:00:00
#SBATCH --mail-type=all
#SBATCH --output=slurm/slurm-%j.out

#eval "python  $@";

eval "python main/Experiment_Control.py --dataset_name eth --dataset_type real --model_type lstm --lstm_pool False --args_set lstm --num_epochs 500 --data_augmentation true --phase train --socialforce false --neighborhood_size 10 --grid_size 10 --plot_name lstm --analyse_real_dataset False --padding True --final_position False --show_traj False --histo false --nl_coarse False --visualize_classified False";

