#!/bin/bash
#SBATCH --job-name="TrajectoryPrediction_SocialForce"
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=10GB
#SBATCH --time=20:00:00
#SBATCH --mail-type=all
#SBATCH --output=slurm/slurm-%j.out

# Example bash script to run experiment with models on synthetic datasets.
# Values of V0 used in thesis: 0 1 2 4 6
# Values of sigma used in thesis: 0.2171 0.4343 0.8686 1.303 1.7371 2.171 2.6058
# for S-LSTM: "python main/Experiment_Control.py --model_type social-lstm --lstm_pool true --args_set social-lstm --num_epochs 200 --data_augmentation true --phase train --socialforce true --neighborhood_size 10 --V0 $i --sigma $j --padding True";
# for LSTM: "python main/Experiment_Control.py --model_type lstm --lstm_pool false --args_set lstm --num_epochs 200 --data_augmentation true --phase train --socialforce true --V0 $i --sigma $j --padding True";

for i in 0 1 2 4 6
do
  for j in 0.2171 0.4343 0.8686 1.303 1.7371 2.171 2.6058
  do
    eval "python main/Experiment_Control.py --model_type social-lstm --lstm_pool True --args_set social-lstm --phase train --num_epochs 250 --socialforce true --V0 $i --sigma $j --data_augmentation true --neighborhood_size 10 --grid_size 10 --padding True --final_position False --show_traj true --nl_ADE False --analyse_coll_avoidance False --CDF_CollAvoid False --nl_classified False --visualize_classified False";
  done
done
