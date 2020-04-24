#!/bin/bash
#SBATCH --job-name="TrajectoryPrediction_SocialForce"
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=10GB
#SBATCH --time=20:00:00
#SBATCH --mail-type=all
#SBATCH --output=slurm/slurm-%j.out

# 0 1 2 4 6
# 0.2171 0.4343 0.8686 1.303 1.7371 2.171 2.6058
# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
# for S-LSTM: "python main/main.py --model_type social-lstm --lstm_pool true --args_set social-lstm --num_epochs 200 --data_augmentation true --phase train --socialforce true --neighborhood_size 10 --plot_name social-lstm --V0 $i --sigma $j";
# for LSTM: "python main/main.py --model_type lstm --lstm_pool false --args_set lstm --num_epochs 200 --data_augmentation true --phase train --socialforce true --plot_name lstm --V0 $i --sigma $j";

for i in 6
do
  for j in 2.6058
  do
    eval "python main/Experiment_Control.py --model_type social-lstm --lstm_pool True --args_set social-lstm --num_epochs 250 --data_augmentation true --phase test --socialforce true --neighborhood_size 10 --grid_size 10 --plot_name social-lstm --padding True --final_position True --show_traj true --histo false --nl_coarse False --visualize_classified False --V0 $i --sigma $j";
  done
done
