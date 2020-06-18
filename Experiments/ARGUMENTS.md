# Experiments - Command-line Flags

You can conduct experiments on your trajectory prediction models by running the script: 

```
python main/Experiment_Control.py
```

This script has a number of command-line flags that you can use in order to configure your experiment:

## General Configurations
* `--socialforce`: Specifies whether to run experiment on synthetic data that is created by the Social Force Model (True) or on data of real-world human motion behavior (False). Default is `False`.
* `--phase`: Specifies whether to `train` (and validate) or `test` your model. Default is `train`. 
* `--show_traj`: Specifies whether to plot the predicted trajectories. Default is `False`.
* `--log_nth`: Defines after how many epochs to log training/validation/test losses. Default is 1.
* `--config`: Specifies config-file for data-preparation. Default is 1.
* `--log`: Specifies log-level. Choose either: `Info`, `Debug`, `Error`, `Critical` or leave unspecified. Default is `""`.

# Configurations for SocialForce-Experiment
* `--V0`: Defines value of `V0` of dataset. Default is 0.
* `--sigma`: Defines value of `sigma` of dataset. Default is 0.

# Configurations for Model
* `--model_name`: Specifies appendix for model name. Default is `""`. 
* `--model_type`: Defines type of model. Choose either: `linear`, `lstm` or `social-lstm`. Default is `lstm`.
* `--save_model`: Specifies whether or not to save trained model. Default is `True`.
* `--load_model`: Specifies whether or not to load existing model, if it exists. Default is `False`.
* `--lstm_pool`: Specifies whether to enable social pooling of model. Default is `False`.
* `--pooling_type`: Defines pooling scheme of model. Default is `social_pooling`.
* `--neighborhood_size`: Specifies neighborhood size of Social LSTM model to one side. Default is 10.
* `--grid_size`: Specifies size of grid of pooling module. Default is 10.
* `--args_set`: Specifies predefined set of configurations for respective model. Choose either: `lstm`, `social-lstm` or leave blank. Default is `""`.

# Configurations for Data-Preparation
* `--dataset_name`: Specifies name of dataset. Default is `to_be_defined`.
* `--dataset_type`: Specifies dataset-type. For real datasets choose: `real`. For synthetic datasets choose either `square` or `rectangle`. Default is `square`.
* `--obs_len`: Specifies length of observed trajectory. Default is 8.
* `--pred_len`: Specifies length of predicted trajectory. Default is 12.
* `--data_augmentation`: Determine whether or not you want to use data augmentation. Default is `False`.
* `--batch_norm`: Determines whether or not you want to use batch normalization. Default is `False`.
* `--max_num`: Specifies maximum number of processed pedestrian-ids. Default is 1000000.
* `--skip`: Specifies skipping rate for data-loader. Default is 20. 
* `--PhysAtt`: Specifies PhysicalAtt. Default is `""`. 
* `--padding`: Specifies whether or not padding should be active. Default is `True`.
* `--final_position`: Determines whether final positions of pedestrians should be passed to model or not. Default is `False`.

# Configurations for Training, Validation and Testing
* `--batch_size`: Specifies batch size. Default is 32.
* `--wd`: Specifies weight decay. Default is 0.0.
* `--lr`: Specifies learning rate. Default is 0.001.
* `--encoder_h_dim`: Specifies dimension of hidden state h of encoder. Default is 64.
* `--decoder_h_dim`: Specifies dimension of hidden state h of decoder. Default is 32.
* `--emb_dim`: Specifies dimension of embedding. Default is 32.
* `--num_epochs`: Specifies number of epochs. Default is 250.
* `--dropout`: Specifies dropout rate. Default is 0.0. 
* `--num_layers`: Specifies number of layers of LSTM/Social LSTM Model. Default is 1. 
* `--optim`: Specifies optimizer. Choose either: `adam`, `rmsprop` or `sgd`. Default is `adam`.

# Configurations for Additional Tools for Analysis
* `--approx_scheme`: Determines approximation scheme for calculation of curvature of trajectory. Default is `menger_curvature`. 
* `--nl_ADE`: Specifies whether or not to calculate nonlinear ADE of model in specific nonlinear regions. Default is `False`. 
* `--threshold_nl`: Defines threshold-value for nonlinear regions. Default is 0.5. 
* `--nl_classified`: Specifies whether or not to analyse and classify trajectories with respect to their degree of nonlinearity. Default is `False`. 
* `--visualize_classified`: Determines whether or not to plot examples of classified trajectories. Default is `False`.
* `--analyse_coll_avoidance`: Specifies whether or not to analyze collision avoidance behavior of models. Default is `False`.
* `--CDF_CollAvoid`: Specifies whether or not to create CDF-plot for collision avoidance behavior of models. Default is  `False`. 
* `--train_loss`: Defines loss on which model trains. Choose either: `ADE`, `nl_loss`, `MSE`, `ADE_nl` or `mixed`. Default is `ADE`. 
* `--plot_gradient_flow`: Specifies whether or not to plot values of gradients for monitoring possible vanishing gradients. Default is `False`.
* `--analyse_real_dataset`: Specifies whether to analyse real dataset with respect to the amount of suitable trajectories in the dataset. Default is `False`.

# Configurations for visdom-module
* `--visdom`: Specifies whether or not to plot via visdom. Default is `False`. 
* `--viz_port`: Specifies port for visdom. Default is 8090.
* `--viz_server`: Specifies server for visdom. Default is `""`. 