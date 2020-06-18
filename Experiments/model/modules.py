import sys, os
sys.path.append(os.path.abspath("."))
import torch
import torch.nn as nn
import argparse

from helper.bool_flag import bool_flag
from model.BaseModel import BaseModel
from model.model_modules import make_mlp, Decoder, Encoder

# ===================== Description =======================
# Implementation of different trajectory prediction models
# =========================================================

class LINEAR(BaseModel):
    """
    Linear model: uses a fully connected layer to predict the next relative position/velocity of each pedestrian.
    The predicted position is then linearly projected towards future time steps. Hence, the model predicts exclusively linear
    future trajectories.
    """
    def __init__(self, args,
                 float_type = torch.float64,
                 device="gpu",
                 input_dim = 2,
                 mlp_dim = 32,
                 **kwargs):
        super().__init__(device, float_type)

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.dropout = args.dropout
        self.mlp_dim = mlp_dim
        self.input_dim = input_dim

        # Specify losses for model: Average Displacement Error (ADE), Final Displacement Error (FDE) and the average of these two losses (AV)
        self.losses.extend(["G_L2", "G_ADE", "G_FDE", "G_AV"])

        self.fc = nn.Sequential(
                nn.Linear((self.obs_len-1) * self.input_dim, self.mlp_dim),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.mlp_dim, self.mlp_dim ),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.mlp_dim, self.input_dim))

    def forward(self, inputs):
        """
        Implements forward run of model. Note that the model predicts the relative position of the time step obs_len + 1.
        It then maps this relative distance on the remaining time steps obs_len + 2, ..., obs_len + pred_len
        :param inputs: Dictionary that holds information about the observed trajectories
        :return: Dictionary with information about the predicted trajectories in absolute coordinates
        """
        input_abs = inputs["in_xy"]
        input_rel = inputs["in_dxdy"]
        batch_size = input_abs.size(1)

        # Predict relative position of next time step (t = obs_len + 1)
        v = self.fc(input_rel.permute(1, 0, 2).reshape(batch_size, -1)).unsqueeze(0)

        # Map predicted distance to remaining time steps
        time = torch.arange(1, self.pred_len+1).to(v).view(-1, 1, 1).repeat(1, batch_size, 1)
        out_xy = time * v + input_abs[-1].unsqueeze(0)

        return {"out_xy": out_xy}


class LSTM(BaseModel):
    """
    This class implements the Vanilla LSTM model (if flag --lstm_pool is set to 'False' AND --model_type is set to 'lstm') and
    the Social LSTM model (if flag --lstm_pool is set to 'True' OR --model_type is set to 'social-lstm').

    A conceptional overview of the Vanilla LSTM model can be found in chapter 3.3.2 of the bachelor's thesis:
    P. Mondorf: 'Modeling Social Interactions for Pedestrian Trajectory Prediction on Real and Synthetic Datasets'.
    Technical University of Munich, 2020.

    A conceptional overview of the Social LSTM model can be found in this paper:
    A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei, and S. Savarese. 'Social LSTM: Human Trajectory Prediction in Crowded Space'.
    In: CVPR (2016), pp. 961–971. Link: http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf
    """
    def __init__(self, args,
                 float_type=torch.float64,
                 device = "gpu",
                 input_dim=2,
                 mlp_dim = 64,
                 pool_dim = None,
                 **kwargs):
        super().__init__(device, float_type)

        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.pool_dim = pool_dim
        self.embedding_dim = args.emb_dim
        self.decoder_h_dim = args.decoder_h_dim
        self.encoder_h_dim = args.encoder_h_dim
        self.dropout = args.dropout
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.batch_norm = args.batch_norm
        self.pool_every_timestep = args.lstm_pool
        self.pooling = args.pooling_type
        self.neighborhood_size = args.neighborhood_size
        self.grid_size = args.grid_size
        self.final_position = args.final_position

        # Specify losses for model: Average Displacement Error (ADE), Final Displacement Error (FDE) and the average of these two losses (AV)
        self.losses.extend(["G_L2",  "G_ADE", "G_FDE", "G_AV"])
        if args.nl_ADE:
            self.losses.extend(["G_ADE_nl_regions"])

        self.encoder = Encoder(
            encoder_h_dim=self.encoder_h_dim,
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
            num_layers= self.num_layers
        )

        self.decoder = Decoder(seq_len=self.pred_len,
                               input_dim=self.input_dim,
                               decoder_h_dim=self.decoder_h_dim,
                               embedding_dim=self.embedding_dim,
                               dropout=self.dropout,
                               mlp_dim=self.mlp_dim,
                               batch_norm=self.batch_norm,
                               num_layers= self.num_layers,
                               pool_every_timestep=self.pool_every_timestep,
                               pooling=self.pooling,
                               neighborhood_size=self.neighborhood_size,
                               grid_size=self.grid_size,
                               final_position=self.final_position
                               )

        if self.mlp_decoder_needed():
            # Use MLP to match the dimensions between encoder and decoder
            self.encoder2decoder = make_mlp(
                [self.encoder_h_dim, self.mlp_dim, self.decoder_h_dim],
                    activation='relu',
                    dropout = self.dropout)


    def mlp_decoder_needed(self):
        if (self.encoder_h_dim != self.decoder_h_dim):
            return True
        else:
            return False


    def init_c(self, batch_size):
        """
        Initialize empty cell state for the decoder
        :param batch_size: Size of batch
        :return: Cell state
        """
        return torch.zeros((1, batch_size, self.decoder_h_dim))


    def forward(self, inputs):
        """
        Implements forward run of model. Note that the model is based on a lstm-based encoder-decoder archictecture, where the
        input is processed by an encoder and the decoder predicts the future trajectory of each pedestrian.
        :param inputs: Dictionary that holds information about the observed trajectories
        :return: Dictionary with information about the predicted trajectories
        """
        batch_size = inputs["in_xy"].size(1)

        _,h = self.encoder(inputs["in_dxdy"])

        if self.mlp_decoder_needed():
            h = self.encoder2decoder(h)

        c = self.init_c(batch_size).to(inputs["in_xy"])

        # Last observed position
        x0=inputs["in_xy"][-1]
        v0=inputs["in_dxdy"][-1]
        state_tuple = (h, c)

        # Provide information about final position
        if self.final_position:
            dest = inputs["gt"][-1]
        else:
            dest = 0

        # If data is loaded with information about all pedestrians in a scene, pass this information to the decoder
        if "seq_start_end" in inputs.keys():
            seq_start_end = inputs["seq_start_end"]
        else:
            seq_start_end = []

        # If missing data is padded, pass information about which trajectories are padded to the decoder
        if "pred_check" in inputs.keys():
            pred_check = inputs["pred_check"]
        else:
            pred_check = torch.ones(batch_size,1)

        # Predict future trajectories
        out = self.decoder(last_pos=x0,
                last_pos_rel=v0,
                state_tuple=state_tuple,
                seq_start_end=seq_start_end,
                pred_check=pred_check,
                dest = dest)

        return out





if __name__ == "__main__":

    parser = argparse.ArgumentParser("Trajectory Prediction Basics")

    # Configs for Model
    parser.add_argument("--model_name", default="", type=str, help="Define model name for saving")
    parser.add_argument("--model_type", default="lstm", type=str,help="Define type of model. Choose either: linear, lstm or social-lstm")
    parser.add_argument("--save_model", default=True, type=bool_flag, help="Save trained model")
    parser.add_argument("--load_model", default=False, type=bool_flag, help="Specify whether to load existing model")
    parser.add_argument("--lstm_pool", default=False, type=bool_flag, help="Specify whether to enable social pooling")
    parser.add_argument("--pooling_type", default="social_pooling", type=str, help="Specify pooling method")
    parser.add_argument("--neighborhood_size", default=10.0, type=float, help="Specify neighborhood size to one side")
    parser.add_argument("--grid_size", default=10, type=int, help="Specify grid size")
    parser.add_argument("--args_set", default="", type=str, help="Specify predefined set of configurations for respective model. Choose either: lstm or social-lstm")

    # Configs for data-preparation
    parser.add_argument("--dataset_name", default="to_be_defined", type=str, help="Specify dataset")
    parser.add_argument("--dataset_type", default="square", type=str, help="Specify dataset-type. For real datasets choose: 'real'. For synthetic datasets choose either 'square' or 'rectangle'")
    parser.add_argument("--obs_len", default=8, type=int, help="Specify length of observed trajectory")
    parser.add_argument("--pred_len", default=12, type=int, help="Specify length of predicted trajectory")
    parser.add_argument("--data_augmentation", default=True, type=bool_flag, help="Specify whether or not you want to use data augmentation")
    parser.add_argument("--batch_norm", default=False, type=bool_flag, help="Batch Normalization")
    parser.add_argument("--max_num", default=1000000, type=int, help="Specify maximum number of ids")
    parser.add_argument("--skip", default=20, type=int, help="Specify skipping rate")
    parser.add_argument("--PhysAtt", default="", type=str, help="Specify physicalAtt")
    parser.add_argument("--padding", default=True, type=bool_flag, help="Specify if padding should be active")
    parser.add_argument("--final_position", default=False, type=bool_flag, help="Specify whether final positions of pedestrians should be passed to model or not")

    # Configs for training, validation, testing
    parser.add_argument("--batch_size", default=32, type=int, help="Specify batch size")
    parser.add_argument("--wd", default=0.03, type=float, help="Specify weight decay")
    parser.add_argument("--lr", default=0.001, type=float, help="Specify learning rate")
    parser.add_argument("--encoder_h_dim", default=64, type=int, help="Specify hidden state dimension h of encoder")
    parser.add_argument("--decoder_h_dim", default=32, type=int, help="Specify hidden state dimension h of decoder")
    parser.add_argument("--emb_dim", default=32, type=int, help="Specify dimension of embedding")
    parser.add_argument("--num_epochs", default=250, type=int, help="Specify number of epochs")
    parser.add_argument("--dropout", default=0.0, type=float, help="Specify dropout rate")
    parser.add_argument("--num_layers", default=1, type=int, help="Specify number of layers of LSTM/Social LSTM Model")
    parser.add_argument("--optim", default="Adam", type=str, help="Specify optimizer. Choose either: adam, rmsprop or sgd")

    # Get arguments
    args = parser.parse_args()

    print("Linear" )
    print(LINEAR(args))

    print("LSTM")
    print(LSTM(args))

