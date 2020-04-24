"""
This script implements the bicycle GAN
"""
import sys, os
sys.path.append(os.path.abspath("."))
import torch
import torch.nn as nn
import argparse

from helper.bool_flag import bool_flag
from model.BaseModel import BaseModel
from model.model_modules import make_mlp, get_noise, Decoder, Encoder

class LINEAR(BaseModel):
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

        self.losses.extend(["G_L2", "G_ADE", "G_FDE", "G_AV"])

        self.fc = nn.Sequential(
                nn.Linear((self.obs_len-1) * self.input_dim, self.mlp_dim),    # initially (self.obs_len-1) * self.input_dim. -> used for relative coordinations (dxdy), otherwise use (self.obs_len)*self.input_dim
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.mlp_dim, self.mlp_dim ),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.mlp_dim, self.input_dim))

    def forward(self, inputs):
        input_abs = inputs["in_xy"]
        input_rel = inputs["in_dxdy"]
        batch_size = input_abs.size(1)
        v = self.fc(input_rel.permute(1, 0, 2).reshape(batch_size, -1)).unsqueeze(0)
        time = torch.arange(1, self.pred_len+1).to(v).view(-1, 1, 1).repeat(1, batch_size, 1)
        out_xy = time * v + input_abs[-1].unsqueeze(0)
        return {"out_xy": out_xy}

class LSTM(BaseModel):
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


        self.losses.extend(["G_L2",  "G_ADE", "G_FDE", "G_AV"])


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
            self.encoder2decoder = make_mlp(
                [self.encoder_h_dim, self.mlp_dim, self.decoder_h_dim],
                    activation='relu',
                    dropout = self.dropout)

    def init_c(self, batch_size):
        return torch.zeros((1, batch_size, self.decoder_h_dim))

    def mlp_decoder_needed(self):
        if (self.encoder_h_dim != self.decoder_h_dim):
            return True
        else:
            return False

    def forward(self, inputs):


        batch_size = inputs["in_xy"].size(1)

        _,h = self.encoder(inputs["in_dxdy"])

        if self.mlp_decoder_needed():
            h = self.encoder2decoder(h)

        c = self.init_c(batch_size).to(inputs["in_xy"])


        #last position
        x0=inputs["in_xy"][-1]
        v0=inputs["in_dxdy"][-1]
        state_tuple = (h, c)

        #provide information about final position
        if self.final_position:
            dest = inputs["gt"][-1]
        else:
            dest = 0    #torch.zeros(x0.shape)

        if "seq_start_end" in inputs.keys():
            seq_start_end = inputs["seq_start_end"]
        else:
            seq_start_end = []

        if "pred_check" in inputs.keys():
            pred_check = inputs["pred_check"]
        else:
            pred_check = torch.ones(batch_size,1)

        out = self.decoder(last_pos=x0,
                last_pos_rel=v0,
                state_tuple=state_tuple,
                seq_start_end=seq_start_end,
                pred_check=pred_check,
                dest = dest)

        return out





if __name__ == "__main__":

    parser = argparse.ArgumentParser("Trajectory Prediction Basics")
    # Config about Model
    parser.add_argument("--model_name", default="", type=str, help="Define model name for saving")
    parser.add_argument("--model_type", default="lstm", type=str, help="Define model type. Either Linear or LSTM Model")
    parser.add_argument("--save_model", default=True, type=bool_flag, help="Save trained model")
    parser.add_argument("--load_model", default=False, type=bool_flag, help="specify whether to load existing model")
    parser.add_argument("--pool_every_timestep", default=False, type=bool_flag,help="specify whether to enable social pooling")
    parser.add_argument("--pooling", default="social_pooling", type=str, help="specify pooling method")

    # Config about data-preparation
    parser.add_argument("--dataset_name", default="hotel", type=str, help="Specify dataset")
    parser.add_argument("--obs_len", default=8, type=int, help="Specify observed length")
    parser.add_argument("--pred_len", default=12, type=int, help="specify predicted length")
    parser.add_argument("--data_augmentation", default=True, type=bool_flag,
                        help="Specify whether or not you want to use data augmentation")
    parser.add_argument("--batch_norm", default=False, type=bool_flag, help="Batch Normalization")
    parser.add_argument("--max_num", default=10000, type=int, help="specify maximum number of ids")
    parser.add_argument("--skip", default=20, type=int, help="specify skipping rate")
    parser.add_argument("--PhysAtt", default="", type=str, help="specify physicalAtt")

    # Config about training, val, testing
    parser.add_argument("--batch_size", default=32, type=int, help="specify batch size")
    parser.add_argument("--wd", default=0.03, type=float, help="specify weight decay")
    parser.add_argument("--lr", default=0.001, type=float, help="specify learning rate")
    parser.add_argument("--decoder_h_dim", default=16, type=int, help="specify decoder_h_dim")
    parser.add_argument("--encoder_h_dim", default=16, type=int, help="specify encoder_h_dim")
    parser.add_argument("--emb_dim", default=8, type=int, help="specify embedding dimension of encoder")
    parser.add_argument("--num_epochs", default=30, type=int, help="specify number of epochs")
    parser.add_argument("--dropout", default=0.0, type=float, help="specify dropout rate")
    parser.add_argument("--num_layers", default=1, type=int, help="specify number of layers for LSTM")
    parser.add_argument("--optim", default="Adam", type=str, help="specify optimizer")

    # Get arguments
    args = parser.parse_args()

    print("Linear" )
    print(LINEAR(args))

    print("LSTM")
    print(LSTM(args))

