import torch 
import torch.nn as nn
import numpy as np

# Creates a MLP with configurable actvation functions and dimensions for each layer
def make_mlp(dim_list, activation, batch_norm=False, dropout=0.0):
    """
    Inputs
    :param dim_list: list of dimensions for each layer in the MLP
    :param activation_list: list of activation function for each layer
    :param batch_norm: choose whether batch_norm should be applied
    :param dropout: choose wheter dropout should be applied
    :return: MLP
    """
    layers = []

    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        noise = torch.cuda.FloatTensor(*shape).normal_(0, 1)
        torch.cuda.synchronize()
        return noise 
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """Encoder is part of TrajectoryGenerator and TrajectoryDiscriminator
         ...
         Input:
         -----------
         - input_dim:        input dimensionality of spatial coordinates (int)
         - h_dim:    dimensionality of hidden state (int)
         - embedding_dim:    dimensionality spatial embedding
         - dropout:          dropout in LSTM layer

         Methods:
         ----------
         - init_hidden( ): initialize hidden state
         - forward()
         """

    def __init__(self, encoder_h_dim=64, input_dim=2, embedding_dim=16,dropout=0.0, num_layers=1):
        super(Encoder, self).__init__()

        self.encoder_h_dim = encoder_h_dim
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

        if embedding_dim:
            self.spatial_embedding = nn.Linear(input_dim, embedding_dim)
            self.encoder = nn.LSTM(embedding_dim, encoder_h_dim, dropout=dropout, num_layers=num_layers)
        else:
            self.encoder = nn.LSTM(input_dim, encoder_h_dim, dropout=dropout, num_layers=num_layers)

    def init_hidden(self, batch, obs_traj):

        return (
            torch.zeros(1, batch, self.encoder_h_dim).to(obs_traj),
            torch.zeros(1, batch, self.encoder_h_dim).to(obs_traj)
        )

    def forward(self, obs_traj, state_tuple=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        if not state_tuple:
            state_tuple = self.init_hidden(batch, obs_traj)
        if self.embedding_dim:
            obs_traj = self.spatial_embedding(obs_traj)
            # obs_traj = obs_traj_embedding.view(
            # -1, batch, self.embedding_dim)

        output, state = self.encoder(obs_traj, state_tuple)
        final_h = state[0]
        return output, final_h


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator

    """
    def __init__(self, seq_len = 12, input_dim = 2, decoder_h_dim=128, embedding_dim=64, dropout=0.0, mlp_dim = 128,
            batch_norm = False, num_layers=1, pool_every_timestep=True, pooling="social_pooling",
            neighborhood_size=2.0, grid_size=8, pool_dim=None, activation_function="relu", final_position = False):
        super(Decoder, self).__init__()

        self.__dict__.update(locals())
        del self.self

        self.spatial_embedding = nn.Sequential(nn.Linear( self.input_dim, self.embedding_dim), nn.Dropout(p=dropout))

        self.hidden2pos = nn.Linear( self.decoder_h_dim , self.input_dim )

        self.decoder = nn.LSTM( self.embedding_dim, self.decoder_h_dim , dropout = dropout, num_layers=num_layers)

        if self.pool_every_timestep:
            if self.pooling == "social_pooling":
                self.pool_net = SocialPooling(h_dim=self.decoder_h_dim,
                                              activation_function=self.activation_function,
                                              batch_norm=self.batch_norm,
                                              dropout=self.dropout,
                                              neighborhood_size=self.neighborhood_size,
                                              grid_size=self.grid_size,
                                              pool_dim=self.pool_dim)

            self.mlp_dims_pool = [self.decoder_h_dim *2, self.mlp_dim, self.decoder_h_dim]
            self.mlp = make_mlp(
                self.mlp_dims_pool,
                activation=self.activation_function,
                batch_norm=batch_norm,
                dropout=dropout
            )

        #if self.final_position:
        self.mlp_dims_final_position = [self.embedding_dim * 2, self.mlp_dim, self.embedding_dim]
        self.mlp_final_position = make_mlp(
            self.mlp_dims_final_position,
            activation=self.activation_function,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def forward(self, last_pos, last_pos_rel, state_tuple, pred_check, dest, scene_img = None, seq_start_end = np.array([[0,3]])):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        Output:
        - pred_traj_fake_rel: tensor of shape (self.seq_len, batch, 2)
        - pred_traj_fake: tensor of shape (self.seq_len, batch, 2)
        - state_tuple[0]: final hidden state
        """

        batch_size = last_pos_rel.size(0)
        pred_traj_fake_rel = []
        pred_traj_fake = []
        softmax_list = []
        final_pos_list = []
        img_patch_list = []
        final_pos_map_decoder_list = []

        for t in range(self.seq_len):

            # for each element in seq_len (for each coordinate-pair) increase dimension by spatial embedding dimension (and Linearity)
            decoder_input = self.spatial_embedding(last_pos_rel)
            decoder_input = decoder_input.view(1, batch_size, self.embedding_dim)

            if self.final_position:
                destination_input = self.spatial_embedding(dest-last_pos)
                destination_input = destination_input.view(-1, self.embedding_dim)
                decoder_input = torch.cat([decoder_input.view(-1, self.embedding_dim), destination_input], dim=1)
                decoder_input = self.mlp_final_position(decoder_input).unsqueeze(dim=0)


            # Decode spatial embedded coordinate pair with last state_tuple
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            # Convert last hidden State (only have one, assuming output == state_tuple -> Need Check!) back t dimensionality of coordinate-pair
            last_pos_rel = self.hidden2pos(output.view(-1, self.decoder_h_dim))
            # Calculate current position
            curr_pos = last_pos_rel + last_pos

            # If decided to use Social-Pooling
            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat([decoder_h.view(-1, self.decoder_h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h).unsqueeze(dim=0)
                state_tuple = (decoder_h, state_tuple[1])

            pred_traj_fake_rel.append(last_pos_rel.clone().view(batch_size, -1))
            pred_traj_fake.append( curr_pos.clone().view(batch_size, -1))
            last_pos = curr_pos

        # All predictions including padded trajectories
        pred_traj_fake_rel_all = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake_all = torch.stack(pred_traj_fake, dim=0)

        #Eliminate output for padded trajectories
        if pred_check.shape[0] <= 1:
            pred_traj_fake = pred_traj_fake_all[:, 0:1, :]
            pred_traj_fake_rel = pred_traj_fake_rel_all[:, 0:1, :]
        else:
            pred_traj_fake = pred_traj_fake_all[:, pred_check.squeeze() > 0,:]
            pred_traj_fake_rel = pred_traj_fake_rel_all[:, pred_check.squeeze() > 0,:]

        output = {"out_xy":  pred_traj_fake,
                 "out_dxdy": pred_traj_fake_rel,
                    "h" : state_tuple[0],
                  "out_xy_all": pred_traj_fake_all,
                  "out_dxdy_all": pred_traj_fake_rel_all
                  }

        return output

class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    def __init__(self, h_dim=64, activation_function='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation_function,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *self.grid_size)
        # Added this part to implementation, otherwise the pooling is going to run into an indexing error
        cell_x[cell_x == self.grid_size] -= 1
        cell_y = torch.floor(((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *self.grid_size)
        cell_y[cell_y == self.grid_size] -= 1
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)


            grid_pos = self.get_grid_locations(top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude

            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            # Get inidicees of curr_end_pos for which it is out of neighborhood
            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange( 0, total_grid_size * num_ped, total_grid_size).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            if grid_pos.max() > curr_pool_h.shape[0]:
                print("ERROR EXCEED!")
                print(grid_pos.max())
                print(curr_pool_h.shape)
            if grid_pos.min() < 0:
                print("ERROR UNDER!")
                print(grid_pos.min())
                print(curr_pool_h.shape)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)

        return pool_h

# Todo: Think about rearrangements, is it ensured to assign the correct hidden states to the correct ped?

if __name__ == "__main__":

    print("Test Encoder")
    print(Encoder())

    print("Test Decoder")
    print(Decoder())

    print("Test SocialPooling")
    print(SocialPooling())


