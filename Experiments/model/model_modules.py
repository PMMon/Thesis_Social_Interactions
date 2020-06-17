import torch 
import torch.nn as nn
import numpy as np

# =========================== Description =============================
# Important modules of the trajectory prediction models, such as the encoder
# and decoder of the Vanilla/Social LSTM Model, the pooling layer of
# the Social LSTM Model, etc.
# =====================================================================

def make_mlp(dim_list, activation, batch_norm=False, dropout=0.0):
    """
    Creates a MLP with configurable activation functions and dimensions for each layer
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


class Encoder(nn.Module):
    """
    Encoder processes the observed trajectories and stores the respective key feature in a latent variable that serves
    as input for the decoder of the model

         Input:
         -----------
         - input_dim: input dimensionality of spatial coordinates (int)
         - encoder_h_dim: dimensionality of hidden state (int)
         - embedding_dim: dimensionality spatial embedding
         - dropout: dropout in LSTM layer
         -----------
         """
    def __init__(self, encoder_h_dim=64, input_dim=2, embedding_dim=16, dropout=0.0, num_layers=1):
        super(Encoder, self).__init__()

        self.encoder_h_dim = encoder_h_dim
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

        if embedding_dim:
            self.spatial_embedding = nn.Linear(input_dim, embedding_dim)
            self.encoder = nn.LSTM(embedding_dim, encoder_h_dim, dropout=dropout, num_layers=num_layers)
        else:
            self.encoder = nn.LSTM(input_dim, encoder_h_dim, dropout=dropout, num_layers=num_layers)


    def init_hidden(self, batch_size, obs_traj):
        """
        Initialize hidden state
        :param batch_size: Size of batch (int)
        :param obs_traj: Observed trajectories of pedestrians
        :return: State tuple
        """
        return (
            torch.zeros(1, batch_size, self.encoder_h_dim).to(obs_traj),
            torch.zeros(1, batch_size, self.encoder_h_dim).to(obs_traj)
        )

    def forward(self, obs_traj, state_tuple=None):
        """
        Implements forward-run of encoder module
        :param obs_traj: Tensor comprising observed trajectories of shape (obs_len, batch_size, 2)
        :param state_tuple: Possible initial state tuple
        :return output: Tensor comprising the output of each LSTM cell. Has shape (obs_len-1, batch_size, self.encoder_h_dim)
        :return final_h: Final hidden state/compressed latent information of shaoe (self.nr_layer, batch_size, self.encoder_h_dim)
        """
        # Encode observed Trajectory
        batch_size = obs_traj.size(1)
        if not state_tuple:
            state_tuple = self.init_hidden(batch_size, obs_traj)
        if self.embedding_dim:
            obs_traj = self.spatial_embedding(obs_traj)

        output, state = self.encoder(obs_traj, state_tuple)
        final_h = state[0]

        return output, final_h


class Decoder(nn.Module):
    """
    Decoder uses the final hidden state of the encoder to predict the next positions of each pedestrian.
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
            # Defines social pooling for the Social LSTM model. Other pooling techniques are possible.
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

        # It is possible to pass the destination of each pedestrian to the network, such that it "knows" where the pedestrians navigate to.
        # Allows for the focus on the ability to predict social interactions between pedestrians.
        if self.final_position:
            self.mlp_dims_final_position = [self.embedding_dim * 2, self.mlp_dim, self.embedding_dim]
            self.mlp_final_position = make_mlp(
                self.mlp_dims_final_position,
                activation=self.activation_function,
                batch_norm=batch_norm,
                dropout=dropout
            )


    def forward(self, last_pos, last_pos_rel, state_tuple, pred_check, dest, scene_img = None, seq_start_end = np.array([[0,3]])):
        """
        Implements forward-run of decoder module
        :param last_pos: Tensor comprising the last positions of observed trajectories. Tensor has shape (batch_size, 2)
        :param last_pos_rel: Tensor comprising the last relative positions of observed trajectories. Tensor has shape (batch_size, 2)
        :param state_tuple: (h, c) - h: last hidden state of encoder, c: initialized cell state. Both tensors have shape (self.num_layers, batch_size, self.decoder_h_dim)
        :param pred_check: Tensor that indicates which trajectory was padded (= 0). Has shape (nr_pedestrians_in_sequence_n, 1)
        :param dest: Destinations of each pedestrian. Tensor of shape (batch_size, 1).
        :param scene_img: Image of scene
        :param seq_start_end: Tensor of shape (nr_sequences,2). The n-th entry contains a tensor of size 1x2 that specifies the ids of the pedestrians in
        the n-th sequence
        :return: Dictionary with predicted future trajectories.
        """
        batch_size = last_pos_rel.size(0)
        pred_traj_fake_rel = []
        pred_traj_fake = []

        for t in range(self.seq_len):

            # For each element in seq_len (for each coordinate-pair) increase dimension by spatial embedding dimension
            decoder_input = self.spatial_embedding(last_pos_rel)
            decoder_input = decoder_input.view(1, batch_size, self.embedding_dim)

            # If True, pass information about the destination of each pedestrians to decoder input
            if self.final_position:
                destination_input = self.spatial_embedding(dest-last_pos)
                destination_input = destination_input.view(-1, self.embedding_dim)
                decoder_input = torch.cat([decoder_input.view(-1, self.embedding_dim), destination_input], dim=1)
                decoder_input = self.mlp_final_position(decoder_input).unsqueeze(dim=0)


            # Decode spatial embedded coordinate pair with last state_tuple
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            # Convert last hidden State back to dimensionality of coordinate-pair
            last_pos_rel = self.hidden2pos(output.view(-1, self.decoder_h_dim))
            # Calculate current position
            curr_pos = last_pos_rel + last_pos

            # If decided to use pooling method, e.g. Social-Pooling
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

        # Eliminate output for padded trajectories since we do not have a suitable ground truth
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
    """
    This class implements the social pooling layer, introduced by:
    A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei, and S. Savarese. 'Social LSTM: Human Trajectory Prediction in Crowded Space'.
    In: CVPR (2016), pp. 961–971. Link: http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf
    """
    def __init__(self, h_dim=64, activation_function='relu', batch_norm=True, dropout=0.0, neighborhood_size=2.0, grid_size=8, pool_dim=None):
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
        """
        The pooling layer partially preserves spatial information using a grid-based pooling mechanism.
        :param ped_pos: Tensor describing the position of the pedestrians in a scene. Has shape (nr_peds_in_sequence, 2)
        :return: Two tensors with coordinates of the top left and bottom right corner of the grid. Both tensors have shape (nr_peds_in_sequence, 2)
        """
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2

        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)

        return top_left, bottom_right


    def get_grid_locations(self, top_left, other_pos):
        """
        Assign locations in grid to surrounding neighbors in order to pool spatially close pedestrians
        :param top_left: Tensor with nr_peds_in_seq times top_left-output of function self.get_bound() -> Top left grid corner for each pedestrian in the sequence
        :param other_pos: Tensor describing nr_peds_in_seq times the position of the pedestrians in a scene. Has shape (nr_peds_in_sequence*nr_peds_in_sequence, 2)
        :return: Tensor with locations in grid. Has shape (nr_peds_in_sequence*nr_peds_in_sequence)
        """
        cell_x = torch.floor(((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *self.grid_size)

        # Added this part to implementation, otherwise the pooling is going to run into an indexing error
        cell_x[cell_x == self.grid_size] -= 1
        cell_y = torch.floor(((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *self.grid_size)
        cell_y[cell_y == self.grid_size] -= 1
        grid_pos = cell_x + cell_y * self.grid_size

        return grid_pos


    def repeat(self, tensor, num_reps):
        """
        Repeats tensors
        :param tensor: 2D tensor of any shape
        :param num_reps: Number of times to repeat each row
        :return: Tensor that repeats each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)

        return tensor


    def forward(self, h_states, seq_start_end, end_pos):
        """
        Actual implementation of the pooling scheme
        :param h_states: Tensor of shape (num_layers, batch_size, h_dim) with hidden states
        :param seq_start_end: A list of tuples which delimit sequences within batch.
        :param end_pos: Absolute end position of obs_traj (batch_size, 2)
        :return: Tensor of shape (batch_size, h_dim)
        """
        pool_h = []

        for _, (start, end) in enumerate(seq_start_end):
            # Get ids of first and last pedestrian in sequence
            start = start.item()
            end = end.item()
            num_ped = end - start

            grid_size = self.grid_size * self.grid_size

            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))

            # Get boundaries of grid
            top_left, bottom_right = self.get_bounds(curr_end_pos)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            # Get grid positions of neighbors for each pedestrian
            grid_pos = self.get_grid_locations(top_left, curr_end_pos).type_as(seq_start_end)

            # == Mark all positions to exclude as non-zero ==

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

            # Perform grid-based pooling with scatter_add-function
            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)

        return pool_h


if __name__ == "__main__":

    print("Test Encoder")
    print(Encoder())

    print("Test Decoder")
    print(Decoder())

    print("Test SocialPooling")
    print(SocialPooling())


