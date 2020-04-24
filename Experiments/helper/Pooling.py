import torch
import torch.nn as nn

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

class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
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
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *self.grid_size)
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

            # todo: modify
            grid_size = self.grid_size * self.grid_size

            # Pick h_states of sequence
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]

            # Repeat whole tensor in dim 1: num_ped times. e.g. 8x4 -> 64x4
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)

            # Get current position and repeat later
            curr_end_pos = end_pos[start:end]

            # Prepare final vector
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))

            # Get bounds -> 8x2 tensors
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            # find grid pos of each ped w.r.t. top left of resp. ped => 1 x num_ped*num_ped tensor
            # ToDo: Modify for debugging
            grid_pos = self.get_grid_locations(top_left, bottom_right).type_as(seq_start_end)
            #grid_pos = self.get_grid_locations(top_left, curr_end_pos).type_as(seq_start_end)
            print("grid_pos: " + str(grid_pos))
            print("dim: " + str(grid_pos.shape))


            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            # TODO: overthink if this won't exclude too many peds -> currently also excludes peds that are wihtin the grid but leave it at the end
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
            # Todo: modify
            offset = torch.arange( 0, total_grid_size * num_ped, total_grid_size).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset

            # grid_pos outside bound = 0
            grid_pos[within_bound != 0] = 0

            # from tensor num_peds*num_peds -> num_peds*num_peds x hidden_dim
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            if grid_pos.shape[0] > curr_hidden_repeat.shape[0] or grid_pos.shape[1] > curr_hidden_repeat.shape[1]: #or grid_pos.shape[2] > curr_hidden_repeat.shape[2]:
                print("First Shape Error!")
                print("dim 0: ")
                print("grid_pos dim0: " + str(grid_pos.shape[0]))
                print("curr_hidden_repeat dim0: " + str(curr_hidden_repeat.shape[0]))
                print("dim 1: ")
                print("grid_pos dim1: " + str(grid_pos.shape[1]))
                print("curr_hidden_repeat dim1: " + str(curr_hidden_repeat.shape[1]))
                #print("dim 2: ")
                #print("grid_pos dim2: " + str(grid_pos.shape[2]))
                #print("curr_hidden_repeat dim2: " + str(curr_hidden_repeat.shape[2]))
                quit()
            if grid_pos.shape[1] > curr_pool_h.shape[1]: #or grid_pos.shape[2] > curr_pool_h.shape[2]:
                print("Second Shape Error!")
                print("dim 1: ")
                print("grid_pos dim1: " + str(grid_pos.shape[1]))
                print("curr_hidden_repeat dim1: " + str(curr_pool_h.shape[1]))
                #print("dim 2: ")
                #print("grid_pos dim2: " + str(grid_pos.shape[2]))
                #print("curr_hidden_repeat dim2: " + str(curr_pool_h.shape[2]))
                quit()

            if grid_pos.max() > curr_pool_h.shape[0]:
                print("ERROR EXCEED!")
                print(grid_pos.max())
                print(curr_pool_h.shape)
                #grid_pos[grid_pos > curr_pool_h.shape[0]] = 0
            if grid_pos.min() < 0:
                print("ERROR UNDER!")
                print(grid_pos.min())
                print(curr_pool_h.shape)
                grid_pos[grid_pos < 0] = 0
            #print("dim 0: ")

            print("grid_pos_max: " + str(grid_pos.max()))
            print("curr_pool_h shape[0]: " + str(curr_pool_h.shape[0]))
            #print("grid_pos dim0: " + str(grid_pos.shape[0]))
            #print("curr_hidden_repeat dim0: " + str(curr_hidden_repeat.shape[0]))
            #print("dim 1: ")
            #print("grid_pos dim1: " + str(grid_pos.shape[1]))
            #print("curr_hidden_repeat dim1: " + str(curr_hidden_repeat.shape[1]))
            #print("curr_hidden_pool dim1: " + str(curr_pool_h.shape[1]))
            print("curr_hidden_repeat: " + str(curr_hidden_repeat))
            print("dim: " + str(curr_hidden_repeat.shape))
            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,curr_hidden_repeat)
            print("curr_pool_h: " + str(curr_pool_h[:65]))
            print("dim: " + str(curr_pool_h.shape))
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))


        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)

        return pool_h


if __name__ == "__main__":
    """
    input: 
        - decoder_h:    1 x peds x dec_dim torch.tensor with latent-information about output
        - seq_start_end: nr_scenes x 2 torch.tensor with ped_ids for each sequence
        - curr_pos: peds x 2 torch.tensor with current position of prediction (relative)
    
    output: 
        - pool_h: peds x dec_dim torch.tensor with pooled hidden state information
    """

    # Assume 8 peds in a scene
    decoder_h_dim = 4
    activation_function = "relu"
    neighborhood_size = 10
    grid_size = 10
    pool_dim = None
    mlp_dim = 128

    pool_net = SocialPooling(h_dim=decoder_h_dim,
                             activation_function=activation_function,
                             batch_norm=False,
                             dropout=0.0,
                             neighborhood_size=neighborhood_size,
                             grid_size=grid_size,
                             pool_dim=pool_dim)

    mlp_dims_pool = [decoder_h_dim * 2, mlp_dim, decoder_h_dim]
    mlp = make_mlp(
        mlp_dims_pool,
        activation=activation_function,
        batch_norm=False,
        dropout=0.0
    )

    seq_start_end = torch.tensor([[0,8]])
    decoder_h = torch.tensor([[[1, 1.5, 2e-2, 3],
                               [-2e-3, 0, 3, 2],
                               [3, 0.5, 1e-2, 3],
                               [-1e-3, 0, -3, 2],
                               [0, 4.5, 4e-2, 3],
                               [-2e-3, 1, 0, 2],
                               [0, 1.3, 22e-2, 3],
                               [-7e-3, 0, 1, 2],
                               ]])
    curr_pos = torch.tensor([[1, 1.5],
                               [-2e-3, 0],
                               [3, 0.5],
                               [-1e-3, 0],
                               [0, 4.5],
                               [-2e-3, 1],
                               [0, 1.3],
                               [-7e-3, 0],
                               ])

    pool_h = pool_net(decoder_h, seq_start_end, curr_pos)
    decoder_h = torch.cat([decoder_h.view(-1, decoder_h_dim), pool_h], dim=1)
    decoder_h = mlp(decoder_h).unsqueeze(dim=0)

    print("finished")

    #state_tuple = (decoder_h, state_tuple[1])
