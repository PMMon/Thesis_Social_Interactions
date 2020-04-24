import torch

def move_opt_state_to_GPU(optimizer):
    print("moving state_dict of optimizer to GPU Memory...")
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()