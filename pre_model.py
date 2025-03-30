import torch
from torch import nn
import matplotlib
from integration import trap_int, sample_update
import math


matplotlib.use('Agg')  # Set the backend to TkAgg
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self, update_samples=None, len_dataset=100):
        super(Model, self).__init__()
        N_x, N_y, N_z, N_w = 25, 7, 6, 5
        self.theta = nn.Parameter(torch.zeros(1, 256, 256))
        self.update_samples = update_samples
        self.num_batch = 0
        self.num_epoch = 0
        self.len_dataset = len_dataset
        self.grid = torch.zeros(self.len_dataset//5, 5, N_x, N_y, N_z, N_w)

        x_vals = torch.linspace(0, 2 * math.pi, steps=N_x, device=device)
        y_vals = torch.linspace(-0.09, 0.09, steps=N_y, device=device)
        z_vals = torch.linspace(-0.09, 0.09, steps=N_z, device=device)
        w_vals = torch.linspace(0.9, 1.1, steps=N_w, device=device)

        self.prev_samples = [[x_vals, y_vals, z_vals, w_vals] for i in range(100)]

    def forward(self, x):
        A = self.theta
        k = self.num_epoch
        j = self.num_batch
        n = x.shape[0]
        loss = 0
        grid_lst = []
        for i in range(n):
            if self.update_samples is not None:
                f_grid = self.update_samples[j][i]
                prev_samples = self.prev_samples[5 * j + i]
                update = sample_update(prev_samples, f_grid, epoch_num=k)
                self.prev_samples[5 * j + i] = [update[0], update[1], update[2], update[3]]
                loss_1, f_grid = trap_int(x[i], A, update=update)
                grid_lst.append(f_grid)
                loss += torch.sum(loss_1)
            else:
                loss_1, f_grid = trap_int(x[i], A)
                grid_lst.append(f_grid)
                loss += torch.sum(loss_1)
        f_grid = torch.stack(grid_lst, dim=0)
        self.grid[j] = f_grid
        grid = self.grid
        return loss - 0.5*torch.norm(A, p=1), A.squeeze(0), grid
