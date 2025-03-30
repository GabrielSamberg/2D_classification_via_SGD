import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
from duck_dataset_playground import RotatedImageDataset
from integration import trap_int, sample_update
import math
from pre_model import Model
import mrcfile
import os
import numpy as np
import argparse


matplotlib.use('Agg')  # Set the backend to TkAgg
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Model2(nn.Module):
    def __init__(self, update_samples=None):
        super(Model2, self).__init__()
        N_x, N_y, N_z, N_w = 25, 7, 6, 5
        self.theta = nn.Parameter(torch.zeros(1, 256, 256))
        self.sigma = nn.Parameter(torch.ones(1))

        self.update_samples = update_samples
        self.num_batch = 0
        self.num_epoch = 0
        self.grid = torch.zeros(20, 5, N_x, N_y, N_z, N_w)  # (num_batches, batch, N_x, N_y, N_z, N_w)

        x_vals = torch.linspace(0, 2 * math.pi, steps=N_x, device=device)
        y_vals = torch.linspace(-0.09, 0.09, steps=N_y, device=device)
        z_vals = torch.linspace(-0.09, 0.09, steps=N_z, device=device)
        w_vals = torch.linspace(0.9, 1.1, steps=N_w, device=device)

        self.prev_samples = [[x_vals, y_vals, z_vals, w_vals] for i in range(100)]  # integration samples for each data sample

    def forward(self, x):
        sigma = torch.sigmoid(self.sigma) + 1e-12
        A = self.theta
        j = self.num_batch
        k = self.num_epoch
        n = x.shape[0]
        loss = 0
        grid_lst = []
        for i in range(n):
            if self.update_samples is not None:
                f_grid = self.update_samples[j][i]
                prev_samples = self.prev_samples[5 * j + i]
                update = sample_update(prev_samples, f_grid, epoch_num=k)
                self.prev_samples[5 * j + i] = [update[0], update[1], update[2], update[3]]
                loss_1, f_grid = trap_int(x[i], A, exp=True, update=update)
                grid_lst.append(f_grid)
                loss += loss_1
            else:
                loss_1, f_grid = trap_int(x[i], A, exp=True)
                grid_lst.append(f_grid)
                loss += loss_1
        f_grid = torch.stack(grid_lst, dim=0)
        self.grid[j] = f_grid
        grid = self.grid
        return loss - 0.5*torch.norm(A, p=1) - torch.log(sigma), A.squeeze(0), grid


class MRCSDataSet(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.mrcs')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])

        with mrcfile.open(file_path, permissive=True) as mrc:
            # Convert data to a numpy array
            data = np.array(mrc.data, dtype=np.float32)

            # You might need to normalize or preprocess data differently depending on your task
            data = (data - np.mean(data)) / np.std(data)

            # Assuming data is already in a shape compatible with your network, otherwise reshape
            # For example, if you have single-channel 2D data for CNN:
            data = data.reshape(1, 256, 256)

        # Return as tensor
        return torch.tensor(data)


def train(epochs, lr_theta, lr_sigma, train_pre_model=False, pre_epochs=0, mrcs_directory=None, generate_dataset=False):
    assert (mrcs_directory == None) or (generate_dataset == False), \
        "Cant generate dataset and load existing one at the same call"
    if type(mrcs_directory) == str:
        dataset = MRCSDataSet(mrcs_directory)

    if generate_dataset == True:
        image_path = 'Trump.png'
        image = Image.open(image_path).convert('L')
        dataset = RotatedImageDataset(image)

    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

    # pre model
    pre_model = Model().to(device)
    pre_optimizer = optim.SGD(pre_model.parameters(), lr=0.01, momentum=0.8, nesterov=True)
    pre_scheduler = StepLR(pre_optimizer, step_size=4, gamma=0.1)

    # model
    model2 = Model2().to(device)

    if train_pre_model:
        # Training the pre model
        pre_model.train()
        pre_model.num_epoch = 0
        loss_lst = []
        for epoch in range(pre_epochs):
            pre_model.num_epoch += 1
            pre_model.num_batch = 0
            for data in dataloader:
                data = data.to(device)
                pre_optimizer.zero_grad()
                # Forward pass
                output, A, grid = pre_model(data)
                if epoch > 0:
                    pre_model.update_samples = grid
                loss = -output
                # Backward pass
                loss.backward()
                pre_optimizer.step()
                pre_model.num_batch += 1

            pre_scheduler.step()  # Update the learning rate
            #
            # You can access the current learning rate
            relative_loss = 'Not yet calculated'
            loss_lst.append(loss.item())
            if epoch >2:
              relative_loss = abs(loss_lst[-1] - loss_lst[-2]) / abs(loss_lst[-1]) + \
              abs(loss_lst[-2] - loss_lst[-3]) / abs(loss_lst[-2])
            current_lr = pre_optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f},Relative_loss:{relative_loss}, Learning_Rate: {current_lr}, pre-model running')

    # optimize the model
    model2.theta = pre_model.theta
    optimizer = optim.SGD([{'params': [model2.theta],
                            'lr': lr_theta}, {'params': [model2.sigma], 'lr': lr_sigma}], momentum=0.8,
                          nesterov=True)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)  # Reduce LR by 90% every 30 epochs
    model2.num_epoch = 0
    for name, param in model2.named_parameters():
        if name == 'theta':
                A = param.squeeze(0)
                img = A.detach().to('cpu').numpy()
                plt.imshow(img, interpolation='nearest')
                plt.axis('off')
                plt.title('Initial Theta')
                plt.savefig(f'Model initial Theta.png')
                print(f'Plot saved to Model initial Theta.png')
    loss_lst = []
    for epoch in range(epochs):
        model2.train()
        model2.num_epoch += 1
        model2.num_batch = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            # Forward pass
            output, A, grid = model2(data)
            if epoch > 0:
                model2.update_samples = grid  # passing down the current function values on the grid for sample update
            loss = -output
            # Backward pass
            loss.backward()
            optimizer.step()
            model2.num_batch += 1

        scheduler.step()  # Update the learning rate
        #
        # You can access the current learning rate
        current_lr_theta = optimizer.param_groups[0]['lr']
        current_lr_sigma = optimizer.param_groups[1]['lr']
        relative_loss = 'Not yet calculated'
        loss_lst.append(loss.item())
        if epoch > 2:
          relative_loss = abs(loss_lst[-1] - loss_lst[-2]) / abs(loss_lst[-1]) + \
          abs(loss_lst[-2] - loss_lst[-3]) / abs(loss_lst[-2])
        print(
            f'Epoch {epoch + 1}, Loss: {loss.item():.4f},Relative_loss:{relative_loss}, Learning_Rates: Sigma: {current_lr_sigma}, Theta:{current_lr_theta}, Model running')
        if epoch % 4 == 0:
          for name, param in model2.named_parameters():
              if name == 'theta':
                  A = param.squeeze(0)
                  img = A.detach().to('cpu').numpy()
                  plt.imshow(img, interpolation='nearest')
                  plt.axis('off')
                  plt.title(f'Theta after {epoch + 1} epochs')
                  plt.savefig(f'Model epoch {epoch + 1}.png')
                  print(f'Plot saved to Model epoch {epoch + 1}.png')
    model2.eval()
    with torch.no_grad():
        for name, param in model2.named_parameters():
            if name == 'theta':
                A = param.squeeze(0)
                img = A.detach().to('cpu').numpy()
                plt.imshow(img, interpolation='nearest')
                plt.axis('off')
                plt.title('Model output')
                plt.savefig(f'Model_output.png')
                print("Plot saved to Model output.png")

    return


def parse_args():
    parser = argparse.ArgumentParser(description='Model training with MRC dataset')


    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the .mrc format dataset')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--learning_rate_theta', type=float, default=0.01,
                        help='Learning rate for optimizer (default: 0.01)')
    parser.add_argument('--learning_rate_sigma', type=float, default=0.001,
                        help='Learning rate for optimizer (default: 0.001)')
    parser.add_argument('--train_pre_model', type=bool, default=True,
                        help = 'Run pre-model optimization for better initialization of the model')
    parser.add_argument('--pre_epochs', type=int, default=7,
                        help='Number of epochs for the pretraining proccess')
    parser.add_argument('--generate_dataset', type=bool, default=False,
                        help='Generating the dataset instead of loading one from given path')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    if args.data_path is None:
        print("No data path provided. Generating our dataset instead")
        args.generate_dataset = True

    else:
        # Validate the data path exists and is an .mrc file
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data path not found: {args.data_path}")


    # Print training configuration
    print("Training configuration:")
    print(f"  Dataset: {args.data_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate theta: {args.learning_rate_theta}")
    print(f"  Learning rate sigma: {args.learning_rate_sigma}")
    print(f"  Train pre model: {args.train_pre_model}")
    print(f"  Epochs for pre model: {args.pre_epochs}")
    print(f"  Generating dataset: {args.generate_dataset}")

    train(args.epochs, args.learning_rate_theta, args.learning_rate_sigma, train_pre_model=args.train_pre_model, pre_epochs=args.pre_epochs, mrcs_directory=args.data_path, generate_dataset=args.generate_dataset)


if __name__ == '__main__':
    main()

