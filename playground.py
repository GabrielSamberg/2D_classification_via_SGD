import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch import nn, optim
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg
import matplotlib.pyplot as plt
import random
from torch.distributions import Beta, Normal
import time
from duck_dataset import CustomImageDataset


mps_device = torch.device("mps")
epsilon = 1e-3


def affine_a(A, angle, translation, scale):
    affine = v2.functional.affine(A, angle=angle, translate=translation, scale=scale, shear=[0])
    # resize_image = transforms.Resize((256, 256))
    # affine_A = resize_image(affine_A)
    return affine


def truncated_translation_gaussian(sigma):
  normal = Normal(0, sigma)
  a, b = normal.sample(), normal.sample()
  if torch.abs(a) > 2*sigma:
    a = 0
  elif torch.abs(b) > 2*sigma:
    b = 0

  return a, b


def trapezodial_integral(A, X, sample_num=20):
    # A lot of moving parts here, triple check for correctness
    dots_rotations = torch.linspace(0, 2 * np.pi, sample_num)
    dots_trans_1 = torch.linspace(-10, 10, sample_num)
    dots_trans_2 = torch.linspace(-10, 10, sample_num)
    dots_scale = torch.linspace(4 / 5, 1 + 4 / 5, sample_num)

    h_1 = 2 * np.pi / sample_num
    h_2 = 20 / sample_num
    h_3 = 20 / sample_num
    h_4 = 1 / sample_num

    beta_pdf = lambda x: 30 * (x - 4 / 5) * (1 + 4 / 5 - x) ** 4
    norm_pdf = lambda x: torch.exp(-x ** 2 / 200)
    func = lambda x, y, z, w: torch.exp(-(torch.norm(X - affine_a(A, w.item(), [y.item(), z.item()], x.item()), p=1) ** 2) / 2) + epsilon

    first_integral = lambda x, y, z: h_1 / 2 * (func(x, y, z, dots_rotations[0]) + func(x, y, z, dots_rotations[-1]) +
                                                2 * torch.sum(torch.tensor([func(x, y, z, i) for i in dots_rotations[1:-1]], device=mps_device)))

    second_integral = lambda x, y: h_2 / 2 * (norm_pdf(dots_trans_1[0]) * first_integral(x, y, dots_trans_1[0]) +
                                              norm_pdf(dots_trans_1[-1]) * first_integral(x, y, dots_trans_1[-1])
                                              + 2 * torch.sum(torch.tensor([norm_pdf(i) * first_integral(x, y, i) for i in dots_trans_1[1:-1]], device=mps_device)))

    third_integral = lambda x: h_3 / 2 * (norm_pdf(dots_trans_2[0]) * second_integral(x, dots_trans_2[0]) +
                                          norm_pdf(dots_trans_2[-1]) * second_integral(x,dots_trans_2[-1])
                                          + 2 * torch.sum(torch.tensor([norm_pdf(i) * second_integral(x, i) for i in dots_trans_2[1:-1]], device=mps_device)))

    forth_integral = (1 / (2 * np.pi)) ** 2 * h_4 / 2 * (beta_pdf(dots_scale[0]) * third_integral(dots_scale[0]) +
                                                         beta_pdf(dots_scale[-1]) * third_integral(dots_scale[-1]) +
                                                         2 * torch.sum(torch.tensor([beta_pdf(i) * third_integral(i) for i in dots_scale[1:-1]], device=mps_device)))

    return forth_integral


def integral(A, X, sample_num=1000):
    angles = [random.randint(1, 72) for i in range(sample_num)]
    translations = [[truncated_translation_gaussian(10)[0], truncated_translation_gaussian(10)[1]] for i in range(sample_num)]
    scales = [(7/2)*Beta(torch.FloatTensor([2]), torch.FloatTensor([5])).sample() for i in range(sample_num)]
    int = 0
    for i in range(sample_num):
        angle = angles[i]
        translation = translations[i]
        scale = scales[i]
        affine_A = affine_a(A, angle, translation, scale)
        int += -torch.norm(X - affine_A, p=1)**2/2  #playing between Frobenius norm and l2

    return int/sample_num


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.theta = nn.Parameter(torch.randn(1, 256, 256))

    def forward(self, x):
        n = x.shape[0]
        A = self.theta
        loss = 0
        for i in range(n):
          loss += torch.log(integral(A, x[i]) + epsilon)
        return loss , A.squeeze(0)


image_path = 'duck.png'
image = Image.open(image_path).convert('L')
dataset = CustomImageDataset(image)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
# Instantiate the model
model = Model().to(mps_device)

# Loss function (since we need to maximize, we minimize the negative of our target function)
def loss_fn(output):
    return -output  # Minimize the negative to maximize the original function

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
  for data in dataloader:
    data = data.to(mps_device)
    optimizer.zero_grad()
    # Forward pass
    output = model(data)[0]
    loss = loss_fn(output)
    # Backward pass
    loss.backward()
    optimizer.step()

  print(f'Epoch {epoch+1}, Loss: {loss.item()}')



for name, param in model.named_parameters():
  print(name, param.shape)
  A = param.squeeze(0)
  A = A.detach().numpy()
  plt.imshow(A, interpolation='nearest')
  plt.axis('off')
  plt.title('A with params')
  plt.show()


model.eval()

# Simulate input data and forward pass
with torch.no_grad():
  for data in dataloader:
    input_data = data
    A = model(input_data)[1]

    # Plotting the output
    output_numpy = A.numpy()
    plt.imshow(output_numpy, interpolation='nearest')
    plt.axis('off')
    plt.title('Model Outputs with eval')
    plt.show()