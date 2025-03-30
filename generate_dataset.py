import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import Beta, Normal
import torch.nn.functional as F
import math
import mrcfile
import os

matplotlib.use('Agg')  # Set the backend to TkAgg


class RotatedImageDataset(Dataset):
    """
    A PyTorch Dataset that generates and stores n rotated versions of an original image.
    """

    def __init__(self, original_image, num_samples=100, transform=None):
        """
        Args:
            original_image (PIL.Image or str): The original image or path to the image
            rotation_function (callable): Function that takes an image and returns n rotated versions
            n_rotations (int): Number of rotated images to generate
            transform (callable, optional): Additional transformations to apply to the rotated images
        """
        self.transform = transform

        # Load the image if a path is provided
        if isinstance(original_image, str):
            self.original_image = Image.open(original_image).convert('L')
        else:
            self.original_image = original_image

        # Generate the n rotated images using the provided function
        self.rotated_images = batch_affine_transform_dataset(self.original_image , batch_size=num_samples)

        # Validate that we received the expected number of images
        assert len(
            self.rotated_images) == num_samples, f"Expected {num_samples} rotated images, but got {len(self.rotated_images)}"

    def __len__(self):
        """Return the total number of rotated images in the dataset."""
        return len(self.rotated_images)

    def __getitem__(self, idx):
        """
        Return the rotated image at the specified index.

        Args:
            idx (int): Index of the rotated image to retrieve

        Returns:
            torch.Tensor: The rotated image as a tensor
        """
        rotated_image = self.rotated_images[idx]

        if self.transform:
            rotated_image = self.transform(rotated_image)

        return rotated_image



def batch_affine_transform_dataset(image, batch_size, device="cuda:0" if torch.cuda.is_available() else "cpu"):
    """
    Apply different affine transformations to a single image in a batched manner.

    Args:
        image (torch.Tensor): Input image tensor of shape [C, H, W]
        batch_size (int): Number of different transformations to apply
        device (str): Device to run computations on

    Returns:
        torch.Tensor: Transformed images of shape [batch_size, C, H, W]
    """

    to_tensor = transforms.ToTensor()
    center_crop = transforms.CenterCrop(256)  # change back
    tranform = transforms.Resize((256, 256))
    image = to_tensor(image)
    image = tranform(image)
    image = center_crop(image)
    image = tranform(image)
    image = image.to(device)
    # image = image + 0.08*torch.randn_like(image)

    image.to(device)
    # Get image dimensions
    C, H, W = image.shape

    # Create a batch of images by repeating the input image
    batched_images = image.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Create random transformation parameters for each image in the batch
    # For rotation: angle in radians between -30 and 30 degrees
    angles = torch.rand(batch_size, 1, device=device)  # -30 to 30 degrees Changed to [0,1]
    angles = angles * (2*math.pi)  # Convert to radians

    # For translation: x and y shifts between -0.2 and 0.2 (relative to image size)
    sigma = 0.045
    alpha = torch.full((batch_size, 1), 0.0, device=device)  # Mean value
    beta = torch.full((batch_size, 1), sigma, device=device)

    epsilon = 0.1
    scale = torch.empty(batch_size, 1).uniform_(1 - epsilon, 1 + epsilon).to(device)


    # Initialize the Normal distribution
    normal = Normal(alpha, beta)
    a, b = normal.sample(), normal.sample()
    translations_x = torch.min(torch.max(a, - 0.09 * torch.ones_like(a)), 0.09 * torch.ones_like(a))
    translations_y = torch.min(torch.max(b, - 0.09 * torch.ones_like(b)), 0.09 * torch.ones_like(b))

    theta = torch.zeros(batch_size, 2, 3, device=device)

    # Fill in rotation and scale parameters
    theta[:, 0, 0] = (scale*torch.cos(angles)).squeeze(1)
    theta[:, 0, 1] = -(scale*torch.sin(angles)).squeeze(1)
    theta[:, 1, 0] = (scale*torch.sin(angles)).squeeze(1)
    theta[:, 1, 1] = (scale*torch.cos(angles)).squeeze(1)

    # Fill in translation parameters
    theta[:, 0, 2] = (translations_x).squeeze(1)
    theta[:, 1, 2] = (translations_y).squeeze(1)

    # Define the output size (same as input)
    out_size = (H, W)

    # Apply the grid_sample-based affine transformation to all images at once
    grid = F.affine_grid(theta, batched_images.size(), align_corners=False)
    transformed_images = F.grid_sample(batched_images, grid, align_corners=False)

    transformed_images = transformed_images + 0.08*torch.randn_like(transformed_images)

    return transformed_images


if __name__ == '__main__':
    def show_image(img, i):
        plt.imshow(img, interpolation='nearest')
        plt.axis('off')
        plt.savefig(f'output_plot_{i}.png')
        print(f'Plot saved to output_plot_{i}.png')


    # Load your image (assuming you have it as 'path_to_your_image.jpg')
    image_path = 'Socrates.png'
    image = Image.open(image_path).convert('L')
    to_tensor = transforms.ToTensor()
    center_crop = transforms.CenterCrop(256)
    tranform = transforms.Resize((256, 256))
    img = to_tensor(image)
    img = tranform(img)
    img = center_crop(img)
    im = tranform(img).squeeze(0)
    plt.imshow(im, interpolation='nearest')
    plt.axis('off')
    plt.title('Ground Truth')
    plt.savefig(f'ground_truth_duck_croped.png')
    print(f'Plot saved to ground_truth_duck_croped.png')

    # Create dataset
    dataset = RotatedImageDataset(image)

    output_dir = 'MRCS_files'
    os.makedirs(output_dir, exist_ok=True)

    # for idx, data in enumerate(dataset):
    #     # Extract tensor (assuming data is (tensor,) or directly a tensor)
    #     tensor = data[0] if isinstance(data, (tuple, list)) else data
    #
    #     # Convert to numpy array (ensure float32)
    #     np_data = tensor.cpu().numpy().astype('float32')
    #
    #     # Define file path
    #     file_path = os.path.join(output_dir, f'image_{idx}.mrcs')
    #
    #     # Save as MRCS
    #     with mrcfile.new(file_path, overwrite=True) as mrc:
    #         mrc.set_data(np_data)
    #
    #     print(f"Saved: {file_path}")

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    # Show some transformed images
    for i, img in enumerate(dataloader):
        show_image(transforms.functional.to_pil_image(img[0].squeeze(0)), i)
        if i == 3:  # Show 5 images
            break