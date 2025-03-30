import torch
import math
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def rotated_input(X, update=None):
    # Choose number of grid points along each dimension
    N_x, N_y, N_z, N_w = 25, 7, 6, 5  # for example

    # Discretize each domain
    if update is not None:
        x_vals, y_vals, z_vals, w_vals = update


    else:
        x_vals = torch.linspace(0, 2*math.pi, steps=N_x, device=device)
        y_vals = torch.linspace(-0.09, 0.09, steps=N_y, device=device)
        z_vals = torch.linspace(-0.09, 0.09, steps=N_z, device=device)
        w_vals = torch.linspace(0.9, 1 + 0.1, steps=N_w, device=device)


    X_grid, Y_grid, Z_grid, W_grid = torch.meshgrid(x_vals, y_vals, z_vals, w_vals, indexing='ij')

    # Flatten the grids to lists of parameter combos for batch processing
    param_count = X_grid.numel()
    X_flat = X_grid.reshape(-1)
    Y_flat = Y_grid.reshape(-1)
    Z_flat = Z_grid.reshape(-1)
    W_flat = W_grid.reshape(-1)

    # Prepare the original image (e.g., 1 x 256 x 256 tensor)
    original_img = X.to(device)

    # Set rotations and scales
    theta = torch.zeros(param_count, 2, 3, device=device)
    cosx = torch.cos(X_flat); sinx = torch.sin(X_flat)
    theta[:, 0, 0] = W_flat*cosx
    theta[:, 0, 1] = -W_flat*sinx
    theta[:, 1, 0] = W_flat*sinx
    theta[:, 1, 1] = W_flat*cosx
    # Set translations
    theta[:, 0, 2] = Y_flat
    theta[:, 1, 2] = Z_flat

    # Use affine_grid to get sampling grids, and grid_sample to apply the transformation
    grid = F.affine_grid(theta, torch.Size([param_count]) + original_img.shape[1:], align_corners=False)
    # Repeat the original image across the batch
    batch_img = original_img.expand(param_count, -1, -1, -1)
    transformed_imgs = F.grid_sample(batch_img, grid, align_corners=False)
    return transformed_imgs

def sample_update(previous_vals, F_vals, epoch_num):
    # Here ended up only updating the rotations, but can be changed to total update of all parameters.
    N_x, N_y, N_z, N_w = 25, 7, 6, 5
    prev_x_vals, prev_y_vals, prev_z_vals, prev_w_vals = previous_vals
    epsilon_x = 20/360 + (3/4)**epoch_num
    # epsilon_y_z = 0.18/10 + (1/2)**epoch_num

    N_x_idx = torch.max(torch.sum(F_vals.view(N_x, N_y*N_z*N_w), dim=1), dim=0)[1]
    N_y_idx = torch.max(torch.sum(F_vals.view(N_y, N_x*N_z*N_w), dim=1), dim=0)[1]
    N_z_idx = torch.max(torch.sum(F_vals.view(N_z, N_y*N_x*N_w), dim=1), dim=0)[1]

    point_x = prev_x_vals[N_x_idx]
    point_y = prev_y_vals[N_y_idx]
    point_z = prev_z_vals[N_z_idx]

    # Update x_vals, y_vals, z_vals, w_vals
    x_vals = torch.linspace(max(point_x - epsilon_x, 0), min(point_x + epsilon_x, 2*math.pi), steps=N_x, device=device)
    # y_vals = torch.linspace(max(point_y - epsilon_y_z, -0.09), min(point_y + epsilon_y_z, 0.09), steps=N_y, device=device)
    # z_vals = torch.linspace(max(point_z - epsilon_y_z, -0.09), min(point_z + epsilon_y_z, 0.09), steps=N_z, device=device)
    y_vals = torch.linspace(-0.09, 0.09, steps=N_y, device=device)
    z_vals = torch.linspace(-0.09, 0.09, steps=N_z, device=device)
    w_vals = torch.linspace(0.9, 1 + 0.1, steps=N_w, device=device)

    return x_vals, y_vals, z_vals, w_vals


def trap_int(X, A, sigma=None, exp=False, update=None):
    # Choose number of grid points along each dimension
    N_x, N_y, N_z, N_w = 25, 7, 6, 5
    # Discretize each domain
    if update!=None:
        x_vals, y_vals, z_vals, w_vals = update

    else:
        x_vals = torch.linspace(0, 2 * math.pi, steps=N_x, device=device)
        y_vals = torch.linspace(-0.09, 0.09, steps=N_y, device=device)
        z_vals = torch.linspace(-0.09, 0.09, steps=N_z, device=device)
        w_vals = torch.linspace(0.9, 1 + 0.1, steps=N_w, device=device)


    # Evaluate PDFs at grid points
    P_x = (1.0 / (2 * math.pi)) * torch.ones_like(x_vals, device=device)  # uniform [0,2π]
    P_y = (1.0 / (math.sqrt(2 * math.pi) * 0.045)) * torch.exp(-0.5 * (y_vals/0.045) ** 2)  # N(0,1) pdf (unnormalized outside [-∞,∞])
    P_z = (1.0 / (math.sqrt(2 * math.pi) * 0.045)) * torch.exp(-0.5 * (z_vals/0.045) ** 2)
    P_w = (1.0 / (2 * 0.1)) * torch.ones_like(w_vals, device=device)  # uniform [1 - epsilon,1 + epsilon]


    # Compute difference and its norm for each transformed image
    A = A.unsqueeze(0)
    transformed_imgs = rotated_input(A, update=update)
    transformed_imgs = transformed_imgs.squeeze(1)
    batch_img = X

    diff = batch_img - transformed_imgs    # shape: [param_count, 256, 256]
    # Flatten spatial dimensions and compute Euclidean norm per image
    # F_vals = diff.view(param_count, -1).norm(p=2, dim=1)      # shape: [param_count]
    if sigma is None:
        F_vals = -torch.norm(diff, p='fro', dim=(1,2)) ** 2 / 2

    else:
        F_vals = -torch.norm(diff, p='fro', dim=(1, 2)) ** 2 / (2*sigma**2)

    # Reshape F values back to 4D grid shape [N_x, N_y, N_z, N_w]
    F_grid = F_vals.view(N_x, N_y, N_z, N_w)

    # Create trapezoidal weights for each dimension
    w_x = torch.ones(N_x, device=device); w_x[0] = w_x[-1] = 0.5
    w_y = torch.ones(N_y, device=device); w_y[0] = w_y[-1] = 0.5
    w_z = torch.ones(N_z, device=device); w_z[0] = w_z[-1] = 0.5
    w_w = torch.ones(N_w, device=device); w_w[0] = w_w[-1] = 0.5

    # Broadcast weights and PDFs to the shape of F_grid
    trapezoid_weights = (w_x.view(N_x, 1, 1, 1) * w_y.view(1, N_y, 1, 1) *
                         w_z.view(1, 1, N_z, 1) * w_w.view(1, 1, 1, N_w))
    if exp:
        # Transforming the desired function into a form that will allow us to pass into log_sum_exp
        dx = (2 * math.pi) / N_x  # step size in x
        dy = 0.18 / N_y  # step size in y
        dz = 0.18 / N_z  # step size in z
        dw = 0.2 / N_w  # step size in w
        average = torch.ones_like(trapezoid_weights)*dx*dy*dz*dw
        pdf_values = (torch.log(P_x.view(N_x, 1, 1, 1)) + torch.log(P_y.view(1, N_y, 1, 1)) +
                      torch.log(P_z.view(1, 1, N_z, 1)) + torch.log((P_w.view(1, 1, 1, N_w)))
                                                        + torch.log(trapezoid_weights) + torch.log(average))
        integrand_grid = F_grid + pdf_values
        integral_estimate = torch.logsumexp(integrand_grid, dim=(0, 1, 2, 3))
        return integral_estimate, F_grid

    # Here we are computing the integral for the pre-model
    pdf_values = P_x.view(N_x, 1, 1, 1) * P_y.view(1, N_y, 1, 1) \
                                        * P_z.view(1, 1, N_z, 1) * P_w.view(1, 1, 1, N_w)

    # Compute the integrand values on the grid (F * PDFs)
    integrand_grid = F_grid * pdf_values

    # Sum up all contributions with trapezoidal weighting
    sum_val = torch.sum(integrand_grid * trapezoid_weights)

    # Multiply by the volume of one hyper-rectangular cell (dx * dy * dz * dw)
    dx = (2 * math.pi) / N_x  # step size in x
    dy = 0.18 / N_y  # step size in y
    dz = 0.18 / N_z  # step size in z
    dw = 0.2 / N_w  # step size in w
    integral_estimate = sum_val * dx * dy * dz * dw
    return integral_estimate, F_grid

