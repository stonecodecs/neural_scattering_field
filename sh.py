## spherical harmonics related functions
import numpy as np
import torch
from PIL import Image
from scipy.special import factorial, lpmv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sh_basis(l, m, theta, phi):
    if m > 0:
        return np.sqrt(2) * K(l, m) * np.cos(m * phi) * lpmv(m, l, np.cos(theta))
    elif m == 0:
        return K(l, 0) * lpmv(l, 0, np.cos(theta))
    else:
        return np.sqrt(2) * K(l, -m) * np.sin(-m * phi) * lpmv(-m, l, np.cos(theta))
    

# normalization constant for SH basis
def K(l, m):
    return np.sqrt((2*l+1)/(4*np.pi) * factorial(l-m) / factorial(l+m))


# Project environment map to SH
# mainly for demonstration
# lmax=2 (9 coefficients is enough for SH lighting)
def project_to_sh(
    env_map,
    phase_function,
    sample_pdf,
    lmax=2,
    num_samples=64,
):
    """
    Projects environment map into spherical harmonic coefficients.
    Sampling is default to uniform distribution.

    Args:
        env_map (torch.tensor): _description_
        lmax (int, optional): level/order of SH. Defaults to 2.
        num_samples (int, optional): Samples to use for MC estimation. Defaults to 64.
        phase_function (_type_, optional): Function that takes random (u,v).
        sample_pdf (_type_, optional): Function that takes in (x) and outputs PDF of phase function.

    Returns:
        SH coefficients (3 for RGB, 9 coefficients)
    """
    H, W, _ = env_map.shape
    num_coeffs = (lmax + 1) ** 2
    c = torch.zeros((num_coeffs, 3))  # SH coefficients (RGB)

    # inverse transform sampling
    u, v = torch.rand(2, num_samples)
    theta_phi = phase_function(u, v)
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    i = torch.floor(((theta / np.pi) * (H-1))).type(torch.int)
    j = torch.floor(((phi / (2*np.pi)) * (W-1))).type(torch.int)
    L = env_map[i, j]  # (num_samples, 3)
    pdf = sample_pdf(theta) # technically cos_theta, but we assume isotropic so it doesn't matter (save computation)

    # SH coefficient
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = sh_basis(l, m, theta, phi)
            index = l * (l + 1) + m   # flattened idx
            c[index] += torch.sum(L / pdf.reshape(num_samples, 1) * Y.reshape(num_samples, 1), axis=0) # (3,)
            
    return c / num_samples 


def reconstruct_sh_image(sh_coeffs, lmax, output_shape=(256, 512)):
    """
    Reconstruct an environment map from SH coefficients.
    NOTE: This will only be called as a preview per # of steps,
          or computed once at the end of training.
    
    Args:
        sh_coeffs (np.ndarray): SH coefficients of shape (num_coeffs, 3).
        lmax (int): Maximum SH band (e.g., 2).
        output_shape (tuple): Output image dimensions (height, width).
    
    Returns:
        np.ndarray: Reconstructed RGB image of shape output_shape.
    """
    H, W = output_shape
    recon_image = torch.zeros((H, W, 3))
    
    # Create grid of spherical coordinates (theta, phi)
    theta = torch.linspace(0, np.pi, H)         # Polar angle (0 to pi)
    phi = torch.linspace(0, 2 * np.pi, W)       # Azimuthal angle (0 to 2pi)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
    
    # For each (l, m), add its contribution to the image
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            index = l * (l + 1) + m  # Flattened index
            coeff = sh_coeffs[index]  # SH coefficient (shape: (3,))
            
            # Compute SH basis function Y for all (theta, phi)
            Y = torch.zeros_like(theta_grid)
            for i in range(H):
                for j in range(W):
                    Y[i, j] = sh_basis(l, m, theta_grid[i, j], phi_grid[i, j])
            
            # Add contribution to the image (broadcast across RGB channels)
            recon_image += Y[..., np.newaxis] * coeff
    
    return recon_image


def plot_sh_texture(recon_image):
    """ Given reconstructed image from SH coefficients, plot reconstruction. """
    recon_normalized = (recon_image - recon_image.min()) / (recon_image.max() - recon_image.min())
    recon_normalized = (recon_normalized * 255).type(torch.int)

    plt.figure(figsize=(10, 5))
    plt.imshow(recon_normalized)
    plt.axis('off')
    plt.title("Reconstructed SH Texture")
    plt.show()


def visualize_spherical_samples(theta, phi, r=1):
    """
    Plots spherial (theta, phi) samples (as red dots) onto a unit sphere.
    Used to debug sampling distributions.

    Args:
        theta (ndarray): polar values for samples (N,)
        phi (ndarray): azimuthal values for samples (N,)
        r (int, optional): Radius of sample sphere. Defaults to 1.
    """
    def spherical_to_cartesian(theta, phi, r=r):
        """ Convert spherical to Cartesian """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    # generate sphere wireframe
    theta_vals = np.linspace(0, np.pi, 30)
    phi_vals = np.linspace(0, 2*np.pi, 30)
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
    x_sphere, y_sphere, z_sphere = spherical_to_cartesian(theta_grid, phi_grid)

    # convert samples to cartesian
    x_points, y_points, z_points = spherical_to_cartesian(theta, phi)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3)
    ax.scatter(x_points, y_points, z_points, color='red', s=100, label="Given Points")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
