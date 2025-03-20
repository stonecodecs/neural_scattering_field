## NeRF essentials beyond the model itself
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Tuple, Callable
from utils import prepare_chunks, local_to_world_rotation, spherical_to_cartesian, cartesian_to_spherical
from sh import project_to_sh, sh_basis


def lerp(
    t: Union[torch.tensor, float],
    near: Union[torch.tensor, float],
    far: Union[torch.tensor, float]
) -> Union[torch.tensor, float]:
    """Linear interpolation between near and far planes."""
    return near + (far - near) * t


## phase function stuff
def sample_isotropic(u, v):
    """
    Sample uniformly across the sphere. 
    Returns theta, phi.
    NOTE: when using with sampling, the result is in RAY coordinates. Change to world coordinates.
    """
    theta = np.arccos(np.clip(2 * u - 1, -1, 1))
    phi = 2 * np.pi * v
    return torch.stack([theta, phi], dim=-1)


def isotropic_pdf(x=0):
    return np.full(len(x), 1 / (4 * np.pi))


def sample_henyey_greenstein(u, v, g):
    """
    HG phase function parametrized by 'g' ranging from [-1, 1]. (u,v) are uniformly sampled.
    NOTE: when using with sampling, the result is in RAY coordinates. Change to world coordinates.
    """
    # u is "cos_theta" in HG
    # g: Asymmetry parameter (-1 is full back-scattering, 1 is full frontal scattering)
    if g == 0: # isotropic, avoid 0 denom
        cos_theta = 2 * u - 1 
    else:
        # HG CDF
        cos_theta = (1 + g**2 - ((1 - g**2) / (1 - g + 2 * g * u))**2) / (2 * g)
    
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    phi = 2 * np.pi * v
    return torch.stack([theta, phi], dim=-1)

def henyey_greenstein_pdf(cos_theta, g):
    return (1 - g**2) / (4 * np.pi * (1 + g**2 - 2 * g * cos_theta)**(3/2))


def gen_HG_pair(g):
    """ Helper function to quickly generate HG(g) and its PDF(g). """
    hg  = lambda u,v,g=g: sample_henyey_greenstein(u,v,g)
    pdf = lambda   x,g=g: henyey_greenstein_pdf(x, g)
    return hg, pdf


def get_camera_poses(poses: torch.tensor, transpose=False, normalize=False) -> Tuple[torch.tensor, torch.tensor]:
    """
    Extracts camera origins and -z vectors from 4x4 extrinsic matrices.

    Args:
        poses (torch.tensor): Camera poses (N, 4, 4)
        transpose (bool): Depending on the dataset, this is either 
        normalize_range (bool): normalize translational vectors to the range specified TODO

    Returns:
        torch.tensor: Camera origins (N, 3)
        torch.tensor: Camera directions (-Z unit vectors) (N, 3)
    """
    poses_ = poses.T if transpose else poses # if column-wise
    cam_origins = poses_[..., 0:3, 3] # translation components of matrix
    cam_directions = -poses_[..., 0:3, 2] # equivalent to -z vector [0,0,-1]
    return cam_origins / cam_origins.max() if normalize else cam_origins, cam_directions


def get_rays(cam_extrinsics: torch.tensor,
    H: int, W: int, focal_x: float, focal_y: float=None, batched=False
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Compute rays through each pixel in a (H, W) viewport grid akin to ray tracing.
    Returns ray origins (H, W, 3) and directions (H, W, 3) for a single image.

    Parameters
    ----------
    cam_extrinsics : torch.tensor
        (4,4) transformation matrix (camera to world).
    H : int
        Viewport height (in pixels)
    W : int
        Viewport width (in pixels)
    focal_x : float
        Horizontal focal length of camera (mm) 
    focal_y : Optional::float
        Vertical focal length
    batched : bool
        If cam_extrinsics is batched (N, 4, 4), returns (N, H, W, 3)

    Returns
    -------
    torch.tensor
        torch.tensor: Ray origins (H,W,3), ray directions (H,W,3).
        (N,H,W,3) if batched.
    """
    if focal_y is None:
        focal_y = focal_x
    if batched and len(cam_extrinsics.shape) == 3:
        batched_rays = torch.vmap(get_rays, in_dims=(0, None, None, None))
        return batched_rays(cam_extrinsics, H, W, focal_x, focal_y=focal_y, batched=False)

    # principal points in the center
    cx, cy = W / 2, H / 2 

    # generate pixel grid
    u, v = torch.meshgrid(
        torch.arange(0, H, 1, dtype=torch.float32, device=cam_extrinsics.device),
        torch.arange(0, W, 1, dtype=torch.float32, device=cam_extrinsics.device),
        indexing="xy"
    )

    # NOTE: depending on interface, z coordinate may be positive instead of negative
    ndc = torch.stack([(u - cx) / focal_x,
                          -(v - cy) / focal_y,
                          -torch.ones_like(u)
                      ], dim=-1)

    # NDC to world coordinates using extrinsic matrix
    # rays_d = ndc @ cam_extrinsics[..., :3, :3].T # transformed (rotated) NDC
    rays_d = torch.sum(ndc[..., None, :] * cam_extrinsics[:3, :3], dim=-1)
    rays_o = cam_extrinsics[..., :3, -1].expand(rays_d.shape) # translation component
    
    return rays_o, rays_d


def sample_rays(
    rays_o: torch.tensor,
    rays_d: torch.tensor, 
    near:int, 
    far:int,
    num_samples:int=64, 
    perturb: Optional[bool] = True,
    inverse_depth: Optional[bool] = False,
    batched: bool=False
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Performs stratified sampling on the given rays from ray_{o,d} between
    near and far plane values. If perturbed, then spacing is randomized.
    Else, the sampling is deterministic, spread equally throughout the planes.
    If batched, then returns [N,] prefixed to the return object shape. 

    Parameters
    ----------
    rays_o : torch.tensor
        Ray origins. ([N,] H, W, 3)
    rays_d : torch.tensor
        Ray directions. ([N,] H, W, 3)
    num_samples : int
        Number of samples to take for each ray.
    near : int
        Near plane as the lower bound to take samples from.
    far : int
        Far plane as the upper bound to take samples from.
    perturb : Optional[bool], optional
        Sample stochasticly from bins following the uniform distribution, by default True
    inverse_depth : Optional[bool], optional
        Follow inverse depth interpolation (inverse LERP), by default False.
        Most probability mass is distributed where 't' is closer to the near plane.
    batched : bool, optional
        If ray_{o,d} are batched, by default False

    Returns
    -------
    torch.tensor
        Samples along the ray in 3D world coordinates ([N,] H,W,S,3),
        and their depth values along corresponding rays ([N,] H,W,S).
        [N,] prefixed if batched == True.
    """

    if batched and len(rays_o.shape) == 4 and len(rays_d.shape) == 4:
        batched_samples = torch.vmap(
            sample_rays,
            in_dims=(0,0, None, None, None), 
            randomness='different')
        return batched_samples(
            rays_o, rays_d, near, far, num_samples,
            perturb=perturb, inverse_depth=inverse_depth, batched=False)
    
    # sample from valid ray space where t: [0, 1)
    samples_t = torch.linspace(0.0, 1.0, num_samples,
                               dtype=torch.float32,
                               device=rays_o.device)

    # interpolation technique to sample from actual 'z' space: [near, far)
    z_vals = 1.0 / lerp(samples_t, near, far) if inverse_depth \
             else  lerp(samples_t, near, far)
    
    if perturb: # uniform stratified sampling
        mids   = (z_vals[1:] + z_vals[:-1]) / 2                  # use as bin boundaries
        lower  = torch.concat([z_vals[:1], mids], dim=-1)        # add min bound
        upper  = torch.concat([mids, z_vals[-1:]], dim=-1)       # add max bound
        t_rand = torch.rand([num_samples], device=rays_o.device) # (S,) random 't's [0,1)
        z_vals = lerp(t_rand, lower, upper)                      # interpolate with above

    z_vals = z_vals.expand(*rays_o.shape[:-1], num_samples)      # (H,W,S)
    
    # sample points in world coordinates (follows ray equation: r = o + dz)
    samples_3d = rays_o.unsqueeze(-2) + \
                 rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
    
    return samples_3d, z_vals

import numpy as np

def eval_env_map(direction, env_map):
    """
    Sample an environment map (assumed to be in equirectangular format)
    given a 3D directional vector.
    
    Args:
        direction (np.array): 3D normalized direction (S, 3).
        env_map (np.array): The environment map image as an array with shape (H, W, 3).
    
    Returns:
        torch.Tensor: The sampled color (or value) from the environment map. (S, 3)
    """
    xyz = direction / torch.norm(direction, dim=-1, keepdim=True)
    
    # convert to spherical coordinates.
    theta, phi = cartesian_to_spherical(xyz).unbind(-1)
    
    # map to positive normalized texture coordinates (u, v)
    u = phi / (2 * np.pi)  # horizontal
    v = theta / np.pi  # vertical
    
    # convert normalized coordinates to pixel coordinates
    height, width, _ = env_map.shape
    pixel_x = torch.floor(u * width % width).type(torch.int)
    pixel_y = torch.floor(v * height % height).type(torch.int)
    return env_map[pixel_y, pixel_x]


def eval_sh(coefs, d, lmax=2, spherical=True):
    """ Get radiance of SH lighting from direction d, given its coefficients. """
    # NOTE: make sure that the direction 'd' is world coordinates
    # an easy to miss bug would be using a direction straight from a sampler (still in ray coordinates)

    # lmax and coefs need to correspond to each other
    if spherical:
        theta, phi = d[..., 0], d[..., 1]
    else: # Cartesian
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        theta = torch.acos(z)
        phi = torch.atan2(y,x)

    L = 0.0  # radiance
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = sh_basis(l, m, theta, phi)
            index = l * (l + 1) + m   # flattened idx
            L += coefs[index] * Y
    return L


def sample_next_directions(ray_dir, phase_function, num_samples):
    """
    Samples 'num_samples' random directions given a phase function.
    Automatically converts into world coordinates. 

    Args:
        ray_dir (3 vector): current direction of the main ray before scattering ([N,]H,W,3)
            (Meaning: direction of ray for all "pixels")
        phase_function (function): phase function that takes in UV random [0,1] values
        num_samples (int): Number of directions to sample.

    Returns:
        Sample directions in shape ([N,], S, 3).
    """
    orig_shape = ray_dir.shape
    ray_dir_flat = ray_dir.reshape(-1, 3)
    R = local_to_world_rotation(ray_dir_flat) # (NHWS,3,3)
    theta_phi = phase_function(*torch.rand(2, num_samples)) # w.r.t the current ray direction [-1, 1]
    local_dirs = spherical_to_cartesian(theta_phi) # (num_samples, 3)
    world_dirs = torch.einsum("bij,kj->bki", R, local_dirs)
    # sample_dirs = R @ spherical_to_cartesian(theta, phi)
    # return sample_dirs  # (3, num_samples) matrix
    return world_dirs.view(*orig_shape[:-1], num_samples, 3)


def single_scatter(
    env_map,
    ray_dirs,
    sample_dirs, # directions of sampling (from predicted g)
    sigma_s,
    g,
    phase_function_pdf=henyey_greenstein_pdf,
    num_samples=64
):
    """
    Returns the approximate light contribution using HG phase function to query the environment map.

    Args:
        env_map (image): env map for lighting
        x (vec3): 3D position of sample points along rays ([N]HWS, 3)
        ray_dir (torch.tensor): ray direction per sample ([N,] H, W, 3) omit S duplicates
        sigma_s (vec3): scattering coef
        g (float): HG parameter
        visibility_network (_type_): VisibilityMLP
        phase_function (_type_, optional): phase function sampler. Defaults to sample_henyey_greenstein.
        phase_function_pdf (_type_, optional): pdf of phase function. Defaults to henyey_greenstein_pdf.
        num_samples (int, optional): Number of samples to take from env_map. Defaults to 64.

    Returns:
        torch.tensor: radiance (NHW,S,3), V_hat (NHW,S,3)
    """
    cos_theta = torch.sum(ray_dirs.view(-1,3) * sample_dirs.view(-1, 3), dim=-1)  # (...) dot product per row
    pdf = torch.nan_to_num(phase_function_pdf(cos_theta, g), nan=1e-6)  # (scalar) NOTE: possibly 1e-6 hardcoded is a bad idea (used to avoid NaN)
    L_env = eval_env_map(cos_theta, env_map) # environment lighting (...,S,3) RGB
    L = torch.mean(L_env * T * sigma_s / pdf, dim=0) # (...,3)
    return L, T


def multi_scatter(sh_coefs: torch.Tensor, sigma_s: float, ray_dir: torch.Tensor, num_samples=64):
    """
    Returns the approximate light contribution in the direction of omega_out (from light to eye)
    using the raw spherical harmonics predicted coefficients from SHNet.
    
    Args:
        sh_coefs (torch.Tensor): Outputs from SH MLP
        sigma_s (float): Scattering Coefficient from ScatterMLP
        current_dir (Tensor): current ray direction
        num_samples (int, optional): Number of samples used to evaluate radiance. Defaults to 64.

    Returns:
        Radiance estimated from Monte Carlo (3,) for RGB
    """
    sample_dirs = sample_next_directions(ray_dir, phase_function=sample_isotropic, num_samples=num_samples)
    L_sh = eval_sh(sh_coefs, sample_dirs, spherical=False)  # get radiance of light at point in the map
    return (L_sh * sigma_s / isotropic_pdf()).mean(dim=0)   # (..., 3); T is baked into the learned coefficients


def volume_render(
    raw: torch.tensor,
    points: torch.tensor,
    z_vals: torch.tensor,
    rays_d: torch.tensor,
    sample_d: torch.tensor,
    v_hats: torch.tensor,
    sh_coefs: torch.tensor,
    env_map: torch.tensor,
    raw_noise_std: Optional[float] = 0.0,
    white_background: Optional[bool] = False,
    batched: Optional[bool] = False
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Volume rendering based on the NeRF MLP's predictions `raw`.
    Here, we take the expectation of all samples along each ray to estimate
    the RGB color, weighted by opacity, @ that pixel location.

    Parameters
    ----------
    raw : torch.tensor
        ScatterMLP predictions. [sigma_t, sigma_s, g] ([N,] H, W, S, 3)
    points : torch.tensor
        3D coordinate points of each sample ([N,] H, W, S, 3)
    z_vals : torch.tensor
        Ray depths (from near plane) ([N,] H, W, S)
    rays_d : torch.tensor
        Ray directions. ([N,] H, W, 3)
    sample_d : torch.tensor
        The random samples (given HG phase function) per scatter point (samples in points)
    v_hats : torch.tensor
        VisibilityMLP outputs for predicting T of (x,dir) pair
    sh_coefs : torch.tensor
        SH coefficients for multi-scatter from SphericalHarmonicsMLP (9, 3)
    env_map : Optional[torch.tensor]
        Lighting from environment map. (H,W,3)
    raw_noise_std : Optional[float], optional
        Standard deviation of noise applied to `raw`, by default 0.0 (no noise)
    white_background : Optional[bool], optional
        Paint empty space white, by default False

    Returns
    -------
    Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]
        - RGB map         ([N,]H,W,3),
        - accumul. map    ([N,]H,W),
        - weights         ([N,]H,W,S), 
        - depth map       ([N,]H,W) 
    """
    if batched:
        batched_vr = torch.vmap(
            volume_render,
            in_dims=(0,0,0), 
            randomness='different')
        return batched_vr(
            raw, z_vals, rays_d, 
            raw_noise_std=raw_noise_std,
            white_background=white_background, batched=False)
    # Here, we already have our samples along each ray (H,W,S)
    # so we just take the difference of 'z' between the samples for transmittance (via sigma_t)
    delta_z = z_vals[..., 1:] - z_vals[..., :-1]  # distance between samples
    delta_z = torch.concat([
        delta_z,
        1e10 * torch.ones_like(delta_z[..., :1], device=z_vals.device)
    ], dim=-1) # add a long distance at the end for "infinity" (transmittance approaches 0)
    # also, keeps shape consistent with z_vals; eventually gets "rolled" out

    # multiply by the direction (NOT unit vectors) norm to get real 3D-scaled distances
    # in this way, rays extend like a rectangular pyramid instead of a spherical cone (using unit vec)
    # (say norm is sqrt(5); then for every z unit, the distance is sqrt(5) per 'z' traveled)
    delta_z = delta_z * torch.linalg.vector_norm(rays_d.unsqueeze(-2), dim=-1) # (H,W,S)
    noise = 0.0  # optional noise to add to density (following Gaussian)
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=z_vals.device) * raw_noise_std

    # predict "initial" transmittance for each sample
    sigma_t, sigma_s, g_phase_param = raw[..., :3]
    transmittance = torch.exp(-nn.functional.relu(sigma_t + noise) * delta_z) # (H,W,S)
    opacity = 1.0 - transmittance
    acc_T = torch.roll(torch.cumprod(transmittance + 1e-10, dim=-1), 1, dims=(-1))
    acc_T[..., 0] = 1.0  # transmittance always starts unfettered (1.0)
    weights = opacity * acc_T   # used for fine sampling on fine-pass

    # transmittance is already applied internally to the functions below
    # add the contributions from single and multi scattering cases
    Ls = single_scatter(env_map, points, rays_d, sigma_s)
    Lm = multi_scatter(sh_coefs, sigma_s, rays_d)
    L = Ls + Lm  # output shape: ([N,] H, W, S, 3)
    
    rgb_map = torch.sum(L * weights.unsqueeze(-1), dim=-2) # sum sample contributions ([N,] H, W, 3)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    # weight accumulation; [0,1]
    acc_map = torch.sum(weights, dim=-1)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map.unsqueeze(-1))

    return rgb_map, acc_map, weights, depth_map


def sample_fine(
    bins: torch.tensor,
    weights: torch.tensor,
    num_samples: int,
    perturb: bool=True,
    batched: bool=False
)-> torch.tensor:
    """
    Performs inverse transform sampling to sample from a PDF based on `weights`
    for the fine pass of the NeRF.

    Parameters
    ----------
    bins : torch.tensor
        Bins to stratified sample from. ([N,] H, W, S-1)
    weights : torch.tensor
        Weights from volume rendering (transmittance * opacity) ([N,] H, W, S-2)
    num_samples : int
        Number of samples to take.
        (NOTE: 'S' here refers to the coarse samples #, NOT this.)
    perturb : bool, optional
        Random uniform sample or equally spaced linear, by default True
    batched : bool, optional
        If data is batched. Then, add batch dimension to return tensors 'N'.

    Returns
    -------
    torch.tensor
        Samples ([N,] H, W, num_samples)
    """

    if batched:
        batched_sample_pdf = torch.vmap(
            sample_fine, 
            in_dims=(0,0), randomness='different')
        return batched_sample_pdf(
            bins, weights, 
            num_samples=num_samples, perturb=perturb, batched=False)

    H, W, S = bins.shape[-3:]
    S += 1  # for # of coarse samples, which bins (...,S-1) originates from.
    # Generating the PDF that samples more from regions of higher density (from 'weights')
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, dim=-1, keepdim=True) 
    cdf = torch.cumsum(pdf, dim=-1)  # CDF = integral(PDF)
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # (-inf, 0) domain = 0
    # CDF=(H,W,S-1) matching shape of 'bins'

    # uniform sampling part
    if perturb:
        u = torch.rand((H, W, num_samples), device=cdf.device)
    else:  # quadrature
        u = torch.linspace(0., 1., num_samples, device=cdf.device)
        u = u.expand(H, W, num_samples)

    u.contiguous() # sequential memory format for slicing/transposing/permuting

    # corresponding CDF 'section' that 'u' falls into (mostly, highest weighted sample range)
    inds = torch.searchsorted(cdf, u, right=True)

    # clamp out of bounds values
    below = torch.clamp(inds - 1, min=0) # gets 'floor' bin
    above = torch.clamp(inds, max=cdf.shape[-1] - 1) # gets 'ceiling' bin
    inds_range = torch.stack([below, above], dim=-1) # (H,W, num_samples, 2)

    # "for all ray samples, select the appropriate cdf/bin ranges using 'u'"
    cdf_range = torch.gather(
            cdf.unsqueeze(-2).expand(H, W, num_samples, S - 1),
            dim=-1, index=inds_range) # resulting CDF value bounds [u_lower, u_upper]
    bins_range = torch.gather(
            bins.unsqueeze(-2).expand(H, W, num_samples, S - 1),
            dim=-1, index=inds_range) # resulting distances (scalar) ALONG ray
    
    # final samples based on position within cdf and bin ranges
    lower_bound, upper_bound = cdf_range[..., 0], cdf_range[..., 1] # CDF bounds
    lower_bin, upper_bin = bins_range[..., 0], bins_range[..., 1]   # bin locations
    denom = upper_bound - lower_bound
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom) # to avoid div by 0
    t = (u - lower_bound) / denom # allows interpolation between upper and lower bounds [0,1)
    samples = lerp(t, lower_bin, upper_bin) # uses 't' above to sample
    
    return samples


def sample_hierarchical(
    rays_o: torch.tensor,
    rays_d: torch.tensor,
    z_vals: torch.tensor,
    weights: torch.tensor,
    num_samples: int,
    perturb: bool=True,
    batched: bool=False,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Performs hierarchical sampling (for NeRF fine-pass) for importance sampling
    along regions of likely high densities. Combines with previous coarse samples.

    Parameters
    ----------
    rays_o : torch.tensor
        Ray origins. ([N,] H, W, 3)
    rays_d : torch.tensor
        Ray directions. ([N,] H, W, 3)
    z_vals : torch.tensor
        Coarse sample distances from the origin based on vector norm. ([N,] H, W, S)
    weights : torch.tensor
        Weights of coarse samples along rays (given by volume rendering). ([N,] H, W, S)
    num_samples : int
        Number of samples to take from the distribution (histogram) given by weights.
        (See: `sample_fine`)
    perturb : bool, optional
        Generate samples randomly (otherwise, linearly), by default True
    batched : bool, optional
        If batched, then prefix shape with batch size 'N', by default False

    Returns
    -------
    torch.tensor
        - 3D sample points ([N,] H, W, S, 3)
        - z_value combined coarse & fine samples ([N,] H, W, S + S_fine)
        - z_value fine samples only ([N,] H, W, S_fine)
    """

    # inverse transform sampling
    bins = (z_vals[..., 1:] + z_vals[..., :-1]) / 2
    fine_z_samples = sample_fine(
        bins, weights[..., 1:-1],
        num_samples=num_samples, perturb=perturb, batched=batched) 
    fine_z_samples = fine_z_samples.detach()

    all_z_samples = torch.sort(torch.concat([z_vals, fine_z_samples], dim=-1), dim=-1).values
    points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * all_z_samples.unsqueeze(-1)
    # points = (H,W,1,3) + (H,W,1,3) * (H,W,num_samples,1)
    return points, all_z_samples, fine_z_samples
    

def nerf_forward(
    # from get_rays
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    # PE function
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    env_map: torch.tensor,
    coarse_nets: dict[str, nn.Module], # coarse network components
    fine_nets: dict[str, nn.Module] = None, # coarse + fine samples, fine network components
    kwargs_sample_stratified: dict = None,
    n_samples_hierarchical: int = 0,
    kwargs_sample_hierarchical: dict = None,
    viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    sh_level: int = 2,
    chunk_size: int = 2**15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    NeRF forward pass.

    Parameters
    ----------
    rays_o : torch.Tensor
        Ray origins. ([N,] H, W, 3)
    rays_d : torch.Tensor
        Ray directions. ([N,] H, W, 3)
    near : float
        Near plane.
    far : float
        Far plane.
    encoding_fn : Callable[[torch.Tensor], torch.Tensor]
        Positional encoding function.
    coarse_model : dict[str, nn.Module)
        Coarse networks (get_rays only; no hierarchical sampling step)
        - feature_mlp
        - scatter_mlp
        - sh_mlp
        - visibility_mlp
    kwargs_sample_stratified : dict, optional
        Additional kwargs for specified internal function, by default None
    n_samples_hierarchical : int, optional
        Additional kwargs for specified internal function, by default 0
    kwargs_sample_hierarchical : dict, optional
        Additional kwargs for specified internal function, by default None
    fine_model : dict[str, nn.Module], optional
        Fine model (augments coarse model with hierarcihcal sampling step), by default None
        - (dict keys same as coarse model; NOTE: use same modules as the ones used in coarse)
    viewdirs_encoding_fn : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
        Positional encoding function for view directions, by default None
        NOTE: However, should be used!
    chunksize : int, optional
        Chunk size, by default 2**15

    Returns
    -------
    Tuple[dict]
        - key 'rgb_map'
        - key 'depth_map'
        - key 'acc_map'
        - key 'weights'
    """

    H, W = rays_o.shape[:2]
    S_coarse = kwargs_sample_stratified.get('num_samples', 64)
    S_fine   = n_samples_hierarchical

    kwargs_sample_stratified = kwargs_sample_stratified or {}
    kwargs_sample_hierarchical = kwargs_sample_hierarchical or {}
    viewdirs_encoding_fn = viewdirs_encoding_fn or encoding_fn
    outputs = {}    # returning a dict of values

    # query coarse points
    points, z_vals = sample_rays(rays_o, rays_d, near, far,
                                 **kwargs_sample_stratified)
    
    # chunkify (for memory) (WITHOUT APPLYING ENCODINGS, different from NeRF)
    pt_chunks, viewdir_chunks = prepare_chunks(
        points.reshape(H*W,-1,3), rays_d.reshape(H*W,-1,3),
        encode=False, chunk_size=chunk_size) # returns (HWS, encoding_dim) for both
    
    outputs['z_vals_stratified'] = z_vals
    
    # COARSE predictions
    raw_preds = {
        "sh_hat": [],
        "sc_hat": [],
        "v_hat": [],
    }

    bounce_dirs_total = []

    # prepare chunks; OUTPUT: (min([N]HW, chunk_size), 3)
    for pt_chunk, viewdir_chunk in zip(pt_chunks, viewdir_chunks):
        # model predictions per chunk (to be aggregated after)
        pt_chunk_enc = encoding_fn(pt_chunk.reshape(-1, pt_chunk.shape[-1]))
        F_out  = coarse_nets["feature_mlp"](pt_chunk_enc)
        sh_out = coarse_nets["sh_mlp"](F_out)
        sc_out = coarse_nets["scatter_mlp"](F_out)
        g = sc_out[..., 4]

        # for each query point, predict T for sampled 64 single-scatter directions
        bounce_dirs = sample_next_directions(
            viewdir_chunk, lambda u,v: sample_henyey_greenstein(u,v,g=g), num_samples=64)
        
        # ENCODE bounce_dirs
        bounce_dirs_enc = encoding_fn(bounce_dirs.reshape(-1, bounce_dirs.shape[-1]))
        
        # "query: at this point and direction, what's my T?"
        V_out = coarse_nets["visibility_mlp"](pt_chunk_enc, bounce_dirs_enc) 
        raw_preds["sh_hat"].append(sh_out)
        raw_preds["sc_hat"].append(sc_out)
        raw_preds["v_hat"].append(V_out)
        bounce_dirs_total.append(bounce_dirs) # (HW, S, 64) after loop + concat? NOTE: NEED TO CHECK THIS

    raw_sc = torch.concat(raw_preds['sc_hat'], dim=0).reshape(H*W, S_coarse, 5) # scatter coefs (st[1],ss[3],g[1])
    raw_sh = torch.concat(raw_preds["sh_hat"], dim=0).reshape(H*W, S_coarse, (2 ** sh_level) - 1) # sh_coefs
    raw_v  = torch.concat(raw_preds["v_hat"],  dim=0).reshape(H*W, S_coarse, 64) # single scatter Ts per scatter sample
    bounce_dirs_total = torch.concat(bounce_dirs_total, dim=0).reshape(H*W, S_coarse, 64) # (num_chunks, 64) NOTE: CHECK THIS

    # input all network outputs to volume_render
    rgb_map, acc_map, weights, depth_map = volume_render(
        raw_sc, points, z_vals, rays_d, bounce_dirs_total, raw_v, raw_sh, env_map)

    # FINE pass predictions; process is basically the same as coarse pass
    if n_samples_hierarchical > 0 and fine_nets is not None:
        # store coarse output maps
        rgb_map_coarse, depth_map_coarse, acc_map_coarse = rgb_map, depth_map, acc_map

        points, all_z, fine_z = sample_hierarchical(
            rays_o, rays_d, z_vals, weights,
            num_samples=n_samples_hierarchical, **kwargs_sample_hierarchical
        )

        pt_chunks, viewdir_chunks = prepare_chunks(
            points.reshape(H*W,-1,3), rays_d.reshape(H*W,-1,3),
            encode=False,
            chunk_size=chunk_size)
        
        raw_preds_fine = {
            "sh_hat": [],
            "sc_hat": [],
            "v_hat": [],
        }

        bounce_dirs_total_fine = []

        for pt_chunk, viewdir_chunk in zip(pt_chunks, viewdir_chunks):
            # model predictions per chunk (to be aggregated after)
            pt_chunk_enc = encoding_fn(pt_chunk.reshape(-1, pt_chunk.shape[-1]))
            F_out_fine  = fine_nets["feature_mlp"](pt_chunk_enc)
            sh_out_fine = fine_nets["sh_mlp"](F_out_fine)
            sc_out_fine = fine_nets["scatter_mlp"](F_out_fine)
            g_fine = sc_out[..., 4]

            # for each query point, predict T for sampled 64 single-scatter directions
            bounce_dirs_fine = sample_next_directions(
                viewdir_chunk, lambda u,v: sample_henyey_greenstein(u,v,g=g_fine), num_samples=64)
            
            # ENCODE bounce_dirs_fine
            bounce_dirs_fine_enc = encoding_fn(bounce_dirs_fine.reshape(-1, bounce_dirs_fine.shape[-1]))

            # "query: at this point and direction, what's my T?"
            V_out_fine = fine_nets["visibility_mlp"](pt_chunk_enc, bounce_dirs_fine_enc) 
            raw_preds_fine["sh_hat"].append(sh_out_fine)
            raw_preds_fine["sc_hat"].append(sc_out_fine)
            raw_preds_fine["v_hat"].append(V_out_fine)
            bounce_dirs_total_fine.append(bounce_dirs_fine) # (HW, S, 64) after loop + concat

        raw_sc_fine = torch.concat(raw_preds_fine['sc_hat'], dim=0).reshape(H*W, S_fine + S_coarse, 5) # scatter coefs (st[1],ss[3],g[1])
        raw_sh_fine = torch.concat(raw_preds_fine["sh_hat"], dim=0).reshape(H*W, S_fine + S_coarse, (2 ** sh_level) - 1) # sh_coefs
        raw_v_fine = torch.concat(raw_preds_fine["v_hat"],  dim=0).reshape(H*W, S_fine + S_coarse, 64) # single scatter Ts per scatter sample
        bounce_dirs_total_fine = torch.concat(bounce_dirs_total_fine, dim=0).reshape(H*W, S_fine + S_coarse, 64).permute(0,1,3,2) # (HW, S, 3, 64)
    
        rgb_map, acc_map, weights, depth_map = volume_render(
            raw_sc_fine, points, z_vals, rays_d, bounce_dirs_total_fine, raw_v_fine, raw_sh_fine, env_map)

        outputs['z_vals_hierarchical'] = fine_z
        outputs['rgb_map_coarse'] = rgb_map_coarse
        outputs['depth_map_coarse'] = depth_map_coarse
        outputs['acc_map_coarse'] = acc_map_coarse

    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    
    return outputs
