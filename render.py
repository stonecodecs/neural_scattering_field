## NeRF essentials beyond the model itself
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Tuple, Callable
from utils import prepare_chunks
from sh import project_to_sh


def lerp(
    t: Union[torch.tensor, float],
    near: Union[torch.tensor, float],
    far: Union[torch.tensor, float]
) -> Union[torch.tensor, float]:
    """Linear interpolation between near and far planes."""
    return near + (far - near) * t


def get_camera_poses(poses: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    Extracts camera origins and -z vectors from 4x4 extrinsic matrices.

    Args:
        poses (torch.tensor): Camera poses (N, 4, 4)

    Returns:
        torch.tensor: Camera origins (N, 3)
        torch.tensor: Camera directions (-Z unit vectors) (N, 3)
    """
    cam_origins = poses[..., 0:3, 3] # translation components of matrix
    cam_directions = -poses[..., 0:3, 2] # equivalent to -z vector [0,0,-1]
    return cam_origins, cam_directions


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


def single_scatter():
    """
    Returns the light contribution from importance sampling the 
    """
    # TODO
    pass


def multi_scatter(sh_coefs: torch.Tensor, sigma_s: float, g: float, num_samples=64):
    """
    Returns the approximate light contribution in the direction of omega_out (from light to eye)
    using the raw spherical harmonics predicted coefficients from SHNet
    """
    # TODO
    pass

def volume_render(
    raw: torch.tensor,
    z_vals: torch.tensor,
    rays_d: torch.tensor,
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
        Model predictions. [RGB, density] ([N,] H, W, S, 4)
    z_vals : torch.tensor
        Ray depths (from near plane) ([N,] H, W, S)
    rays_d : torch.tensor
        Ray directions. ([N,] H, W, 3)
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
    # TODO: add in single and multiple scattering here

    if batched:
        batched_vr = torch.vmap(
            volume_render,
            in_dims=(0,0,0), 
            randomness='different')
        return batched_vr(
            raw, z_vals, rays_d, 
            raw_noise_std=raw_noise_std,
            white_background=white_background, batched=False)

    delta_z = z_vals[..., 1:] - z_vals[..., :-1]  # distance between samples
    delta_z = torch.concat([
        delta_z,
        1e10 * torch.ones_like(delta_z[..., :1], device=z_vals.device)
    ], dim=-1) # add a long distance at the end for "infinity" (transmittance approaches 0)
    # also, keeps shape consistent with z_vals; eventually gets "rolled" out

    # multiply by the direction (NOT unit vectors) norm to get real 3D-scaled distances
    # in this way, rays extend like a rectangular pyramid instead of a sphere (if unit vec)
    # (say norm is sqrt(5); then for every z unit, the distance is sqrt(5) per 'z' traveled)
    delta_z = delta_z * torch.linalg.vector_norm(rays_d.unsqueeze(-2), dim=-1) # (H,W,S)
    noise = 0.0  # optional noise to add to density (following Gaussian)
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=z_vals.device) * raw_noise_std

    # predict density + transmittance for each sample
    rgb, density = raw[..., :3], raw[..., 3]    # TODO: change for msnerf (raw predicts more than just rgbc now)
    transmittance = torch.exp(-nn.functional.relu(density + noise) * delta_z) # (H,W,S)
    opacity = 1.0 - transmittance
    acc_T = torch.roll(torch.cumprod(transmittance + 1e-10, dim=-1), 1, dims=(-1))
    acc_T[..., 0] = 1.0  # transmittance always starts unfettered (1.0)
    weights = opacity * acc_T

    rgb = torch.sigmoid(rgb)
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2) # broadcast + take expectation over all samples
    depth_map = torch.sum(weights * z_vals, dim=-1)
    # disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, dim=-1))

    # weight accumulation; [0,1] where 0 is fully transparent, and 1 is fully absorbed
    acc_map = torch.sum(weights, dim=-1)

    # if white background
    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map.unsqueeze(-1))

    return rgb_map, acc_map, weights, depth_map


def sample_pdf(
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
            sample_pdf, 
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
        Number of samples to take from the distribution given by weights.
        (See: `sample_pdf`)
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
    fine_z_samples = sample_pdf(
        bins, weights[..., 1:-1],
        num_samples=num_samples, perturb=perturb, batched=batched) 
    fine_z_samples = fine_z_samples.detach()

    all_z_samples = torch.sort(torch.concat([z_vals, fine_z_samples], dim=-1), dim=-1).values
    points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * all_z_samples.unsqueeze(-1)
    # points = (H,W,1,3) + (H,W,1,3) * (H,W,num_samples,1)
    return points, all_z_samples, fine_z_samples
    

def nerf_forward(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  near: float,
  far: float,
  encoding_fn: Callable[[torch.Tensor], torch.Tensor],
  coarse_model: nn.Module,
  kwargs_sample_stratified: dict = None,
  n_samples_hierarchical: int = 0,
  kwargs_sample_hierarchical: dict = None,
  fine_model = None,
  viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
  chunk_size: int = 2**15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    NeRF forward pass. (So far, only takes non-batched data)

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
    coarse_model : nn.Module
        Coarse model (get_rays only; no hierarchical sampling step)
    kwargs_sample_stratified : dict, optional
        Additional kwargs for specified internal function, by default None
    n_samples_hierarchical : int, optional
        Additional kwargs for specified internal function, by default 0
    kwargs_sample_hierarchical : dict, optional
        Additional kwargs for specified internal function, by default None
    fine_model : _type_, optional
        Fine model (augments coarse model with hierarcihcal sampling step), by default None
    viewdirs_encoding_fn : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
        Positional encoding function for view directions, by default None
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
    # TODO: enable batching 
    H, W = rays_o.shape[:2]
    S_coarse = kwargs_sample_stratified.get('num_samples', 64)
    S_fine   = n_samples_hierarchical

    kwargs_sample_stratified = kwargs_sample_stratified or {}
    kwargs_sample_hierarchical = kwargs_sample_hierarchical or {}
    viewdirs_encoding_fn = viewdirs_encoding_fn or encoding_fn
    outputs = {}    # returning a dict of values

    # query points
    points, z_vals = sample_rays(rays_o, rays_d, near, far,
                                 **kwargs_sample_stratified)
    
    pt_chunks, viewdir_chunks = prepare_chunks(points.reshape(H*W,-1,3), rays_d.reshape(H*W,-1,3),
                                encoding_fn=encoding_fn, 
                                viewdirs_encoding_fn=viewdirs_encoding_fn,
                                chunk_size=chunk_size)
    # returns (HWS, encoding_dim) for both
    
    outputs['z_vals_stratified'] = z_vals
    
    # COARSE predictions
    raw_preds = []

    for pt_chunk, viewdir_chunk in zip(pt_chunks, viewdir_chunks):
        coarse_preds = coarse_model(pt_chunk, viewdir_chunk)
        raw_preds.append(coarse_preds)

    raw = torch.concat(raw_preds, dim=0).reshape(H, W, S_coarse, 4)
    # with torch.no_grad():
    #   print(f"RAW{raw.shape}:\nMEAN:{torch.mean(raw.reshape(-1,4), dim=0)}")

    rgb_map, acc_map, weights, depth_map = volume_render(raw, z_vals, rays_d)

    # FINE pass predictions; process is basically the same as coarse pass
    if n_samples_hierarchical > 0 and fine_model is not None:
        rgb_map_coarse, depth_map_coarse, acc_map_coarse = rgb_map, depth_map, acc_map

        points, all_z, fine_z = sample_hierarchical(
            rays_o, rays_d, z_vals, weights,
            num_samples=n_samples_hierarchical, **kwargs_sample_hierarchical
        )

        # prepare chunks again
        pt_chunks, viewdir_chunks = prepare_chunks(
            points.reshape(H*W,-1,3), rays_d.reshape(H*W,-1,3),
            encoding_fn=encoding_fn,
            viewdirs_encoding_fn=viewdirs_encoding_fn,
            chunk_size=chunk_size)
        
        raw_preds_fine = []
        for pt_chunk, viewdir_chunk in zip(pt_chunks, viewdir_chunks):
            fine_preds = fine_model(pt_chunk, viewdir_chunk)
            raw_preds_fine.append(fine_preds)

        raw_preds_fine = torch.concat(raw_preds_fine, dim=0)
        raw_preds_fine = raw_preds_fine.reshape(H, W, S_fine + S_coarse, 4)
        
        rgb_map, acc_map, weights, depth_map = volume_render(raw_preds_fine, all_z, rays_d)

        outputs['z_vals_hierarchical'] = fine_z
        outputs['rgb_map_coarse'] = rgb_map_coarse
        outputs['depth_map_coarse'] = depth_map_coarse
        outputs['acc_map_coarse'] = acc_map_coarse

    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    
    return outputs