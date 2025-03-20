## miscellaneous functions
import torch
import numpy as np
from typing import List, Dict, Union, Callable
import matplotlib.pyplot as plt
from PIL import Image
import json
import os


def measure_PSNR(image, ref, maxval=255):
    """
    Measures PSNR between a given 'source' image and GT 'reference'

    Args:
        source (image): some tensor of shape (H,W,C)
        ref (image): some tensor of shape (H,W,C)
        maxval(float): maximum value of image (default 255)
    """
    mse = measure_MSE(ref, image)
    return 10.0 * torch.log10((maxval ** 2) / mse)


def measure_MSE(x, y):
    """ Mean Squared Error """
    return torch.sum(torch.sqrt((x - y) ** 2))


def compute_focal_length(fovx, img_shape, is_aniso=False):
    """
    Computes focal length from camera_angle_x in transforms_{}.json 
    If is_aniso is True (used when fovx!=fovy), then 2nd parameter is not None.
    Returns focal_x, focal_y.
    """
    H, W = img_shape
    aspect_ratio = H / W
    focal_x = (W / 2) / np.tan(fovx / 2)
    focal_y = None
    if is_aniso:
        fovy = 2 * np.arctan2(np.tan(fovx / 2) * aspect_ratio)
        focal_y = (H / 2) / np.tan(fovy / 2)
    return focal_x, focal_y


def get_chunks(
        inputs: torch.tensor,
        chunk_size: int=2**15
) -> List[torch.tensor]:
    """Splits `input` (num_rays) into chunks."""
    return [inputs[i:i + chunk_size] for i in range(0, inputs.shape[0], chunk_size)]


def prepare_chunks(
    points: torch.tensor,
    rays_d: torch.tensor,
    encoding_fn: Callable[[torch.tensor], torch.tensor]=None,
    viewdirs_encoding_fn: Callable[[torch.tensor], torch.tensor]=None,
    encode=False,
    chunk_size: int=2**15
) -> List[torch.tensor]:
    """Positional encoding + chunking step"""

    points_enc = encoding_fn(points) # (HW,S,points_L)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True) # (HWS,3) unit vec
    viewdirs = viewdirs.expand(points.shape)
    if encode:
        viewdirs = viewdirs_encoding_fn(viewdirs) # (HW,S,enc_L)
        points_enc = get_chunks(points_enc.reshape(-1, points_enc.shape[-1]), chunk_size=chunk_size)
        viewdirs_enc = get_chunks(viewdirs_enc.reshape(-1, viewdirs_enc.shape[-1]), chunk_size=chunk_size)
        return points_enc, viewdirs_enc  # (HW,S,points_L), (HW,S,enc_L)
    
    points_ = get_chunks(points.reshape(-1, points.shape[-1]), chunk_size=chunk_size)
    viewdirs_ = get_chunks(viewdirs.reshape(-1, viewdirs.shape[-1]), chunk_size=chunk_size)
    return points_, viewdirs_


def quaternion_to_rotation_matrix(q):
  """ Converts quaternion into a 3x3 rotation matrix. """
  qx, qy, qz, qw = q
  return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
  ])

def spherical_to_cartesian(theta_phi):
    """
    From theta, phi in spherical coordinates, convert to a Cartesian 3D directional vector.

    """
    theta = theta_phi[..., 0]
    phi = theta_phi[..., 1]
    
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    
    return torch.stack([x, y, z], dim=-1)
    # theta, phi = torch.as_tensor(theta), torch.as_tensor(phi)
    # x = (torch.sin(theta) * torch.cos(phi)).view(1, -1)
    # y = (torch.sin(theta) * torch.sin(phi)).view(1, -1)
    # z = (torch.cos(theta)).view(1, -1)
    # return torch.vstack((x,y,z))


def cartesian_to_spherical(xyz):
    x, y, z = xyz.unbind(-1)
    rho = torch.norm(xyz, dim=-1, keepdim=False)
    theta = torch.acos(z / rho)
    phi = torch.atan2(y, x)

    return torch.stack([theta, phi], dim=-1)
    # rho = np.sqrt(x**2 + y**2 + z**2)
    # theta = np.arccos(z / rho)
    # phi = np.arctan2(y, x)
    # return theta, phi


def local_to_world_rotation(target_dir):
    """
    Constructs a rotation matrix to convert directions from a local coordinate system 
    (where +z is aligned with `target_dir`) to world coordinates.
    
    Args:
        target_dir (torch.tensor): (N, 3) The direction vector in world coordinates (e.g., [a, b, c]).
    
    Returns:
        R (np.ndarray): (N,3,3) Rotation matrix.
    """
    target_dir = target_dir / torch.norm(target_dir, dim=-1, keepdim=True) # make sure unitvec
    
    temp_up = torch.zeros_like(target_dir, dtype=torch.float32)
    temp_up[..., 2] = 1.0

    tangent = torch.where(
        (torch.abs(target_dir[..., 2]) > 0.9).unsqueeze(-1),
        torch.cross(target_dir, temp_up, dim=-1),
        torch.cross(target_dir, temp_up, dim=-1)
    )
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True)
    
    # Compute bitangent and normalize
    bitangent = torch.cross(target_dir, tangent, dim=-1)
    bitangent = bitangent / torch.norm(bitangent, dim=-1, keepdim=True)
    
    # Stack to create rotation matrices
    return torch.stack([tangent, bitangent, target_dir], dim=-1).type(dtype=torch.float32)


def camera_extrinsics(q, t):
  """
  Given quaternion q and translation t, produces a 4x4 matrix representing
  camera extrinsics.
  """
  R = quaternion_to_rotation_matrix(q)
  mat = np.eye(4) # 4x4
  mat[:3, :3] = R
  mat[:3, 3]  = t
  return mat


def get_images_from_folder(path, scale_to=None):
    # scale_to resizes to the desired shape if not None
    images = []
    for filename in os.listdir(path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # allowed formats
            img = Image.open(os.path.join(path, filename))
            if scale_to is not None:
              img = img.resize(size=scale_to)
            img_array = np.array(img)

            images.append(img_array)
    return np.array(images)


def load_transforms_json(json_path: str, filename: str):
    """
    Load transforms_train.json data into PyTorch tensors.
    Follows Instant-NGP json standard (includes all camera intrinsics)

    Args:
        json_path (str): File path to the transforms_train.json file.
        filename (str): name of transforms file.

    Returns:
        dict: A dictionary containing:
            - 'poses' (torch.Tensor): Camera extrinsic matrices [N, 4, 4]
            - 'focal' (torch.Tensor): Focal length(s) [N]
    """
    cwd = os.getcwd()
    os.chdir(json_path)
    with open(filename, 'r') as f:
        data = json.load(f)

    poses = []
    # first_img_path = data["frames"][0]["file_path"]
    # if first_img_path.endswith(".png"):
    #     first_img = Image.open(os.path.join(first_img_path.replace('\\', '/')))
    # else:
    #     first_img = Image.open(os.path.join(first_img_path.replace('\\', '/') + ".png"))
    # W, H = first_img.size
    # focal_x, focal_y = compute_focal_length(data['camera_angle_x'], (H,W)) # (1,)
    focal_x, focal_y = data["fl_x"], data["fl_y"]
    for frame in data['frames']:
        transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32)  # (4,4)
        poses.append(transform_matrix)

    poses = torch.stack(poses)  # [N, 4, 4]
    os.chdir(cwd)   # change back directory
    return {'poses': poses, 'focal': (focal_x, focal_y)}


def retrieve_sfm_data(camera_path: str, images_path: str, multi_cam=False):
    """
    Gets camera poses from **COLMAP SfM** reconstruction files.
    If data is from transforms_{}.json, use `TBD` instead.
    If multi_cam is true, images consisted of using more than a single camera.
    (NOT IMPLEMENTED).
    Returns (N,4,4) pose matrices, and focal length (as a single value OR dictionary).
    """

    focal_len = {} if multi_cam else 0
    poses = []

    with open(camera_path, 'r') as f:
      for line in f:
          line = line.strip()
          if line.startswith("#"):
            continue
          # ID, model, W, H, focal, cx, cy, k1 (distortion coef) 
          tokens = line.split()
          id_, model_, W, H, focal, cx, cy, k = tokens # focal should be the same
    
    with open(images_path, 'r') as f:
      for line in f:
          line = line.strip()
          if line.startswith("#") or not line.endswith(".jpg"):
            continue
          # ID, qw,qx,qy,qz,tx,ty,tz, cam_id, name
          tokens = line.split()
          id_, qw, qx, qy, qz, tx, ty, tz, cam_id, name = tokens
          poses.append(camera_extrinsics(
            np.array([float(qx), float(qy), float(qz), float(qw)]),
            np.array([float(tx), float(ty), float(tz)]),
          ))

    focal_len = float(focal)
    return poses, focal_len


def ray_plot(origins, directions) -> None:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')

    # Compute axis limits to ensure equal scaling
    max_origin = torch.max(torch.abs(origins)).item()
    max_limit = max(max_origin, 1.0)  # Ensure at least [-1, 1] range if origins are near zero

    ax.set_xlim(-max_limit, max_limit)
    ax.set_ylim(-max_limit, max_limit)
    ax.set_zlim(-max_limit, max_limit)

    ax.quiver(
        origins[..., 0].flatten(),
        origins[..., 1].flatten(),
        origins[..., 2].flatten(),
        directions[..., 0].flatten(),
        directions[..., 1].flatten(),
        directions[..., 2].flatten(),
        length=0.5,
        normalize=False
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
