from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.utils.misc import torch_compile
from nerfstudio.cameras.cameras import Cameras

# Taken from nerfstudio/models/splatfacto.py

@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat.to(R.device).to(torch.float32)


def get_intrinsics_matrix(frame):
    K = torch.zeros((3, 3), dtype=torch.float32)
    K[0,0] = frame["fl_x"]
    K[1,1] = frame["fl_y"]
    K[0,2] = frame["cx"]
    K[1,2] = frame["cy"]
    K[2,2] = 1
    return K


def adjust_intrinsic(K, W, H, crop_w, crop_h, new_w, new_h):
    # Step 1: Adjust for center crop
    K[0, 2] = K[0, 2] - (W - crop_w) / 2  # new cx
    K[1, 2] = K[1, 2] - (H - crop_h) / 2  # new cy

    # Step 2: Adjust for resize
    scale_w = new_w / crop_w
    scale_h = new_h / crop_h
    K[0, :] *= scale_w
    K[1, :] *= scale_h

    return K


def adjust_camera_intrinsics(camera: Cameras, new_w=512, new_h=512, crop_w=None, crop_h=None):
    """
    Adjusts the camera intrinsics after center cropping and resizing the image.

    Parameters:
    - camera: Camera object with intrinsics, width, and height attributes.
    - new_w: int, optional
        The new width after resizing.
    - new_h: int, optional
        The new height after resizing.
    - crop_w: int, optional
        The width of the center crop.
    - crop_h: int, optional
        The height of the center crop.
    """
    intrinsics = camera.get_intrinsics_matrices()[0]

    # Set crop size to camera's original dimensions if not provided
    crop_w = crop_w or camera.width.item()
    crop_h = crop_h or camera.height.item()

    # Adjust cx, cy for center crop
    intrinsics[0, 2] = intrinsics[0, 2].to(camera.device) - (camera.width - crop_w) / 2  # new cx
    intrinsics[1, 2] = intrinsics[1, 2].to(camera.device) - (camera.height - crop_h) / 2  # new cy

    # Adjust for resize 
    scale_w = new_w / crop_w
    scale_h = new_h / crop_h
    intrinsics[0, :] *= scale_w
    intrinsics[1, :] *= scale_h

    # Update the camera object with new parameters
    camera.cx = torch.as_tensor(intrinsics[0, 2]).view(1, 1).to(camera.device)
    camera.cy = torch.as_tensor(intrinsics[1, 2]).view(1, 1).to(camera.device)
    camera.fx = torch.as_tensor(intrinsics[0, 0]).view(1, 1).to(camera.device)
    camera.fy = torch.as_tensor(intrinsics[1, 1]).view(1, 1).to(camera.device)
    camera.width = torch.as_tensor(new_w).view(1, 1).to(camera.device)
    camera.height = torch.as_tensor(new_h).view(1, 1).to(camera.device)


def unproject_image_to_3d(intrinsics, depth, Rt):
    """
    Unprojects all pixels in an image to 3D space using known camera intrinsics, depth map, and extrinsics.

    Parameters:
    - intrinsics: torch.Tensor (3x3) 
        The camera intrinsic matrix.
    - depth: torch.Tensor (H, W) 
        Depth values for each pixel in the image.
    - Rt: torch.Tensor (4x4) 
        The rotation and traslation matrix from world to camera coordinates.

    Returns:
    - torch.Tensor (H, W, 3): The 3D world coordinates for each pixel.
    """

    H, W = depth.shape  # Image height and width

    # Step 1: Create a grid of (u, v) coordinates for each pixel in the image
    u_coords, v_coords = torch.meshgrid(torch.arange(W, device=depth.device), torch.arange(H, device=depth.device), indexing='xy')  # Shape (W, H)

    # Step 2: Flatten the grid into a (N, 2) tensor, where N = H * W
    pixel_coords = torch.stack([u_coords, v_coords], dim=-1).reshape(-1, 2).float()  # Shape (N, 2)

    # Step 3: Flatten the depth map to match pixel coordinates
    depths = depth.reshape(-1)  # Shape (N)

    # Step 4: Inverse of the intrinsic matrix (K^-1)
    K_inv = torch.inverse(intrinsics)

    # Step 5: Convert the pixel coordinates to homogeneous form (u, v, 1)
    ones = torch.ones((pixel_coords.shape[0], 1), dtype=torch.float32, device=pixel_coords.device)
    pixel_coords_homogeneous = torch.cat([pixel_coords, ones], dim=1)  # Shape (N, 3)

    # Step 6: Get the 3D camera coordinates (X_c, Y_c, Z_c)
    camera_coords = (K_inv @ pixel_coords_homogeneous.T).T  # Shape (N, 3)

    # Step 7: Scale by depth values (Z_c)
    camera_coords = camera_coords * depths.view(-1, 1)  # Shape (N, 3)

    # Step 8: Convert 3D camera coordinates to homogeneous form by appending a 1
    camera_coords_homogeneous = torch.cat([camera_coords, ones], dim=1)  # Shape (N, 4)

    # Step 9: Compute the inverse of extrinsic matrix (R | t)
    Rt_inv = torch.inverse(Rt)  # Inverse extrinsics

    # Step 10: Transform from camera coordinates to world coordinates
    world_coords_homogeneous = (Rt_inv @ camera_coords_homogeneous.T).T  # Shape (N, 4)

    # Step 11: Extract the world coordinates (X_w, Y_w, Z_w)
    world_coords = world_coords_homogeneous[:, :3]  # Shape (N, 3)

    # Step 12: Reshape the world coordinates back to (H, W, 3)
    world_coords = world_coords.reshape(H, W, 3)

    return world_coords


def project_3d_to_2d(intrinsics, points_3d, Rt):
    """
    Projects 3D points in the world coordinates to 2D pixel coordinates in the camera frame using a combined Rt matrix.

    Parameters:
    - intrinsics: torch.Tensor (3x3)
        The camera intrinsic matrix.
    - points_3d: torch.Tensor (N, 3)
        The 3D world coordinates of the points to be projected.
    - Rt: torch.Tensor (4x4)
        The combined rotation and translation matrix from world to camera coordinates (extrinsics).

    Returns:
    - torch.Tensor (N, 2): The 2D pixel coordinates of the projected points.
    """

    N = points_3d.shape[0]  # Number of 3D points

    # Step 1: Convert the 3D points to homogeneous coordinates (append 1 to each point)
    ones = torch.ones((N, 1), dtype=torch.float32, device=Rt.device)  # Shape (N, 1)
    points_3d_homogeneous = torch.cat([points_3d, ones], dim=-1)  # Shape (N, 4)

    # Step 2: Transform 3D points from world coordinates to camera coordinates using Rt
    points_3d_camera_homogeneous = (Rt @ points_3d_homogeneous.T).T  # Shape (N, 4)

    # Step 3: Extract the x, y, z coordinates from the transformed points
    x, y, z = points_3d_camera_homogeneous[:, 0], points_3d_camera_homogeneous[:, 1], points_3d_camera_homogeneous[:, 2]

    # Step 4: Project the 3D points onto the 2D image plane using the intrinsic matrix
    u = (intrinsics[0, 0] * x + intrinsics[0, 2] * z) / z  # u = f_x * X_c / Z_c + c_x
    v = (intrinsics[1, 1] * y + intrinsics[1, 2] * z) / z  # v = f_y * Y_c / Z_c + c_y

    # Step 5: Stack the projected u and v coordinates to form the 2D pixel coordinates
    pixel_coords = torch.stack([u, v], dim=-1)  # Shape (N, 2)

    return pixel_coords


def warp_to_new_view(src_frame_intrinsics, src_Rt, depth_map, target_frame_intrinsics, target_Rt):
    """
    Warps an image from one view to another using the depth map and known camera parameters.

    Parameters:
    - src_frame_intrinsics: torch.Tensor (3x3)
        The camera intrinsic matrix for the source view.
    - depth_map: torch.Tensor (H, W)
        Depth values for each pixel in the source image.
    - src_Rt: torch.Tensor (4x4)
        The rotation translation matrix from world to source camera coordinates.
    - target_frame_intrinsics: torch.Tensor (3x3)
        The camera intrinsic matrix for the target view.
    - target_Rt: torch.Tensor (4x4)
        The rotation translation matrix from world to target camera coordinates.

    Returns:
    - torch.Tensor (H, W, 2): The 2D pixel coordinates in the target view for each pixel in the source view.
    """

    # Step 1: Unproject the source view to 3D world coordinates
    world_coords = unproject_image_to_3d(src_frame_intrinsics, depth_map, src_Rt)  # Shape (H, W, 3)
    H, W, _ = world_coords.shape

    # Step 2: Flatten the world coordinates to shape (N, 3), where N = H * W
    world_coords_flat = world_coords.reshape(-1, 3)  # Shape (N, 3)

    # Step 3: Project the 3D world points to the 2D image plane of the target view
    target_pixel_coords_flat = project_3d_to_2d(target_frame_intrinsics, world_coords_flat, target_Rt)  # Shape (N, 2)

    # Step 4: Reshape the 2D pixel coordinates back to (H, W, 2)
    target_pixel_coords = target_pixel_coords_flat.reshape(H, W, 2)

    return target_pixel_coords


def warp_image(image1, warped_pixel_coords):
    """
    Warps image1 to the new view using warped pixel coordinates.

    Parameters:
    - image1: torch.Tensor (B, C, H, W)
        The source image tensor with shape (batch, channels, height, width).
    - warped_pixel_coords: torch.Tensor (H, W, 2)
        The warped pixel coordinates tensor that maps image1 to image2.

    Returns:
    - torch.Tensor: The warped image with the same shape as image1.
    - torch.Tensor: The non-seen mask indicating which pixels are valid.
    """

    B, C, H, W = image1.shape  # Batch size, channels, height, width

    # Step 1: Normalize the pixel coordinates to the range [-1, 1] as required by grid_sample
    warped_pixel_coords_norm = torch.empty_like(warped_pixel_coords)
    warped_pixel_coords_norm[..., 0] = (warped_pixel_coords[..., 0] / (W - 1)) * 2 - 1  # Normalize x to [-1, 1]
    warped_pixel_coords_norm[..., 1] = (warped_pixel_coords[..., 1] / (H - 1)) * 2 - 1  # Normalize y to [-1, 1]

    # Step 2: Rearrange the coordinates from (H, W, 2) to (B, H, W, 2) for grid_sample
    warped_pixel_coords_norm = warped_pixel_coords_norm.unsqueeze(0).expand(B, -1, -1, -1)  # Shape (B, H, W, 2)

    # Step 3: Use grid_sample to warp the image
    warped_image = F.grid_sample(image1, warped_pixel_coords_norm, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Step 4: Create a mask for non-seen pixels
    # Check which pixel coordinates are within the bounds of the original image
    valid_mask = (warped_pixel_coords[..., 0] >= 0) & (warped_pixel_coords[..., 0] < W) & \
                 (warped_pixel_coords[..., 1] >= 0) & (warped_pixel_coords[..., 1] < H)

    # Convert the mask to the same shape as warped_image (B, H, W)
    non_seen_mask = (~valid_mask).to(image1.dtype).unsqueeze(0).expand(B, H, W)  # Invert the mask to get non-seen pixels

    return warped_image, non_seen_mask


def compute_mask_and_difference(dst_frame_intrinsics_adjusted, dst_frame_viewmat, depth_map, 
                                src_frame_intrinsics_adjusted, src_frame_viewmat, 
                                src_image, dst_image):
    warped_pixel_coords = warp_to_new_view(dst_frame_intrinsics_adjusted[0], dst_frame_viewmat[0], 
                                           depth_map, src_frame_intrinsics_adjusted[0], src_frame_viewmat[0])
    warped_image, non_seen_mask = warp_image(src_image, warped_pixel_coords)
    mask = 1 - non_seen_mask.squeeze()
    diff_dst_and_warped = ((warped_image - dst_image) * mask).abs().mean(dim=1, keepdim=True)
    diff_dst_and_warped_thresholded = (diff_dst_and_warped > 0.2).float()
    mask = (1 - diff_dst_and_warped_thresholded.squeeze()) * mask
    
    return mask