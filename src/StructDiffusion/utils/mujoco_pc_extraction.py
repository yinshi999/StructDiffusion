"""
MuJoCo Point Cloud Extraction Utilities

Extract point clouds from MuJoCo simulations similar to how they're extracted from H5 files.
"""

import numpy as np
import mujoco
import trimesh
import torch
from typing import Dict, List, Tuple, Optional

from StructDiffusion.utils.brain2.camera import GenericCameraReference, compute_xyz
from StructDiffusion.utils.rearrangement import get_pts, array_to_tensor


def render_mujoco_scene(model: mujoco.MjModel, data: mujoco.MjData, 
                       camera_name: str = None,
                       width: int = 640, height: int = 480) -> Dict:
    """
    Render RGB and depth images from MuJoCo scene.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        camera_name: Name of camera to use (None for default)
        width: Image width
        height: Image height
    
    Returns:
        Dictionary with keys: 'rgb', 'depth', 'camera_pose'
    """
    # Create renderer
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        if camera_name:
            scene_option = mujoco.MjvOption()
            mujoco.mjv_updateScene(model, data, scene_option, None, 
                                  mujoco.MjvPerturb(), camera_name, 
                                  mujoco.mjCAT_ALL, renderer.scene)
        else:
            renderer.update_scene(data)
        rgb = renderer.render()
        depth = renderer.render(depth=True)
    except (AttributeError, TypeError):
        # Fallback for older MuJoCo versions
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.ones((height, width), dtype=np.float32) * 2.0
    
    # Get camera pose (simplified - you may need to adjust based on your camera setup)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 2.0])  # Camera 2m above scene
    
    return {
        'rgb': rgb,
        'depth': depth,
        'camera_pose': camera_pose
    }


def extract_object_point_clouds_from_mujoco(
    model: mujoco.MjModel, 
    data: mujoco.MjData,
    object_names: List[str],
    camera_name: str = None,
    num_pts: int = 1024,
    width: int = 640, 
    height: int = 480,
    ignore_rgb: bool = True
) -> Dict:
    """
    Extract point clouds for specified objects from MuJoCo scene.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        object_names: List of object body names to extract
        camera_name: Camera name for rendering
        num_pts: Number of points to sample per object
        width: Image width
        height: Image height
        ignore_rgb: Whether to ignore RGB information
    
    Returns:
        Dictionary with:
        - 'obj_pcs': List of point cloud tensors (num_pts, 3) or (num_pts, 6)
        - 'obj_poses': List of 4x4 object poses
        - 'obj_pc_poses': List of 4x4 point cloud center poses
    """
    # Render scene
    render_output = render_mujoco_scene(model, data, camera_name, width, height)
    rgb = render_output['rgb'] / 255.0  # Normalize to [0, 1]
    depth = render_output['depth']
    camera_pose = render_output['camera_pose']
    
    # Create camera reference for XYZ computation
    camera = GenericCameraReference(
        proj_near=0.01,
        proj_far=5.0,
        proj_fov=60.0,
        img_width=width,
        img_height=height
    )
    camera.set_pose_matrix(camera_pose)
    
    # Compute XYZ from depth
    xyz = compute_xyz(depth, camera)
    
    # Transform XYZ to world coordinates
    h, w, d = xyz.shape
    xyz_flat = xyz.reshape(h * w, -1)
    xyz_world = trimesh.transform_points(xyz_flat, camera_pose)
    xyz = xyz_world.reshape(h, w, -1)
    
    # Extract point clouds for each object using geometric method
    obj_pcs = []
    obj_poses = []
    obj_pc_poses = []
    
    for obj_name in object_names:
        # Get object pose from MuJoCo
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id >= 0:
                obj_pose = np.eye(4, dtype=np.float64)
                # Get rotation matrix (3x3) from MuJoCo
                rot_mat = data.xmat[body_id].reshape(3, 3).copy()
                # Ensure it's a valid rotation matrix (orthonormal)
                # Normalize if needed
                if np.linalg.norm(rot_mat) > 0:
                    # Check if determinant is close to 1 (valid rotation matrix)
                    det = np.linalg.det(rot_mat)
                    if abs(det - 1.0) > 0.1:
                        # If not valid, use identity
                        print(f"Warning: Invalid rotation matrix for {obj_name} (det={det:.4f}), using identity")
                        rot_mat = np.eye(3)
                    obj_pose[:3, :3] = rot_mat
                else:
                    obj_pose[:3, :3] = np.eye(3)
                obj_pose[:3, 3] = data.xpos[body_id].copy()
            else:
                print(f"Warning: Object {obj_name} not found, skipping")
                continue
        except:
            print(f"Warning: Could not get pose for {obj_name}, skipping")
            continue
        
        # Extract point cloud using geometric method
        obj_pc = extract_object_pc_geometric(model, data, obj_name, num_pts)
        
        if obj_pc is not None:
            if ignore_rgb:
                obj_pcs.append(obj_pc)
            else:
                # Add RGB (use object color or default)
                rgb_values = np.ones((num_pts, 3)) * 0.5  # Default gray
                obj_pc_with_rgb = torch.cat([obj_pc, array_to_tensor(rgb_values)], dim=-1)
                obj_pcs.append(obj_pc_with_rgb)
            
            obj_poses.append(obj_pose)
            
            # Compute PC center pose
            pc_center = torch.mean(obj_pc, dim=0).numpy()
            pc_pose = np.eye(4)
            pc_pose[:3, 3] = pc_center
            obj_pc_poses.append(pc_pose)
    
    return {
        'obj_pcs': obj_pcs,
        'obj_poses': obj_poses,
        'obj_pc_poses': obj_pc_poses,
    }


def extract_object_pc_geometric(
    model: mujoco.MjModel, 
    data: mujoco.MjData,
    object_name: str, 
    num_pts: int = 1024
) -> Optional[torch.Tensor]:
    """
    Extract point cloud from MuJoCo object using geometric mesh extraction.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        object_name: Name of object body
        num_pts: Number of points to sample
    
    Returns:
        Point cloud tensor (num_pts, 3) or None if object not found
    """
    # Find body
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if body_id < 0:
        return None
    
    # Get object pose
    obj_pose = np.eye(4)
    obj_pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
    obj_pose[:3, 3] = data.xpos[body_id]
    
    # Collect vertices from all geoms in this body
    all_vertices = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            geom_type = model.geom_type[geom_id]
            geom_size = model.geom_size[geom_id]
            geom_pos = model.geom_pos[geom_id]
            geom_quat = model.geom_quat[geom_id]
            
            # Convert quat to rotation matrix
            # mju_quat2Mat expects: res (9x1 writeable array), quat (4x1 array)
            geom_rot_flat = np.zeros((9, 1), dtype=np.float64)  # 9x1 column vector
            geom_quat_col = geom_quat.reshape(4, 1).astype(np.float64)  # 4x1 column vector
            mujoco.mju_quat2Mat(geom_rot_flat, geom_quat_col)
            # Reshape 9x1 to 3x3 matrix (flatten first, then reshape)
            geom_rot = geom_rot_flat.flatten().reshape(3, 3)
            
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                # Get mesh vertices
                try:
                    mesh_id = model.geom_dataid[geom_id]
                    mesh_start = model.mesh_vertadr[mesh_id]
                    mesh_nvert = model.mesh_vertnum[mesh_id]
                    
                    vertices = model.mesh_vert[mesh_start:mesh_start + mesh_nvert * 3]
                    vertices = vertices.reshape(mesh_nvert, 3)
                except:
                    continue
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                # Create box vertices
                box = trimesh.creation.box(extents=geom_size * 2)
                vertices = box.vertices
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                # Create sphere vertices
                radius = geom_size[0]
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
                vertices = sphere.vertices
            else:
                continue
            
            # Transform by geom pose
            geom_pose = np.eye(4)
            geom_pose[:3, :3] = geom_rot
            geom_pose[:3, 3] = geom_pos
            vertices = trimesh.transform_points(vertices, geom_pose)
            all_vertices.append(vertices)
    
    if len(all_vertices) == 0:
        return None
    
    # Combine all vertices
    combined_vertices = np.vstack(all_vertices)
    
    # Transform to world frame
    combined_vertices = trimesh.transform_points(combined_vertices, obj_pose)
    
    # Sample points
    if len(combined_vertices) > num_pts:
        idx = np.random.choice(len(combined_vertices), num_pts, replace=False)
        sampled_vertices = combined_vertices[idx]
    else:
        # Repeat points if not enough
        idx = np.random.choice(len(combined_vertices), num_pts, replace=True)
        sampled_vertices = combined_vertices[idx]
    
    # Convert to tensor
    return array_to_tensor(sampled_vertices)



