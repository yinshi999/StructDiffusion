"""
MuJoCo to StructDiffusion Format Converter

Convert MuJoCo simulation data to the format expected by SemanticArrangementDataset.
"""

import numpy as np
import torch
from typing import Dict, List, Optional

from StructDiffusion.utils.mujoco_pc_extraction import extract_object_point_clouds_from_mujoco
from StructDiffusion.utils.transformations import euler_matrix
import mujoco


def safe_inv(pose: np.ndarray) -> np.ndarray:
    """
    Safely compute inverse of a 4x4 transformation matrix.
    Returns identity if matrix is singular.
    """
    try:
        det = np.linalg.det(pose[:3, :3])  # Check rotation part
        if abs(det) < 1e-6:
            print(f"Warning: Singular matrix (det={det:.6f}), using identity")
            return np.eye(4)
        return np.linalg.inv(pose)
    except np.linalg.LinAlgError:
        print("Warning: LinAlgError during inversion, using identity")
        return np.eye(4)


def get_mujoco_raw_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_object_names: List[str],
    goal_specification: Dict,
    other_object_names: List[str] = None,
    camera_name: str = None,
    num_pts: int = 1024,
    max_num_target_objects: int = 11,
    max_num_distractor_objects: int = 5,
    use_virtual_structure_frame: bool = True,
    ignore_distractor_objects: bool = True,
    ignore_rgb: bool = True,
    width: int = 640,
    height: int = 480
) -> Dict:
    """
    Convert MuJoCo scene to StructDiffusion input format (same as get_raw_data).
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_object_names: List of target object names to rearrange
        goal_specification: Dictionary with structure parameters:
            - 'type': 'circle', 'line', 'tower', 'dinner'
            - 'position': [x, y, z]
            - 'rotation': [rx, ry, rz] (Euler angles)
            - 'radius': float (for circle)
            - 'length': float (for line)
        other_object_names: List of other (distractor/anchor) object names
        camera_name: Camera name for rendering
        num_pts: Number of points per object
        max_num_target_objects: Maximum number of target objects
        max_num_distractor_objects: Maximum number of distractor objects
        use_virtual_structure_frame: Whether to use virtual structure frame
        ignore_distractor_objects: Whether to ignore distractor objects
        ignore_rgb: Whether to ignore RGB information
        width: Image width
        height: Image height
    
    Returns:
        Dictionary in format expected by SemanticArrangementDataset.get_raw_data():
        - 'pcs': List of point cloud tensors
        - 'sentence': List of (value, token_type) tuples
        - 'goal_poses': List of 4x4 goal pose matrices
        - 'type_index': List of type indices
        - 'position_index': List of position indices
        - 'pad_mask': List of padding masks
        - 't': timestep (set to 0)
    """
    if other_object_names is None:
        other_object_names = []
    
    # Extract point clouds
    all_object_names = target_object_names + other_object_names
    pc_data = extract_object_point_clouds_from_mujoco(
        model, data, all_object_names, camera_name, num_pts, width, height, ignore_rgb
    )
    
    obj_pcs = pc_data['obj_pcs']
    obj_poses = pc_data['obj_poses']
    obj_pc_poses = pc_data['obj_pc_poses']
    
    # Split into target and other objects
    target_pcs = obj_pcs[:len(target_object_names)]
    target_poses = obj_poses[:len(target_object_names)]
    target_pc_poses = obj_pc_poses[:len(target_object_names)]
    
    other_pcs = obj_pcs[len(target_object_names):] if not ignore_distractor_objects else []
    other_poses = obj_poses[len(target_object_names):] if not ignore_distractor_objects else []
    other_pc_poses = obj_pc_poses[len(target_object_names):] if not ignore_distractor_objects else []
    
    # Compute goal poses
    if use_virtual_structure_frame:
        goal_structure_pose = euler_matrix(
            goal_specification["rotation"][0],
            goal_specification["rotation"][1],
            goal_specification["rotation"][2]
        )
        goal_structure_pose[:3, 3] = goal_specification["position"]
        goal_structure_pose_inv = np.linalg.inv(goal_structure_pose)
    else:
        goal_structure_pose = None
        goal_structure_pose_inv = None
    
    # Compute goal poses for target objects based on structure
    goal_pc_poses = compute_goal_poses_from_structure(
        goal_specification, target_object_names, target_poses, target_pc_poses,
        use_virtual_structure_frame, goal_structure_pose_inv
    )
    
    # Prepare sentence (language tokens)
    sentence, sentence_pad_mask = prepare_sentence(goal_specification)
    
    # Pad data
    for i in range(max_num_target_objects - len(target_pcs)):
        target_pcs.append(torch.zeros_like(target_pcs[0], dtype=torch.float32))
        goal_pc_poses.append(np.eye(4))
    
    if not ignore_distractor_objects:
        for i in range(max_num_distractor_objects - len(other_pcs)):
            other_pcs.append(torch.zeros_like(target_pcs[0], dtype=torch.float32))
    
    # Prepare type_index, position_index, pad_mask
    max_num_shape_parameters = len(sentence)
    
    if use_virtual_structure_frame:
        if ignore_distractor_objects:
            pcs = target_pcs
            type_index = [0] * max_num_shape_parameters + [2] + [3] * max_num_target_objects
            position_index = list(range(max_num_shape_parameters)) + [0] + list(range(max_num_target_objects))
            pad_mask = sentence_pad_mask + [0] + [0 if i < len(target_object_names) else 1 
                                                  for i in range(max_num_target_objects)]
            goal_poses = ([goal_structure_pose] if goal_structure_pose is not None else [np.eye(4)]) + goal_pc_poses
        else:
            pcs = other_pcs + target_pcs
            type_index = [0] * max_num_shape_parameters + [1] * max_num_distractor_objects + [2] + [3] * max_num_target_objects
            position_index = (list(range(max_num_shape_parameters)) + 
                           list(range(max_num_distractor_objects)) + 
                           [0] + 
                           list(range(max_num_target_objects)))
            pad_mask = (sentence_pad_mask + 
                       [0 if i < len(other_object_names) else 1 for i in range(max_num_distractor_objects)] +
                       [0] +
                       [0 if i < len(target_object_names) else 1 for i in range(max_num_target_objects)])
            goal_poses = ([goal_structure_pose] if goal_structure_pose is not None else [np.eye(4)]) + goal_pc_poses
    else:
        if ignore_distractor_objects:
            pcs = target_pcs
            type_index = [0] * max_num_shape_parameters + [3] * max_num_target_objects
            position_index = list(range(max_num_shape_parameters)) + list(range(max_num_target_objects))
            pad_mask = sentence_pad_mask + [0 if i < len(target_object_names) else 1 
                                             for i in range(max_num_target_objects)]
            goal_poses = goal_pc_poses
        else:
            pcs = other_pcs + target_pcs
            type_index = [0] * max_num_shape_parameters + [1] * max_num_distractor_objects + [3] * max_num_target_objects
            position_index = (list(range(max_num_shape_parameters)) + 
                           list(range(max_num_distractor_objects)) + 
                           list(range(max_num_target_objects)))
            pad_mask = (sentence_pad_mask + 
                       [0 if i < len(other_object_names) else 1 for i in range(max_num_distractor_objects)] +
                       [0 if i < len(target_object_names) else 1 for i in range(max_num_target_objects)])
            goal_poses = goal_pc_poses
    
    return {
        "pcs": pcs,
        "sentence": sentence,
        "goal_poses": goal_poses,
        "type_index": type_index,
        "position_index": position_index,
        "pad_mask": pad_mask,
        "t": 0,  # timestep
        "filename": "mujoco_scene"  # placeholder
    }


def compute_goal_poses_from_structure(
    goal_specification: Dict,
    target_object_names: List[str],
    current_poses: List[np.ndarray],
    current_pc_poses: List[np.ndarray],
    use_virtual_structure_frame: bool,
    goal_structure_pose_inv: Optional[np.ndarray]
) -> List[np.ndarray]:
    """
    Compute goal poses for target objects based on structure specification.
    """
    structure_type = goal_specification["type"]
    num_objects = len(target_object_names)
    goal_pc_poses = []
    
    if structure_type == "circle":
        radius = goal_specification.get("radius", 0.1)
        center = np.array(goal_specification["position"][:2])
        rotation_z = goal_specification["rotation"][2]
        
        for i in range(num_objects):
            angle = 2 * np.pi * i / num_objects + rotation_z
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = goal_specification["position"][2]
            
            goal_pose = euler_matrix(0, 0, angle)
            goal_pose[:3, 3] = [x, y, z]
            
            # Safely compute inverse
            current_pose_inv = safe_inv(current_poses[i])
            goal_pc_pose = goal_pose @ current_pose_inv @ current_pc_poses[i]
            
            if use_virtual_structure_frame and goal_structure_pose_inv is not None:
                goal_pc_pose = goal_structure_pose_inv @ goal_pc_pose
            
            goal_pc_poses.append(goal_pc_pose)
    
    elif structure_type == "line":
        length = goal_specification.get("length", 0.2)
        center = np.array(goal_specification["position"][:2])
        rotation_z = goal_specification["rotation"][2]
        
        for i in range(num_objects):
            t = (i - (num_objects - 1) / 2) / max(1, num_objects - 1)
            offset = t * length / 2
            
            x = center[0] + offset * np.cos(rotation_z)
            y = center[1] + offset * np.sin(rotation_z)
            z = goal_specification["position"][2]
            
            goal_pose = euler_matrix(0, 0, rotation_z)
            goal_pose[:3, 3] = [x, y, z]
            
            # Safely compute inverse
            current_pose_inv = safe_inv(current_poses[i])
            goal_pc_pose = goal_pose @ current_pose_inv @ current_pc_poses[i]
            
            if use_virtual_structure_frame and goal_structure_pose_inv is not None:
                goal_pc_pose = goal_structure_pose_inv @ goal_pc_pose
            
            goal_pc_poses.append(goal_pc_pose)
    
    elif structure_type in ["tower", "dinner"]:
        center = np.array(goal_specification["position"][:2])
        rotation_z = goal_specification["rotation"][2]
        base_z = goal_specification["position"][2]
        
        for i in range(num_objects):
            obj_height = 0.05  # Default height
            z = base_z + i * obj_height
            
            goal_pose = euler_matrix(0, 0, rotation_z)
            goal_pose[:3, 3] = [center[0], center[1], z]
            
            # Safely compute inverse
            current_pose_inv = safe_inv(current_poses[i])
            goal_pc_pose = goal_pose @ current_pose_inv @ current_pc_poses[i]
            
            if use_virtual_structure_frame and goal_structure_pose_inv is not None:
                goal_pc_pose = goal_structure_pose_inv @ goal_pc_pose
            
            goal_pc_poses.append(goal_pc_pose)
    
    else:
        # Default: keep current poses
        goal_pc_poses = current_pc_poses.copy()
    
    return goal_pc_poses


def prepare_sentence(goal_specification: Dict) -> tuple:
    """
    Prepare sentence tokens from goal specification.
    """
    sentence = []
    structure_type = goal_specification["type"]
    
    if structure_type in ["circle", "line"]:
        sentence.append((structure_type, "shape"))
        sentence.append((goal_specification["rotation"][2], "rotation"))
        sentence.append((goal_specification["position"][0], "position_x"))
        sentence.append((goal_specification["position"][1], "position_y"))
        if structure_type == "circle":
            sentence.append((goal_specification.get("radius", 0.1), "radius"))
        elif structure_type == "line":
            sentence.append((goal_specification.get("length", 0.2) / 2.0, "radius"))
        sentence_pad_mask = [0] * 5
    else:
        sentence.append((structure_type, "shape"))
        sentence.append((goal_specification["rotation"][2], "rotation"))
        sentence.append((goal_specification["position"][0], "position_x"))
        sentence.append((goal_specification["position"][1], "position_y"))
        sentence.append(("PAD", None))
        sentence_pad_mask = [0] * 4 + [1]
    
    return sentence, sentence_pad_mask



