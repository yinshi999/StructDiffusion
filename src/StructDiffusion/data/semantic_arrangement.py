import copy
import cv2
import h5py
import numpy as np
import os
import trimesh
import torch
from tqdm import tqdm
import json
import random
import mujoco

from torch.utils.data import DataLoader

# Local imports
from StructDiffusion.utils.rearrangement import show_pcs, get_pts, combine_and_sample_xyzs
from StructDiffusion.language.tokenizer import Tokenizer

import StructDiffusion.utils.brain2.camera as cam
import StructDiffusion.utils.brain2.image as img
import StructDiffusion.utils.transformations as tra


class SemanticArrangementDataset(torch.utils.data.Dataset):

    def __init__(self, data_roots, index_roots, split, tokenizer,
                 max_num_target_objects=11, max_num_distractor_objects=5,
                 max_num_shape_parameters=7, max_num_rearrange_features=1, max_num_anchor_features=3,
                 num_pts=1024,
                 use_virtual_structure_frame=True, ignore_distractor_objects=True, ignore_rgb=True,
                 filter_num_moved_objects_range=None, shuffle_object_index=False,
                 data_augmentation=True, debug=False, **kwargs):
        """

        Note: setting filter_num_moved_objects_range=[k, k] and max_num_objects=k will create no padding for target objs

        :param data_root:
        :param split: train, valid, or test
        :param shuffle_object_index: whether to shuffle the positions of target objects and other objects in the sequence
        :param debug:
        :param max_num_shape_parameters:
        :param max_num_objects:
        :param max_num_rearrange_features:
        :param max_num_anchor_features:
        :param num_pts:
        :param use_stored_arrangement_indices:
        :param kwargs:
        """

        self.use_virtual_structure_frame = use_virtual_structure_frame
        self.ignore_distractor_objects = ignore_distractor_objects
        self.ignore_rgb = ignore_rgb and not debug

        self.num_pts = num_pts
        self.debug = debug

        self.max_num_objects = max_num_target_objects
        self.max_num_other_objects = max_num_distractor_objects
        self.max_num_shape_parameters = max_num_shape_parameters
        self.max_num_rearrange_features = max_num_rearrange_features
        self.max_num_anchor_features = max_num_anchor_features
        self.shuffle_object_index = shuffle_object_index

        # used to tokenize the language part
        self.tokenizer = tokenizer

        # retrieve data
        self.data_roots = data_roots
        self.arrangement_data = []
        arrangement_steps = []
        for ddx in range(len(data_roots)):
            data_root = data_roots[ddx]
            index_root = index_roots[ddx]
            arrangement_indices_file = os.path.join(data_root, index_root, "{}_arrangement_indices_file_all.txt".format(split))
            if os.path.exists(arrangement_indices_file):
                with open(arrangement_indices_file, "r") as fh:
                    arrangement_steps.extend([(os.path.join(data_root, f[0]), f[1]) for f in eval(fh.readline().strip())])
            else:
                print("{} does not exist".format(arrangement_indices_file))
        # only keep the goal, ignore the intermediate steps
        for filename, step_t in arrangement_steps:
            if step_t == 0:
                if "data00026058" in filename or "data00011415" in filename or "data00026061" in filename or "data00700565" in filename:
                    continue
                self.arrangement_data.append((filename, step_t))
        # if specified, filter data
        if filter_num_moved_objects_range is not None:
            self.arrangement_data = self.filter_based_on_number_of_moved_objects(filter_num_moved_objects_range)
        print("{} valid sequences".format(len(self.arrangement_data)))

        # Data Aug
        self.data_augmentation = data_augmentation
        # additive noise
        self.gp_rescale_factor_range = [12, 20]
        self.gaussian_scale_range = [0., 0.003]
        # multiplicative noise
        self.gamma_shape = 1000.
        self.gamma_scale = 0.001

    def filter_based_on_number_of_moved_objects(self, filter_num_moved_objects_range):
        assert len(list(filter_num_moved_objects_range)) == 2
        min_num, max_num = filter_num_moved_objects_range
        print("Remove scenes that have less than {} or more than {} objects being moved".format(min_num, max_num))
        ok_data = []
        for filename, step_t in self.arrangement_data:
            h5 = h5py.File(filename, 'r')
            moved_objs = h5['moved_objs'][()].split(',')
            if min_num <= len(moved_objs) <= max_num:
                ok_data.append((filename, step_t))
        print("{} valid sequences left".format(len(ok_data)))
        return ok_data

    def get_data_idx(self, idx):
        # Create the datum to return
        file_idx = np.argmax(idx < self.file_to_count)
        data = h5py.File(self.data_files[file_idx], 'r')
        if file_idx > 0:
            # for lang2sym, idx is always 0
            idx = idx - self.file_to_count[file_idx - 1]
        return data, idx, file_idx

    def add_noise_to_depth(self, depth_img):
        """ add depth noise """
        multiplicative_noise = np.random.gamma(self.gamma_shape, self.gamma_scale)
        depth_img = multiplicative_noise * depth_img
        return depth_img

    def add_noise_to_xyz(self, xyz_img, depth_img):
        """ TODO: remove this code or at least celean it up"""
        xyz_img = xyz_img.copy()
        H, W, C = xyz_img.shape
        gp_rescale_factor = np.random.randint(self.gp_rescale_factor_range[0],
                                              self.gp_rescale_factor_range[1])
        gp_scale = np.random.uniform(self.gaussian_scale_range[0],
                                     self.gaussian_scale_range[1])
        small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
        additive_noise = np.random.normal(loc=0.0, scale=gp_scale, size=(small_H, small_W, C))
        additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
        xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]
        return xyz_img

    def random_index(self):
        return self[np.random.randint(len(self))]

    def _get_rgb(self, h5, idx, ee=True):
        RGB = "ee_rgb" if ee else "rgb"
        rgb1 = img.PNGToNumpy(h5[RGB][idx])[:, :, :3] / 255.  # remove alpha
        return rgb1

    def _get_depth(self, h5, idx, ee=True):
        DEPTH = "ee_depth" if ee else "depth"

    def _get_images(self, h5, idx, ee=True):
        if ee:
            RGB, DEPTH, SEG = "ee_rgb", "ee_depth", "ee_seg"
            DMIN, DMAX = "ee_depth_min", "ee_depth_max"
        else:
            RGB, DEPTH, SEG = "rgb", "depth", "seg"
            DMIN, DMAX = "depth_min", "depth_max"
        dmin = h5[DMIN][idx]
        dmax = h5[DMAX][idx]
        rgb1 = img.PNGToNumpy(h5[RGB][idx])[:, :, :3] / 255.  # remove alpha
        depth1 = h5[DEPTH][idx] / 20000. * (dmax - dmin) + dmin
        seg1 = img.PNGToNumpy(h5[SEG][idx])

        valid1 = np.logical_and(depth1 > 0.1, depth1 < 2.)

        # proj_matrix = h5['proj_matrix'][()]
        camera = cam.get_camera_from_h5(h5)
        if self.data_augmentation:
            depth1 = self.add_noise_to_depth(depth1)

        xyz1 = cam.compute_xyz(depth1, camera)
        if self.data_augmentation:
            xyz1 = self.add_noise_to_xyz(xyz1, depth1)

        # Transform the point cloud
        # Here it is...
        # CAM_POSE = "ee_cam_pose" if ee else "cam_pose"
        CAM_POSE = "ee_camera_view" if ee else "camera_view"
        cam_pose = h5[CAM_POSE][idx]
        if ee:
            # ee_camera_view has 0s for x, y, z
            cam_pos = h5["ee_cam_pose"][:][:3, 3]
            cam_pose[:3, 3] = cam_pos

        # Get transformed point cloud
        h, w, d = xyz1.shape
        xyz1 = xyz1.reshape(h * w, -1)
        xyz1 = trimesh.transform_points(xyz1, cam_pose)
        xyz1 = xyz1.reshape(h, w, -1)

        scene1 = rgb1, depth1, seg1, valid1, xyz1

        return scene1

    def __len__(self):
        return len(self.arrangement_data)

    def _get_ids(self, h5):
        """
        get object ids

        @param h5:
        @return:
        """
        ids = {}
        for k in h5.keys():
            if k.startswith("id_"):
                ids[k[3:]] = h5[k][()]
        return ids

    def _load_mesh_and_sample_pc(self, mesh_path, num_pts, apply_transform=None):
        """
        Load mesh from OBJ file and sample point cloud from it.
        
        @param mesh_path: Path to the .obj file
        @param num_pts: Number of points to sample
        @param apply_transform: Optional 4x4 transformation matrix to apply to the points
        @return: Point cloud tensor of shape (num_pts, 3)
        """
        try:
            # Load mesh from OBJ file
            mesh = trimesh.load(mesh_path, force='mesh')
            
            # Sample points on the surface (uniform by area)
            points, face_indices = trimesh.sample.sample_surface(mesh, count=num_pts)
            
            # Apply transformation if provided
            if apply_transform is not None:
                points = trimesh.transform_points(points, apply_transform)
            
            # Convert to tensor
            points_tensor = torch.FloatTensor(points)
            
            return points_tensor
        except Exception as e:
            print(f"Error loading mesh from {mesh_path}: {e}")
            # Return random points as fallback
            return torch.randn(num_pts, 3) * 0.1
    
    def _get_mesh_path_for_object(self, h5, obj_name, base_mesh_dir="mujoco/meshes/assets"):
        """
        Get the mesh file path for an object. 
        
        First tries to read from h5 file if available, otherwise constructs path from object name.
        
        @param h5: HDF5 file handle
        @param obj_name: Object name
        @param base_mesh_dir: Base directory for mesh files
        @return: Path to mesh file
        """
        # Try to get mesh path from h5 file
        mesh_path_key = f"{obj_name}_mesh_path"
        if mesh_path_key in h5.keys():
            return str(np.array(h5[mesh_path_key]))
        
        # Try to extract object type from goal_specification
        try:
            goal_specification = json.loads(str(np.array(h5["goal_specification"])))
            # Look for object in rearrange, anchor, or distract lists
            for category in ['rearrange', 'anchor', 'distract']:
                if category in goal_specification:
                    for obj_spec in goal_specification[category]['objects']:
                        if isinstance(obj_spec, dict) and 'mesh' in obj_spec:
                            # Assume mesh file is named after object type
                            mesh_filename = obj_spec['mesh']
                            # Construct full path
                            import os
                            # Get the directory of the h5 file
                            h5_dir = os.path.dirname(h5.filename)
                            # Go up to the repo root and construct path
                            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(h5_dir)))
                            return os.path.join(repo_root, base_mesh_dir, mesh_filename)
        except:
            pass
        
        # Fallback: construct path from object name (assuming object type is in the name)
        # For example: "object_0" might map to "bowl.obj" or similar
        # This is a placeholder - you should customize based on your naming convention
        import os
        h5_dir = os.path.dirname(h5.filename) if hasattr(h5, 'filename') else "."
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(h5_dir)))
        
        # Default to bowl.obj as example (you should modify this based on your data)
        default_mesh = "bowl.obj"
        return os.path.join(repo_root, base_mesh_dir, default_mesh)
    
    def _get_mesh_path_from_mujoco(self, model, body_name, scene_xml_dir="."):
        """
        Get the mesh file path for an object from MuJoCo model.
        
        @param model: MuJoCo model
        @param body_name: Body name in MuJoCo scene
        @param scene_xml_dir: Directory where the scene XML is located
        @return: Path to mesh file or None if not a mesh geom
        """
        try:
            # Find the body
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                return None
            
            # Find geoms belonging to this body
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == body_id:
                    geom_type = model.geom_type[geom_id]
                    
                    # Check if it's a mesh geom
                    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                        mesh_id = model.geom_dataid[geom_id]
                        # Get mesh file path from model
                        # MuJoCo stores mesh paths in the compiled model
                        # We need to extract it from the mesh name and asset path
                        mesh_name_adr = model.name_meshadr[mesh_id]
                        mesh_name = model.names[mesh_name_adr:].decode('utf-8').split('\x00')[0]
                        
                        # Find corresponding asset file
                        # The mesh file is typically defined in the XML <asset> section
                        # For now, construct path from mesh name
                        mesh_filename = f"{mesh_name.replace('_mesh', '')}.obj"
                        return os.path.join(scene_xml_dir, "meshes", "assets", mesh_filename)
            
            return None
        except Exception as e:
            print(f"Error getting mesh path for {body_name}: {e}")
            return None

    def get_positive_ratio(self):
        num_pos = 0
        for d in self.arrangement_data:
            filename, step_t = d
            if step_t == 0:
                num_pos += 1
        return (len(self.arrangement_data) - num_pos) * 1.0 / num_pos

    def get_object_position_vocab_sizes(self):
        return self.tokenizer.get_object_position_vocab_sizes()

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_data_index(self, idx):
        filename = self.arrangement_data[idx]
        return filename

    def get_raw_data_from_mujoco(self, model, data, target_object_names, goal_specification,
                                  other_object_names=None, scene_xml_dir=".",
                                  inference_mode=False, shuffle_object_index=False):
        """
        Get data from a MuJoCo scene instead of h5 file.
        
        @param model: MuJoCo model
        @param data: MuJoCo data (current state)
        @param target_object_names: List of target object body names
        @param goal_specification: Goal specification dict (same format as h5)
        @param other_object_names: List of other object body names (distractors/anchors)
        @param scene_xml_dir: Directory where scene.xml is located
        @param inference_mode: Whether in inference mode
        @param shuffle_object_index: Whether to shuffle object indices
        @return: datum dictionary
        """
        if other_object_names is None:
            other_object_names = []
        
        num_rearrange_objs = len(target_object_names)
        num_other_objs = len(other_object_names)
        
        assert num_rearrange_objs <= self.max_num_objects
        assert num_other_objs <= self.max_num_other_objects
        
        all_objs = target_object_names + other_object_names
        target_objs = target_object_names
        other_objs = other_object_names
        
        structure_parameters = goal_specification["shape"]
        
        # Important: ensure the order is correct
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            target_objs = target_objs[::-1]
        elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
            target_objs = target_objs
        else:
            raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))
        all_objs = target_objs + other_objs
        
        ###################################
        # getting object point clouds FROM MESH FILES
        obj_pcs = []
        obj_pad_mask = []
        current_pc_poses = []
        other_obj_pcs = []
        other_obj_pad_mask = []
        current_obj_poses = []
        
        for obj in all_objs:
            # Get the current pose of the object from MuJoCo
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
            if body_id < 0:
                print(f"Warning: Body {obj} not found in MuJoCo model")
                continue
            
            current_pose = np.eye(4, dtype=np.float64)
            # MuJoCo stores rotation matrix in row-major order
            rot_mat = data.xmat[body_id].reshape(3, 3).copy()
            
            # Validate rotation matrix
            det = np.linalg.det(rot_mat)
            if abs(det - 1.0) > 0.1 or np.linalg.norm(rot_mat) < 0.1:
                # Invalid rotation matrix, use identity
                print(f"Warning: Invalid rotation matrix for {obj} (det={det:.4f}), using identity")
                rot_mat = np.eye(3, dtype=np.float64)
            
            current_pose[:3, :3] = rot_mat
            current_pose[:3, 3] = data.xpos[body_id].copy()
            
            # Get mesh file path for this object from MuJoCo
            mesh_path = self._get_mesh_path_from_mujoco(model, obj, scene_xml_dir)
            
            if mesh_path is None or not os.path.exists(mesh_path):
                print(f"Warning: Mesh file not found for {obj}, generating simple geometry")
                # For non-mesh geoms (box, sphere, etc.), generate point cloud from primitive
                obj_xyz = self._generate_pc_from_mujoco_geom(model, data, obj)
            else:
                # Load mesh and sample point cloud
                obj_xyz = self._load_mesh_and_sample_pc(mesh_path, self.num_pts, apply_transform=current_pose)
            
            # Generate RGB values
            if not self.ignore_rgb:
                obj_rgb = torch.ones(self.num_pts, 3) * 0.5  # Gray color
            
            if obj in target_objs:
                if self.ignore_rgb:
                    obj_pcs.append(obj_xyz)
                else:
                    obj_pcs.append(torch.concat([obj_xyz, obj_rgb], dim=-1))
                obj_pad_mask.append(0)
                pc_pose = np.eye(4)
                pc_pose[:3, 3] = torch.mean(obj_xyz, dim=0).numpy()
                current_pc_poses.append(pc_pose)
                current_obj_poses.append(current_pose)
            elif obj in other_objs:
                if self.ignore_rgb:
                    other_obj_pcs.append(obj_xyz)
                else:
                    other_obj_pcs.append(torch.concat([obj_xyz, obj_rgb], dim=-1))
                other_obj_pad_mask.append(0)
            else:
                raise Exception
        
        ###################################
        # computes goal positions for objects
        if self.use_virtual_structure_frame:
            goal_structure_pose = tra.euler_matrix(structure_parameters["rotation"][0], 
                                              structure_parameters["rotation"][1],
                                              structure_parameters["rotation"][2])
            goal_structure_pose[:3, 3] = [structure_parameters["position"][0], 
                                     structure_parameters["position"][1],
                                     structure_parameters["position"][2]]
            goal_structure_pose_inv = np.linalg.inv(goal_structure_pose)
        
        # For MuJoCo, you need to provide goal poses separately
        # This is a placeholder - you should set these based on your task
        goal_obj_poses = []
        goal_pc_poses = []
        for current_pc_pose, current_obj_pose in zip(current_pc_poses, current_obj_poses):
            # Placeholder: goal is same as current (you should modify this)
            goal_pose = current_obj_pose.copy()
            
            # Compute transformation from current to goal
            try:
                current_obj_pose_inv = np.linalg.inv(current_obj_pose)
                goal_pc_pose = goal_pose @ current_obj_pose_inv @ current_pc_pose
            except np.linalg.LinAlgError:
                # If matrix is singular, goal pose equals current pose
                print(f"Warning: Singular matrix when computing goal pose, using identity transform")
                goal_pc_pose = current_pc_pose.copy()
            
            if self.use_virtual_structure_frame:
                goal_pc_pose = goal_structure_pose_inv @ goal_pc_pose
            
            goal_obj_poses.append(goal_pose)
            goal_pc_poses.append(goal_pc_pose)
        
        ###################################
        # preparing sentence
        sentence = []
        sentence_pad_mask = []
        
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            if structure_parameters["type"] == "circle":
                sentence.append((structure_parameters["radius"], "radius"))
            elif structure_parameters["type"] == "line":
                sentence.append((structure_parameters["length"] / 2.0, "radius"))
            for _ in range(5):
                sentence_pad_mask.append(0)
        else:
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            for _ in range(4):
                sentence_pad_mask.append(0)
            sentence.append(("PAD", None))
            sentence_pad_mask.append(1)
        
        ###################################
        # paddings
        for i in range(self.max_num_objects - len(target_objs)):
            obj_pcs.append(torch.zeros_like(obj_pcs[0], dtype=torch.float32))
            obj_pad_mask.append(1)
            goal_pc_poses.append(np.eye(4))
        
        for i in range(self.max_num_other_objects - len(other_objs)):
            other_obj_pcs.append(torch.zeros_like(obj_pcs[0], dtype=torch.float32))
            other_obj_pad_mask.append(1)
        
        ###################################
        # shuffle if needed
        if shuffle_object_index:
            shuffle_target_object_indices = list(range(len(target_objs)))
            random.shuffle(shuffle_target_object_indices)
            shuffle_object_indices = shuffle_target_object_indices + list(range(len(target_objs), self.max_num_objects))
            obj_pcs = [obj_pcs[i] for i in shuffle_object_indices]
            goal_pc_poses = [goal_pc_poses[i] for i in shuffle_object_indices]
            if inference_mode:
                goal_obj_poses = [goal_obj_poses[i] for i in shuffle_object_indices]
                current_obj_poses = [current_obj_poses[i] for i in shuffle_object_indices]
                target_objs = [target_objs[i] for i in shuffle_target_object_indices]
                current_pc_poses = [current_pc_poses[i] for i in shuffle_object_indices]
        
        ###################################
        # construct final datum
        if self.use_virtual_structure_frame:
            if self.ignore_distractor_objects:
                pcs = obj_pcs
                type_index = [0] * self.max_num_shape_parameters + [2] + [3] * self.max_num_objects
                position_index = list(range(self.max_num_shape_parameters)) + [0] + list(range(self.max_num_objects))
                pad_mask = sentence_pad_mask + [0] + obj_pad_mask
            else:
                pcs = other_obj_pcs + obj_pcs
                type_index = [0] * self.max_num_shape_parameters + [1] * self.max_num_other_objects + [2] + [3] * self.max_num_objects
                position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_other_objects)) + [0] + list(range(self.max_num_objects))
                pad_mask = sentence_pad_mask + other_obj_pad_mask + [0] + obj_pad_mask
            goal_poses = [goal_structure_pose] + goal_pc_poses
        else:
            if self.ignore_distractor_objects:
                pcs = obj_pcs
                type_index = [0] * self.max_num_shape_parameters + [3] * self.max_num_objects
                position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_objects))
                pad_mask = sentence_pad_mask + obj_pad_mask
            else:
                pcs = other_obj_pcs + obj_pcs
                type_index = [0] * self.max_num_shape_parameters + [1] * self.max_num_other_objects + [3] * self.max_num_objects
                position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_other_objects)) + list(range(self.max_num_objects))
                pad_mask = sentence_pad_mask + other_obj_pad_mask + obj_pad_mask
            goal_poses = goal_pc_poses
        
        datum = {
            "pcs": pcs,
            "sentence": sentence,
            "goal_poses": goal_poses,
            "type_index": type_index,
            "position_index": position_index,
            "pad_mask": pad_mask,
            "t": num_rearrange_objs,
            "filename": "mujoco_scene"
        }
        
        if inference_mode:
            datum["goal_obj_poses"] = goal_obj_poses
            datum["current_obj_poses"] = current_obj_poses
            datum["target_objs"] = target_objs
            datum["goal_specification"] = goal_specification
            datum["current_pc_poses"] = current_pc_poses
        
        return datum
    
    def _generate_pc_from_mujoco_geom(self, model, data, body_name):
        """
        Generate point cloud from MuJoCo primitive geom (box, sphere, etc.).
        
        @param model: MuJoCo model
        @param data: MuJoCo data
        @param body_name: Body name
        @return: Point cloud tensor
        """
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return torch.randn(self.num_pts, 3) * 0.1
        
        # Get body pose
        obj_pose = np.eye(4, dtype=np.float64)
        rot_mat = data.xmat[body_id].reshape(3, 3).copy()
        
        # Validate rotation matrix
        det = np.linalg.det(rot_mat)
        if abs(det - 1.0) > 0.1 or np.linalg.norm(rot_mat) < 0.1:
            rot_mat = np.eye(3, dtype=np.float64)
        
        obj_pose[:3, :3] = rot_mat
        obj_pose[:3, 3] = data.xpos[body_id]
        
        # Find geom and generate primitive
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                geom_type = model.geom_type[geom_id]
                geom_size = model.geom_size[geom_id]
                
                if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    box = trimesh.creation.box(extents=geom_size * 2)
                    points, _ = trimesh.sample.sample_surface(box, count=self.num_pts)
                elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    sphere = trimesh.creation.icosphere(subdivisions=2, radius=geom_size[0])
                    points, _ = trimesh.sample.sample_surface(sphere, count=self.num_pts)
                else:
                    # Fallback to random points
                    points = np.random.randn(self.num_pts, 3) * 0.05
                
                # Transform to world frame
                points = trimesh.transform_points(points, obj_pose)
                return torch.FloatTensor(points)
        
        return torch.randn(self.num_pts, 3) * 0.1

    def get_raw_data(self, idx, inference_mode=False, shuffle_object_index=False):
        """

        :param idx:
        :param inference_mode:
        :param shuffle_object_index: used to test different orders of objects
        :return:
        """

        filename, _ = self.arrangement_data[idx]

        h5 = h5py.File(filename, 'r')
        ids = self._get_ids(h5)
        all_objs = sorted([o for o in ids.keys() if "object_" in o])
        goal_specification = json.loads(str(np.array(h5["goal_specification"])))
        num_rearrange_objs = len(goal_specification["rearrange"]["objects"])
        num_other_objs = len(goal_specification["anchor"]["objects"] + goal_specification["distract"]["objects"])
        assert len(all_objs) == num_rearrange_objs + num_other_objs, "{}, {}".format(len(all_objs), num_rearrange_objs + num_other_objs)
        assert num_rearrange_objs <= self.max_num_objects
        assert num_other_objs <= self.max_num_other_objects

        # important: only using the last step
        step_t = num_rearrange_objs

        target_objs = all_objs[:num_rearrange_objs]
        other_objs = all_objs[num_rearrange_objs:]

        structure_parameters = goal_specification["shape"]

        # Important: ensure the order is correct
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            target_objs = target_objs[::-1]
        elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
            target_objs = target_objs
        else:
            raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))
        all_objs = target_objs + other_objs

        ###################################
        # getting scene images and point clouds (for inference mode compatibility)
        if inference_mode:
            scene = self._get_images(h5, step_t, ee=True)
            rgb, depth, seg, valid, xyz = scene
            initial_scene = scene
        else:
            # For training, we don't need the full scene rendering
            rgb = None

        # getting object point clouds FROM MESH FILES
        obj_pcs = []
        obj_pad_mask = []
        current_pc_poses = []
        other_obj_pcs = []
        other_obj_pad_mask = []
        
        for obj in all_objs:
            # Get the current pose of the object from h5 file
            current_pose = h5[obj][step_t]
            
            # Get mesh file path for this object
            mesh_path = self._get_mesh_path_for_object(h5, obj)
            
            # Load mesh and sample point cloud
            # The mesh is loaded in its canonical pose, then transformed to current pose
            obj_xyz = self._load_mesh_and_sample_pc(mesh_path, self.num_pts, apply_transform=current_pose)
            
            # Generate RGB values (can be loaded from mesh if available, or use defaults)
            if not self.ignore_rgb:
                # Default: use a neutral color or extract from mesh material
                obj_rgb = torch.ones(self.num_pts, 3) * 0.5  # Gray color

            if obj in target_objs:
                if self.ignore_rgb:
                    obj_pcs.append(obj_xyz)
                else:
                    obj_pcs.append(torch.concat([obj_xyz, obj_rgb], dim=-1))
                obj_pad_mask.append(0)
                pc_pose = np.eye(4)
                pc_pose[:3, 3] = torch.mean(obj_xyz, dim=0).numpy()
                current_pc_poses.append(pc_pose)
            elif obj in other_objs:
                if self.ignore_rgb:
                    other_obj_pcs.append(obj_xyz)
                else:
                    other_obj_pcs.append(torch.concat([obj_xyz, obj_rgb], dim=-1))
                other_obj_pad_mask.append(0)
            else:
                raise Exception

        ###################################
        # computes goal positions for objects
        # Important: because of the noises we added to point clouds, the rearranged point clouds will not be perfect
        if self.use_virtual_structure_frame:
            goal_structure_pose = tra.euler_matrix(structure_parameters["rotation"][0], structure_parameters["rotation"][1],
                                              structure_parameters["rotation"][2])
            goal_structure_pose[:3, 3] = [structure_parameters["position"][0], structure_parameters["position"][1],
                                     structure_parameters["position"][2]]
            goal_structure_pose_inv = np.linalg.inv(goal_structure_pose)

        goal_obj_poses = []
        current_obj_poses = []
        goal_pc_poses = []
        for obj, current_pc_pose in zip(target_objs, current_pc_poses):
            goal_pose = h5[obj][0]
            current_pose = h5[obj][step_t]
            if inference_mode:
                goal_obj_poses.append(goal_pose)
                current_obj_poses.append(current_pose)

            goal_pc_pose = goal_pose @ np.linalg.inv(current_pose) @ current_pc_pose
            if self.use_virtual_structure_frame:
                goal_pc_pose = goal_structure_pose_inv @ goal_pc_pose
            goal_pc_poses.append(goal_pc_pose)

        # transform current object point cloud to the goal point cloud in the world frame
        if self.debug:
            new_obj_pcs = [copy.deepcopy(pc.numpy()) for pc in obj_pcs]
            for i, obj_pc in enumerate(new_obj_pcs):

                current_pc_pose = current_pc_poses[i]
                goal_pc_pose = goal_pc_poses[i]
                if self.use_virtual_structure_frame:
                    goal_pc_pose = goal_structure_pose @ goal_pc_pose
                print("current pc pose", current_pc_pose)
                print("goal pc pose", goal_pc_pose)

                goal_pc_transform = goal_pc_pose @ np.linalg.inv(current_pc_pose)
                print("transform", goal_pc_transform)
                new_obj_pc = copy.deepcopy(obj_pc)
                new_obj_pc[:, :3] = trimesh.transform_points(obj_pc[:, :3], goal_pc_transform)
                print(new_obj_pc.shape)

                # visualize rearrangement sequence (new_obj_xyzs), the current object before moving (obj_xyz), and other objects
                new_obj_pcs[i] = new_obj_pc
                new_obj_pcs[i][:, 3:] = np.tile(np.array([1, 0, 0], dtype=np.float), (new_obj_pc.shape[0], 1))
                new_obj_rgb_current = np.tile(np.array([0, 1, 0], dtype=np.float), (new_obj_pc.shape[0], 1))
                show_pcs([pc[:, :3] for pc in new_obj_pcs] + [pc[:, :3] for pc in other_obj_pcs] + [obj_pc[:, :3]],
                         [pc[:, 3:] for pc in new_obj_pcs] + [pc[:, 3:] for pc in other_obj_pcs] + [new_obj_rgb_current],
                         add_coordinate_frame=True)
            show_pcs([pc[:, :3] for pc in new_obj_pcs], [pc[:, 3:] for pc in new_obj_pcs], add_coordinate_frame=True)

        # pad data
        for i in range(self.max_num_objects - len(target_objs)):
            obj_pcs.append(torch.zeros_like(obj_pcs[0], dtype=torch.float32))
            obj_pad_mask.append(1)
        for i in range(self.max_num_other_objects - len(other_objs)):
            other_obj_pcs.append(torch.zeros_like(obj_pcs[0], dtype=torch.float32))
            other_obj_pad_mask.append(1)

        ###################################
        # preparing sentence
        sentence = []
        sentence_pad_mask = []

        # structure parameters
        # 5 parameters
        structure_parameters = goal_specification["shape"]
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            if structure_parameters["type"] == "circle":
                sentence.append((structure_parameters["radius"], "radius"))
            elif structure_parameters["type"] == "line":
                sentence.append((structure_parameters["length"] / 2.0, "radius"))
            for _ in range(5):
                sentence_pad_mask.append(0)
        else:
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            for _ in range(4):
                sentence_pad_mask.append(0)
            sentence.append(("PAD", None))
            sentence_pad_mask.append(1)

        ###################################
        # paddings
        for i in range(self.max_num_objects - len(target_objs)):
            goal_pc_poses.append(np.eye(4))

        ###################################
        if self.debug:
            print("---")
            print("all objects:", all_objs)
            print("target objects:", target_objs)
            print("other objects:", other_objs)
            print("goal specification:", goal_specification)
            print("sentence:", sentence)
            show_pcs([pc[:, :3] for pc in obj_pcs + other_obj_pcs], [pc[:, 3:] for pc in obj_pcs + other_obj_pcs], add_coordinate_frame=True)

        assert len(obj_pcs) == len(goal_pc_poses)
        ###################################

        # shuffle the position of objects
        if shuffle_object_index:
            shuffle_target_object_indices = list(range(len(target_objs)))
            random.shuffle(shuffle_target_object_indices)
            shuffle_object_indices = shuffle_target_object_indices + list(range(len(target_objs), self.max_num_objects))
            obj_pcs = [obj_pcs[i] for i in shuffle_object_indices]
            goal_pc_poses = [goal_pc_poses[i] for i in shuffle_object_indices]
            if inference_mode:
                goal_obj_poses = [goal_obj_poses[i] for i in shuffle_object_indices]
                current_obj_poses = [current_obj_poses[i] for i in shuffle_object_indices]
                target_objs = [target_objs[i] for i in shuffle_target_object_indices]
                current_pc_poses = [current_pc_poses[i] for i in shuffle_object_indices]

        ###################################
        if self.use_virtual_structure_frame:
            if self.ignore_distractor_objects:
                # language, structure virtual frame, target objects
                pcs = obj_pcs
                type_index = [0] * self.max_num_shape_parameters + [2] + [3] * self.max_num_objects
                position_index = list(range(self.max_num_shape_parameters)) + [0] + list(range(self.max_num_objects))
                pad_mask = sentence_pad_mask + [0] + obj_pad_mask
            else:
                # language, distractor objects, structure virtual frame, target objects
                pcs = other_obj_pcs + obj_pcs
                type_index = [0] * self.max_num_shape_parameters + [1] * self.max_num_other_objects + [2] + [3] * self.max_num_objects
                position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_other_objects)) + [0] + list(range(self.max_num_objects))
                pad_mask = sentence_pad_mask + other_obj_pad_mask + [0] + obj_pad_mask
            goal_poses = [goal_structure_pose] + goal_pc_poses
        else:
            if self.ignore_distractor_objects:
                # language, target objects
                pcs = obj_pcs
                type_index = [0] * self.max_num_shape_parameters + [3] * self.max_num_objects
                position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_objects))
                pad_mask = sentence_pad_mask + obj_pad_mask
            else:
                # language, distractor objects, target objects
                pcs = other_obj_pcs + obj_pcs
                type_index = [0] * self.max_num_shape_parameters + [1] * self.max_num_other_objects + [3] * self.max_num_objects
                position_index = list(range(self.max_num_shape_parameters)) + list(range(self.max_num_other_objects)) + list(range(self.max_num_objects))
                pad_mask = sentence_pad_mask + other_obj_pad_mask + obj_pad_mask
            goal_poses = goal_pc_poses

        datum = {
            "pcs": pcs,
            "sentence": sentence,
            "goal_poses": goal_poses,
            "type_index": type_index,
            "position_index": position_index,
            "pad_mask": pad_mask,
            "t": step_t,
            "filename": filename
        }

        if inference_mode:
            datum["rgb"] = rgb
            datum["goal_obj_poses"] = goal_obj_poses
            datum["current_obj_poses"] = current_obj_poses
            datum["target_objs"] = target_objs
            datum["initial_scene"] = initial_scene
            datum["ids"] = ids
            datum["goal_specification"] = goal_specification
            datum["current_pc_poses"] = current_pc_poses

        return datum

    @staticmethod
    def convert_to_tensors(datum, tokenizer):
        tensors = {
            "pcs": torch.stack(datum["pcs"], dim=0),
            "sentence": torch.LongTensor(np.array([tokenizer.tokenize(*i) for i in datum["sentence"]])),
            "goal_poses": torch.FloatTensor(np.array(datum["goal_poses"])),
            "type_index": torch.LongTensor(np.array(datum["type_index"])),
            "position_index": torch.LongTensor(np.array(datum["position_index"])),
            "pad_mask": torch.LongTensor(np.array(datum["pad_mask"])),
            "t": datum["t"],
            "filename": datum["filename"]
        }
        return tensors

    def __getitem__(self, idx):

        datum = self.convert_to_tensors(self.get_raw_data(idx, shuffle_object_index=self.shuffle_object_index),
                                        self.tokenizer)

        return datum

    def single_datum_to_batch(self, x, num_samples, device, inference_mode=True):
        tensor_x = {}

        tensor_x["pcs"] = x["pcs"].to(device)[None, :, :, :].repeat(num_samples, 1, 1, 1)
        tensor_x["sentence"] = x["sentence"].to(device)[None, :].repeat(num_samples, 1)
        if not inference_mode:
            tensor_x["goal_poses"] = x["goal_poses"].to(device)[None, :, :, :].repeat(num_samples, 1, 1, 1)

        tensor_x["type_index"] = x["type_index"].to(device)[None, :].repeat(num_samples, 1)
        tensor_x["position_index"] = x["position_index"].to(device)[None, :].repeat(num_samples, 1)
        tensor_x["pad_mask"] = x["pad_mask"].to(device)[None, :].repeat(num_samples, 1)

        return tensor_x

    @staticmethod
    def get_mujoco_raw_data(model, data, target_object_names, goal_specification, 
                            other_object_names=None, camera_name=None, num_pts=1024,
                            max_num_target_objects=11, max_num_distractor_objects=5,
                            use_virtual_structure_frame=True, ignore_distractor_objects=True,
                            ignore_rgb=True, width=640, height=480):
        """
        Convenience method to get raw data from MuJoCo scene.
        This is a wrapper around the mujoco_to_repo_converter function.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            target_object_names: List of target object names
            goal_specification: Goal specification dictionary
            other_object_names: List of other object names
            camera_name: Camera name
            num_pts: Number of points per object
            max_num_target_objects: Maximum number of target objects
            max_num_distractor_objects: Maximum number of distractor objects
            use_virtual_structure_frame: Whether to use virtual structure frame
            ignore_distractor_objects: Whether to ignore distractor objects
            ignore_rgb: Whether to ignore RGB
            width: Image width
            height: Image height
        
        Returns:
            Dictionary in same format as get_raw_data()
        """
        from StructDiffusion.utils.mujoco_to_repo_converter import get_mujoco_raw_data as _get_mujoco_raw_data
        return _get_mujoco_raw_data(
            model, data, target_object_names, goal_specification,
            other_object_names, camera_name, num_pts,
            max_num_target_objects, max_num_distractor_objects,
            use_virtual_structure_frame, ignore_distractor_objects,
            ignore_rgb, width, height
        )


def compute_min_max(dataloader):

    # tensor([-0.3557, -0.3847,  0.0000, -1.0000, -1.0000, -0.4759, -1.0000, -1.0000,
    #         -0.9079, -0.8668, -0.9105, -0.4186])
    # tensor([0.3915, 0.3494, 0.3267, 1.0000, 1.0000, 0.8961, 1.0000, 1.0000, 0.8194,
    #         0.4787, 0.6421, 1.0000])
    # tensor([0.0918, -0.3758, 0.0000, -1.0000, -1.0000, 0.0000, -1.0000, -1.0000,
    #         -0.0000, 0.0000, 0.0000, 1.0000])
    # tensor([0.9199, 0.3710, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000, -0.0000,
    #         0.0000, 0.0000, 1.0000])

    min_value = torch.ones(16) * 10000
    max_value = torch.ones(16) * -10000
    for d in tqdm(dataloader):
        goal_poses = d["goal_poses"]
        goal_poses = goal_poses.reshape(-1, 16)
        current_max, _ = torch.max(goal_poses, dim=0)
        current_min, _ = torch.min(goal_poses, dim=0)
        max_value[max_value < current_max] = current_max[max_value < current_max]
        max_value[max_value > current_min] = current_min[max_value > current_min]
    print(f"{min_value} - {max_value}")