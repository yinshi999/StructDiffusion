import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf

from StructDiffusion.data.semantic_arrangement import SemanticArrangementDataset
from StructDiffusion.language.tokenizer import Tokenizer
from StructDiffusion.models.pl_models import ConditionalPoseDiffusionModel
from StructDiffusion.diffusion.sampler import Sampler
from StructDiffusion.diffusion.pose_conversion import get_struct_objs_poses
from StructDiffusion.utils.files import get_checkpoint_path_from_dir, replace_config_for_testing_data
from StructDiffusion.utils.batch_inference import move_pc_and_create_scene_simple, visualize_batch_pcs
import mujoco

def main(args, cfg):

    pl.seed_everything(args.eval_random_seed)

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    checkpoint_dir = os.path.join(cfg.WANDB.save_dir, cfg.WANDB.project, args.checkpoint_id, "checkpoints")
    checkpoint_path = get_checkpoint_path_from_dir(checkpoint_dir)

    if args.eval_mode == "infer":

        tokenizer = Tokenizer(cfg.DATASET.vocab_dir)
        # override ignore_rgb for visualization
        cfg.DATASET.ignore_rgb = False
        sampler = Sampler(ConditionalPoseDiffusionModel, checkpoint_path, device)

        if args.use_mujoco:
            # Use MuJoCo scene input
            from StructDiffusion.utils.mujoco_to_repo_converter import get_mujoco_raw_data
            
            print("=" * 50)
            print("Loading MuJoCo scene...")
            # Load MuJoCo scene
            mujoco_scene_path = os.path.join(os.path.dirname(__file__), "..", "mujoco", "scene.xml")
            if not os.path.exists(mujoco_scene_path):
                raise FileNotFoundError(f"MuJoCo scene not found at: {mujoco_scene_path}")
            
            mujoco_model = mujoco.MjModel.from_xml_path(mujoco_scene_path)
            mujoco_data = mujoco.MjData(mujoco_model)
            print(f"Loaded MuJoCo scene with {mujoco_model.nbody} bodies")
            
            # Define objects and goal specification
            target_objects = ["box", "sphere", "bowl"]
            goal_spec = {
                "type": "circle",
                "position": [0.4, 0.0, 0.55],
                "rotation": [0.0, 0.0, 0.0],
                "radius": 0.1
            }
            
            print(f"Target objects: {target_objects}")
            print(f"Goal specification: {goal_spec}")
            print("Extracting point clouds from MuJoCo...")
            
            # Get raw data from MuJoCo (same format as get_raw_data)
            try:
                raw_datum = get_mujoco_raw_data(
                    mujoco_model, mujoco_data, target_objects, goal_spec,
                    num_pts=cfg.DATASET.num_pts,
                    max_num_target_objects=cfg.DATASET.max_num_target_objects,
                    max_num_distractor_objects=cfg.DATASET.max_num_distractor_objects,
                    use_virtual_structure_frame=cfg.DATASET.use_virtual_structure_frame,
                    ignore_distractor_objects=cfg.DATASET.ignore_distractor_objects,
                    ignore_rgb=cfg.DATASET.ignore_rgb
                )
                print(f"Successfully extracted {len(raw_datum['pcs'])} point clouds")
            except Exception as e:
                print(f"Error extracting point clouds: {e}")
                import traceback
                traceback.print_exc()
                return
            
            print("Converting to tensors...")
            sentence_str = tokenizer.convert_structure_params_to_natural_language(raw_datum["sentence"])
            print(f"Goal: {sentence_str}")
            
            datum = SemanticArrangementDataset.convert_to_tensors(raw_datum, tokenizer)
            print(f"Point cloud shapes: {datum['pcs'].shape}")
            print(f"Number of poses: {datum['goal_poses'].shape[0]}")
            
            # Create batch (same logic as single_datum_to_batch)
            batch = {}
            batch["pcs"] = datum["pcs"].to(device)[None, :, :, :].repeat(args.num_samples, 1, 1, 1)
            batch["sentence"] = datum["sentence"].to(device)[None, :].repeat(args.num_samples, 1)
            batch["type_index"] = datum["type_index"].to(device)[None, :].repeat(args.num_samples, 1)
            batch["position_index"] = datum["position_index"].to(device)[None, :].repeat(args.num_samples, 1)
            batch["pad_mask"] = datum["pad_mask"].to(device)[None, :].repeat(args.num_samples, 1)
            
            print("Running inference...")
            num_poses = datum["goal_poses"].shape[0]
            xs = sampler.sample(batch, num_poses)
            print(f"Generated {len(xs)} samples")
            
            print("Transforming point clouds to goal positions...")
            struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])
            new_obj_xyzs = move_pc_and_create_scene_simple(batch["pcs"], struct_pose, pc_poses_in_struct)
            print(f"Transformed point clouds shape: {new_obj_xyzs.shape}")
            
            print("Visualizing results...")
            visualize_batch_pcs(new_obj_xyzs, args.num_samples, limit_B=10, trimesh=True)
            print("=" * 50)
        else:
            # Use original H5 dataset
            dataset = SemanticArrangementDataset(split="test", tokenizer=tokenizer, **cfg.DATASET)
            
            data_idxs = np.random.permutation(len(dataset))
            for di in data_idxs:
                raw_datum = dataset.get_raw_data(di)
                print(tokenizer.convert_structure_params_to_natural_language(raw_datum["sentence"]))
                datum = dataset.convert_to_tensors(raw_datum, tokenizer)
                batch = dataset.single_datum_to_batch(datum, args.num_samples, device, inference_mode=True)

                num_poses = datum["goal_poses"].shape[0]
                xs = sampler.sample(batch, num_poses)

                struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])
                new_obj_xyzs = move_pc_and_create_scene_simple(batch["pcs"], struct_pose, pc_poses_in_struct)
                visualize_batch_pcs(new_obj_xyzs, args.num_samples, limit_B=10, trimesh=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='../configs/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='../configs/conditional_pose_diffusion.yaml',
                        type=str)
    parser.add_argument("--testing_data_config_file", help='config yaml file',
                        default='../configs/testing_data.yaml',
                        type=str)
    parser.add_argument("--checkpoint_id",
                        default="ConditionalPoseDiffusion",
                        type=str)
    parser.add_argument("--eval_mode",
                        default="infer",
                        type=str)
    parser.add_argument("--eval_random_seed",
                        default=42,
                        type=int)
    parser.add_argument("--num_samples",
                        default=10,
                        type=int)
    parser.add_argument("--use_mujoco",
                        action="store_true",
                        help="Use MuJoCo scene.xml instead of H5 dataset")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.base_config_file)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(base_cfg, cfg)

    testing_data_cfg = OmegaConf.load(args.testing_data_config_file)
    testing_data_cfg = OmegaConf.merge(base_cfg, testing_data_cfg)
    replace_config_for_testing_data(cfg, testing_data_cfg)

    main(args, cfg)


