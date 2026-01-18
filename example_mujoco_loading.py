"""
Example: Loading data from MuJoCo scene instead of h5 files

This demonstrates how to use the new get_raw_data_from_mujoco() method.
"""

import mujoco
import numpy as np
import torch
import sys
import os

import pdb

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from StructDiffusion.data.semantic_arrangement import SemanticArrangementDataset
from StructDiffusion.language.tokenizer import Tokenizer


def example_mujoco_loading():
    """
    Example of loading data from a MuJoCo scene
    """
    print("="*70)
    print("Loading data from MuJoCo scene")
    print("="*70)
    
    # 1. Load MuJoCo scene
    scene_path = "mujoco/scene.xml"
    print(f"\n1. Loading MuJoCo model from: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    print(f"   ✓ Loaded model with {model.nbody} bodies")
    
    # 2. Print available bodies
    print(f"\n2. Available bodies in scene:")
    for i in range(model.nbody):
        body_name_adr = model.name_bodyadr[i]
        body_name = model.names[body_name_adr:].decode('utf-8').split('\x00')[0]
        if body_name:  # Skip empty names
            print(f"   - {body_name}")
    
    # 3. Define target objects and goal specification
    print(f"\n3. Defining task:")
    target_object_names = ["bowl", "sphere", "box"]
    other_object_names = []  # No distractors for this example
    
    # Goal specification (same format as h5 files)
    goal_specification = {
        "shape": {
            "type": "circle",
            "rotation": [0, 0, 0],
            "position": [0.4, 0.0, 0.6],
            "radius": 0.15
        },
        "rearrange": {
            "objects": target_object_names
        },
        "anchor": {
            "objects": []
        },
        "distract": {
            "objects": []
        }
    }
    print(f"   Target objects: {target_object_names}")
    print(f"   Goal: Arrange in {goal_specification['shape']['type']} at {goal_specification['shape']['position']}")
    
    # 4. Create dataset instance
    print(f"\n4. Creating dataset instance:")
    tokenizer = Tokenizer("/home/patricia/Desktop/Learn/codes/StructDiffusion/testing_data/type_vocabs_coarse.json")
    
    dataset = SemanticArrangementDataset(
        data_roots=[],  # Empty since we're not using h5 files
        index_roots=[],
        split="test",
        tokenizer=tokenizer,
        max_num_target_objects=11,
        max_num_distractor_objects=5,
        num_pts=1024,
        use_virtual_structure_frame=True,
        ignore_distractor_objects=True,
        ignore_rgb=True,
        debug=False
    )
    print(f"   ✓ Dataset created")
    
    # 5. Load data from MuJoCo scene
    print(f"\n5. Loading point clouds from MuJoCo scene:")
    scene_xml_dir = "mujoco"
    
    datum = dataset.get_raw_data_from_mujoco(
        model=model,
        data=data,
        target_object_names=target_object_names,
        goal_specification=goal_specification,
        other_object_names=other_object_names,
        scene_xml_dir=scene_xml_dir,
        inference_mode=True,
        shuffle_object_index=False
    )
    
    print(f"   ✓ Loaded {len(datum['pcs'])} point clouds")
    for i, pc in enumerate(datum['pcs']):
        if isinstance(pc, torch.Tensor):
            # pdb.set_trace()
            print(f"      Object {i}: {pc.shape} points")
    
    # 6. Convert to tensors
    print(f"\n6. Converting to tensor format:")
    tensors = SemanticArrangementDataset.convert_to_tensors(datum, tokenizer)
    print(f"   ✓ Tensor shapes:")
    print(f"      pcs: {tensors['pcs'].shape}")
    print(f"      sentence: {tensors['sentence'].shape}")
    print(f"      goal_poses: {tensors['goal_poses'].shape}")
    print(f"      type_index: {tensors['type_index'].shape}")
    print(f"      position_index: {tensors['position_index'].shape}")
    print(f"      pad_mask: {tensors['pad_mask'].shape}")
    
    # 7. Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print("="*70)
    print("✓ Successfully loaded point clouds from MuJoCo scene")
    print("✓ Data is ready for the diffusion model")
    print()
    print("Key differences from h5 loading:")
    print("  • Object poses read from MuJoCo data (data.xpos, data.xmat)")
    print("  • Mesh paths extracted from MuJoCo model")
    print("  • Point clouds sampled from mesh files or primitives")
    print("  • Goal specification provided manually (not from h5)")
    print()
    print("Next steps:")
    print("  • Set goal poses based on your task requirements")
    print("  • Run inference with the loaded data")
    print("  • Update MuJoCo scene with predicted poses")
    print("="*70)


if __name__ == "__main__":
    try:
        example_mujoco_loading()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. MuJoCo installed (pip install mujoco)")
        print("  2. Scene file at mujoco/scene.xml")
        print("  3. Tokenizer vocab files in testing_data/")
        import traceback
        traceback.print_exc()
