"""
Quick test script to verify MuJoCo point cloud extraction works.
"""

import os
import sys
import mujoco
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from StructDiffusion.utils.mujoco_pc_extraction import extract_object_point_clouds_from_mujoco

def main():
    print("Testing MuJoCo point cloud extraction...")
    
    # Load MuJoCo scene
    mujoco_scene_path = os.path.join(os.path.dirname(__file__), "..", "mujoco", "scene.xml")
    if not os.path.exists(mujoco_scene_path):
        print(f"ERROR: MuJoCo scene not found at: {mujoco_scene_path}")
        return
    
    print(f"Loading scene from: {mujoco_scene_path}")
    model = mujoco.MjModel.from_xml_path(mujoco_scene_path)
    data = mujoco.MjData(model)
    
    print(f"Scene loaded. Number of bodies: {model.nbody}")
    print(f"Number of geoms: {model.ngeom}")
    
    # List all bodies
    print("\nAvailable bodies:")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            print(f"  - {body_name}")
    
    # Test extraction
    target_objects = ["box", "sphere", "bowl"]
    print(f"\nExtracting point clouds for: {target_objects}")
    
    try:
        pc_data = extract_object_point_clouds_from_mujoco(
            model, data, target_objects, num_pts=1024, ignore_rgb=True
        )
        
        print(f"\nSuccess! Extracted {len(pc_data['obj_pcs'])} point clouds:")
        for i, (obj_name, pc) in enumerate(zip(target_objects, pc_data['obj_pcs'])):
            print(f"  {obj_name}: shape {pc.shape}, dtype {pc.dtype}")
            print(f"    Center: {pc.mean(dim=0).numpy()}")
        
        print("\nPoint cloud extraction test PASSED!")
        
    except Exception as e:
        print(f"\nERROR during extraction: {e}")
        import traceback
        traceback.print_exc()
        print("\nPoint cloud extraction test FAILED!")

if __name__ == "__main__":
    main()


