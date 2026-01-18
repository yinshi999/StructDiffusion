import trimesh
# load mesh (auto-detect .obj, .stl, etc.)
mesh = trimesh.load('/home/patricia/Desktop/Learn/codes/StructDiffusion/mujoco/meshes/assets/bowl.obj', force='mesh')

# quick visual check in trimesh (opens an external viewer)
mesh.show()

# sample N points on the surface (uniform by area)
points, face_indices = trimesh.sample.sample_surface(mesh, count=200000)

# export as PLY for visualizing in CloudCompare / Open3D
pc = trimesh.points.PointCloud(points)
import pdb; pdb.set_trace()
pc.export('cloud_from_obj.ply')