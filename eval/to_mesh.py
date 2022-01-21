import numpy as np
import trimesh
from skimage import measure
import open3d as o3d
from scipy.spatial.distance import cdist


voxel = np.loadtxt('8_pred.txt')
voxels = voxel[:,3]


voxels[voxels>-1] = 0
voxels[voxels==-1] = 1

voxels[voxels==0]= -1 



voxels = voxels.reshape((64, 64, 64))


vertices, faces, normals, _ = measure.marching_cubes_lewiner(voxels, level=0)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)



o3d_mesh = mesh.as_open3d
o3d_mesh.translate((0,0,0), relative=False)
R = o3d_mesh.get_rotation_matrix_from_xyz((0,np.pi/2,0))
o3d_mesh.rotate(R, o3d_mesh.get_center())



o3d.io.write_triangle_mesh('mesh.ply',o3d_mesh)




ply = trimesh.load_mesh('mesh.ply')


vertices = ply.vertices
faces = ply.faces




d =  np.loadtxt('8_pred.txt')
data = d[d[:,3]>-1]
point = data[:,:3]
color = data[:,3:].clip(0,1)



dis_mat = cdist(vertices[faces[:,1]],point) 
index_pre = np.argmin(dis_mat, axis=1)


A = np.full((color[index_pre].shape[0], 1), 255)
face_color = np.concatenate((color[index_pre]*255, A), axis=1)


ply.visual.face_colors = face_color



ply.export('demo.ply')




