
import matplotlib.pyplot as plt
import numpy as np
import firedrake as fd

out_dir = 'data/'

V = []
C = []
for i in range(64):
    in_file = f'data/outputs/out_{i}.h5'
    with fd.CheckpointFile(in_file, 'r') as afile:
        
        print(i)
        mesh = afile.load_mesh()
        
        if i == 0:
            V_cg = fd.FunctionSpace(mesh, 'CG', 1)
            triangles = V_cg.cell_node_list
            coordinates = mesh.coordinates.dat.data
            
            np.save(f'{out_dir}/coordinates.npy', coordinates)
            np.save(f'{out_dir}/triangles.npy', triangles)
        
        v = afile.load_function(mesh, "velocity")
        V.append(v.dat.data)
        
        C_i = []
        for j in range(20):
            c = afile.load_function(mesh, "c", idx=j)
            C_i.append(c.dat.data)
            
        C_i = np.array(C_i)
        #plt.scatter(coordinates[:,0], coordinates[:,1], c=C_i[-1])
        #plt.show()
        C.append(C_i)
        
V = np.array(V)
C = np.array(C)      

np.save(f'{out_dir}/V.npy', V)
np.save(f'{out_dir}/C.npy', C)