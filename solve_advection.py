# -*- coding: utf-8 -*-

"""
Solve a time dependent advection problem with some simple velocity fields. 
"""

import matplotlib.pyplot as plt
import numpy as np
import firedrake as fd


mesh = fd.Mesh('data/mesh.msh')

V = fd.FunctionSpace(mesh, "DG", 0)
Q = fd.VectorFunctionSpace(mesh, "DG", 0)


phi = fd.TestFunction(V)
c = fd.TrialFunction(V)

c0 = fd.Function(V, name="c")
Dt = fd.Constant(1.)
c_mid = 0.5 * (c + c0) 
n = fd.FacetNormal(mesh)

# advective velocity
velocity = fd.Function(Q, name='velocity')
velocity.interpolate(fd.Constant((0.0, 0.0)))
vnorm = fd.sqrt(fd.dot(velocity, velocity))

# upwind term
vn = 0.5*(fd.dot(velocity, n) + abs(fd.dot(velocity, n)))
# transient term
F_t = phi*((c - c0))*fd.dx

#  Intial condition 
g = fd.Function(V)
gx = fd.Function(V)
gy = fd.Function(V)
x = fd.SpatialCoordinate(mesh)
gx.interpolate(x[0])
gy.interpolate(x[1])
indexes1 = np.logical_and(gx.dat.data > 45., gx.dat.data < 55.)
indexes2 = np.logical_and(gy.dat.data > 45., gy.dat.data < 55.)
indexes = np.logical_and(indexes1, indexes2)
g.dat.data[indexes] = 1.
c0.assign(g)

# advection form
F_a = Dt*((phi('+') - phi('-'))*(vn('+')*c_mid('+') -
                                 vn('-')*c_mid('-'))*fd.dS
          + phi*c_mid*vn*fd.ds
          - fd.inner(fd.grad(phi), velocity*c_mid)*fd.dx)

# full weak form
F = F_t + F_a 

c_ = fd.Function(V, name="c")
problem = fd.LinearVariationalProblem(fd.lhs(F), fd.rhs(F), c_)
solver = fd.LinearVariationalSolver(problem)

# Velocity field parameters
thetas = np.linspace(0.,2*np.pi, 8)
v_mags = np.linspace(0.1, 1., 8)
thetas, v_mags = np.meshgrid(thetas, v_mags)
thetas = thetas.flatten()
v_mags = v_mags.flatten()


def solve(i):
    # Solve advection problem, write results
        
    theta = thetas[i]
    v_mag = v_mags[i]
    out_file = f'data/outputs/out_{i}.h5'
    
    with fd.CheckpointFile(out_file, 'w') as afile:
       
        v = fd.Constant((v_mag*np.cos(theta), v_mag*np.sin(theta)))
        velocity.interpolate(v)
        
        afile.save_mesh(mesh)
        afile.save_function(velocity)
        c0.assign(g)

        t = 0.0
        it = 0
        dt = 0.5
        Dt.assign(dt)
        sim_time = 10. 

        while t < sim_time:
            
            afile.save_function(c0, idx=it)
            it += 1
            t += dt
            solver.solve()
            c0.assign(c_)
            #print(fd.assemble(c0*fd.dx))
            
for i in range(len(thetas)):
    print(i)
    solve(i)


