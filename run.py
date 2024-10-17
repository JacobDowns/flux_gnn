# -*- coding: utf-8 -*-

"""
Steady state advection.
"""

import matplotlib.pyplot as plt
import numpy as np
import firedrake as fd


# domain parameters
order = 0
n = 2
v_inlet = fd.Constant((5.0, 0.))


mesh = fd.Mesh('dev_experiments/meshes/mesh1.msh')

# 3) Setting problem (FunctionSpace, Init.Bound.Condition, VariationalForms)

DG1 = fd.FunctionSpace(mesh, "DG", order)
vDG1 = fd.VectorFunctionSpace(mesh, "DG", order)


phi = fd.TestFunction(DG1)
c = fd.TrialFunction(DG1)

c0 = fd.Function(DG1, name="c")
dt = np.sqrt(1.)
Dt = fd.Constant(dt)
c_mid = 0.5 * (c + c0)  # Crank-Nicolson timestepping
n = fd.FacetNormal(mesh)

# advective velocity
velocity = fd.Function(vDG1, name='velocity')
velocity.interpolate(v_inlet)
vnorm = fd.sqrt(fd.dot(velocity, velocity))

# upwind term
vn = 0.5*(fd.dot(velocity, n) + abs(fd.dot(velocity, n)))
# transient term
F_t = phi*((c - c0))*fd.dx

#g = 1.0 
g = fd.Function(DG1)
gx = fd.Function(DG1)
gy = fd.Function(DG1)
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
F = F_t + F_a #- g*phi*fd.dx

# CFL
outfile = fd.File("plots/adr_dg.pvd")

limiter = fd.VertexBasedLimiter(DG1)  # Kuzmin slope limiter

c_ = fd.Function(DG1, name="c")
problem = fd.LinearVariationalProblem(fd.lhs(F), fd.rhs(F), c_)
solver = fd.LinearVariationalSolver(problem)


thetas = np.linspace(0.,2*np.pi, 10)
v_mags = np.linspace(0.1, 1., 10)
thetas, v_mags = np.meshgrid(thetas, v_mags)
thetas = thetas.flatten()
v_mags = v_mags.flatten()

def solve(i):
    
    theta = thetas[i]
    v_mag = v_mags[i]
    out_file = f'dev_experiments/fluxgnn/output/out_{i}.h5'
    
    with fd.CheckpointFile(out_file, 'w') as afile:
        
        
       
        v = fd.Constant((v_mag*np.cos(theta), v_mag*np.sin(theta)))
        velocity.interpolate(v)
        
        
        afile.save_mesh(mesh)
        afile.save_function(velocity)
        
        #c0.assign(0.*c0)
        c0.assign(g)

        # initialize timestep
        t = 0.0
        it = 0
        dt = 0.25
        Dt.assign(dt)
        sim_time = 10. 

        
        while t < sim_time:
            
            afile.save_function(c0, idx=it)
            
            # move next time step
            it += 1
            t += dt
            #print("* iteration= {:4d}, dtime= {:8.6f}, time={:8.6f}".format(it, dt, t))
            solver.solve()

            # update sol.
            c0.assign(c_)
            print(fd.assemble(c0*fd.dx))
            #outfile.write(c0, time=t)

for i in range(len(thetas)):
    print(i)
    solve(i)


