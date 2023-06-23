import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['export OPENBLAS_NUM_THREADS']='1'

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import firedrake
import icepack
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    weertman_sliding_law as m
)
import icepack.models.friction
import icepack.plot
from firedrake import sqrt, inner, div, conditional
from firedrake.plot import triplot
import tqdm
import numpy as np


def melt_rate(melt, temp):
    Lx, Ly = 64e3, 64e3
    Nx = 64
    Ny = 64
    mesh = firedrake.RectangleMesh(Nx//2, Ny//2, Lx, Ly)

    Q = firedrake.FunctionSpace(mesh, "CG", 2)
    V = firedrake.VectorFunctionSpace(mesh, "CG", 2)

    x, y = firedrake.SpatialCoordinate(mesh)
    xfunc = firedrake.interpolate(x, Q)
    yfunc = firedrake.interpolate(y, Q)

    b_in, b_out = 200, -400
    sol_index = 0

    b = firedrake.interpolate(b_in - (b_in - b_out) * x / Lx, Q)

    s_in, s_out = 850, 50
    s0 = firedrake.interpolate(s_in - (s_in - s_out) * x / Lx, Q)

    h0 = firedrake.interpolate(s0 - b, Q)

    h_in = s_in - b_in
    δs_δx = (s_out - s_in) / Lx
    τ_D = -ρ_I * g * h_in * δs_δx
    # print(f"{1000 * τ_D} kPa")

    u_in, u_out = 20, 2400
    velocity_x = u_in + (u_out - u_in) * (x / Lx) ** 2

    u0 = firedrake.interpolate(firedrake.as_vector((velocity_x, 0)), V)

    T = firedrake.Constant(temp)
    A = icepack.rate_factor(T)

    expr = (0.95 - 0.05 * x / Lx) * τ_D / u_in**(1 / m)
    C = firedrake.interpolate(expr, Q)

    p_W = ρ_W * g * firedrake.max_value(0, h0 - s0)
    p_I = ρ_I * g * h0
    ϕ = 1 - p_W / p_I

    def weertman_friction_with_ramp(**kwargs):
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        s = kwargs["surface"]
        C = kwargs["friction"]

        p_W = ρ_W * g * firedrake.max_value(0, h - s)
        p_I = ρ_I * g * h
        ϕ = 1 - p_W / p_I
        return icepack.models.friction.bed_friction(
            velocity=u,
            friction=C * ϕ,
        )

    model_weertman = icepack.models.IceStream(friction=weertman_friction_with_ramp)
    opts = {"dirichlet_ids": [1], "side_wall_ids": [3, 4]}
    solver_weertman = icepack.solvers.FlowSolver(model_weertman, **opts)

    u0 = solver_weertman.diagnostic_solve(
        velocity=u0, thickness=h0, surface=s0, fluidity=A, friction=C
    )

    # Basal shear stress plot
    expr = -1e3 * C * ϕ * sqrt(inner(u0, u0)) ** (1 / melt - 1) * u0
    τ_b = firedrake.interpolate(expr, V)

    # Mass flux plot
    f = firedrake.interpolate(-div(h0 * u0), Q)

    # Take initial glacier state and project it forward until it reaches a steady state
    num_years = 250
    timesteps_per_year = 2 # 6 months

    δt = 1.0 / timesteps_per_year
    num_timesteps = num_years * timesteps_per_year 

    # Constant
    expr =  firedrake.Constant(1.5) - conditional(x / Lx > 0.5, melt, 0.0)  
    a = firedrake.interpolate(expr, Q)

    h = h0.copy(deepcopy=True)
    u = u0.copy(deepcopy=True)

    ice_stream_sol = np.zeros((Nx, Ny, 50))
    index_sol = 0
    for step in tqdm.trange(num_timesteps):        
        h = solver_weertman.prognostic_solve(
            δt,
            thickness=h,
            velocity=u,
            accumulation=a,
            thickness_inflow=h0,
        )
        s = icepack.compute_surface(thickness=h, bed=b)

        u = solver_weertman.diagnostic_solve(
            velocity=u,
            thickness=h,
            surface=s,
            fluidity=A,
            friction=C,
        )
        if step%10==0:            
            # Velocity
            test = np.vstack((xfunc.dat.data, yfunc.dat.data, u.dat.data[:,0])).T
            test = test[(-test[:,1]).argsort()]
            test = test.reshape((Nx+1, Ny+1, 3))
            for j in range(65):
                curr = np.vstack((test[j,:,0], test[j,:,1], test[j,:,2])).T
                curr = curr[curr[:,0].argsort()]
                curr = curr.T
                test[j,:,0] = curr[0,:]
                test[j,:,1] = curr[1,:]
                test[j,:,2] = curr[2,:]
            ice_stream_sol[:,:,index_sol]  = test[:Nx, :Ny, 2]
            index_sol+=1

    np.save("../../../mnt/mwt/Data/much_data/melt_{}_temp_{}.npy".format(melt, temp), ice_stream_sol)
    return np.amax(ice_stream_sol[:, :, -1])



temp_vals = []

# temps = np.linspace(170.0, 268.0, 99) # try np.linspace(1.13969697 , 7.0, 84) because it stopped
# # melts = np.linspace(0.01, 7.0, 100)
# melts = np.linspace(3.39909091 , 7.0, 52)

temps = np.linspace(80.0, 179.0, 100)
melts = np.linspace(0.01, 2, 100)
for melt in melts:
    for temp in temps:
        print(melt, temp)
        temp_vals.append(melt_rate(melt, temp))

        
# finer_temps = np.linspace(245.0, 265.0, 50)
# finer_melts = np.linspace(0.01, 2.0, 50)

# for m in finer_melts:
#     for t in finer_temps:
#         print(m, t)
#         temp_vals.append(melt_rate(m, t))
        
np.save("../../../mnt/mwt/Data/much_data/2D_vals.npy", temp_vals)
