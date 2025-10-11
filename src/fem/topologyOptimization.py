# Import some useful modules.
import sys
import numpy as onp

import os


os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import meshio
import jax
import jax_smi
jax_smi.initialise_tracking()
# jax.config.update('jax_disable_jit', True)
# jax.config.update("jax_enable_x64", True)
from functools import partial
import jax.numpy as np
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/fem 目录下）
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh, box_mesh_gmsh
from jax_fem.mma import optimize

from src.fem.SigmaInterpreter import SigmaInterpreter
from src.WFC.TileHandler import TileHandler
from src.WFC.adjacencyCSR import build_hex8_adjacency_with_meshio
from src.WFC.stableWaveFunctionCollapse import waveFunctionCollapse


# Do some cleaning work. Remove old solution files.
# data_path = os.path.join(os.path.dirname(__file__), 'data')
data_path ='data'

files = glob.glob(os.path.join(data_path, f'vtk/*'))
# for f in files:
#     os.remove(f)

# Define constitutive relationship.
# Generally, JAX-FEM solves -div.(f(u_grad,alpha_1,alpha_2,...,alpha_N)) = b.
# Here, we have f(u_grad,alpha_1,alpha_2,...,alpha_N) = sigma(u_grad, theta),
# reflected by the function 'stress'. The functions 'custom_init'and 'set_params'
# override base class methods. In particular, set_params sets the design variable theta.
class Elasticity(Problem):
    def custom_init(self,*additional_info):
        # Override base class method.
        # Set up 'self.fe.flex_inds' so that location-specific TO can be realized.
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))
        self.sigma:SigmaInterpreter = self.additional_info[0]

    def get_tensor_map(self):
        def stress(u_grad, weights):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            # Emax = 70.e3
            # Emin = 1e-3*Emax
            # nu = 0.3
            # penal = 3.
            # E = Emin + (Emax - Emin)*theta[0]**penal
            # epsilon = 0.5*(u_grad + u_grad.T)
            # eps11 = epsilon[0, 0]
            # eps22 = epsilon[1, 1]
            # eps12 = epsilon[0, 1]
            # sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22)
            # sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
            # sig12 = E/(1 + nu)*eps12
            # sigma = np.array([[sig11, sig12], [sig12, sig22]])

            return self.sigma(u_grad,weights)
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 0., -1e3])
        return [surface_map]

    def set_params(self, params):
        # Override base class method.
        # full_params = np.ones((self.fe.num_cells, params.shape[1])) #theta的话1(cells, 1), tile(cells,tilesnum)
        # full_params = full_params.at[self.fe.flex_inds].set(params) 
        # weights = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1) #(cells, quads, tilesnum)
        # self.full_params = full_params
        # self.internal_vars = [weights] #[(cells,tilesnum)] theta的话(cells,1)
        weights = np.ones((self.fe.num_cells, self.fe.num_quads, params.shape[1]))
        weights = weights.at[self.fe.flex_inds].set(np.repeat(params[:, None, :], self.fe.num_quads, axis=1))
        self.internal_vars = [weights]

    def compute_compliance(self, sol):
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)
        u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, dim)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points) # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val

# Specify mesh-related information. We use first-order quadrilateral element.
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly, Lz = 10., 10., 10.
Nx, Ny, Nz = 10, 10, 10
if not os.path.exists("data/msh/box.msh"):
    meshio_mesh = box_mesh_gmsh(Nx=Nx,Ny=Ny,Nz=Nz,domain_x=Lx,domain_y=Ly,domain_z=Lz,data_dir="data",ele_type=ele_type)
else:
    meshio_mesh = meshio.read("data/msh/box.msh")
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
# Define boundary conditions and values.
def fixed_location(point):
    return np.isclose(point[2], 0., atol=0.1+1e-5)

def load_location(point):
    return np.isclose(point[2], Lz, atol=0.1+1e-5)

def dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[fixed_location]*3, [0, 1, 2], [dirichlet_val]*3]

location_fns = [load_location]

tileHandler = TileHandler(typeList=['solid','void','weird'], direction=(('back',"front"),("left","right"),("top","bottom")))
tileHandler.setConnectiability(fromTypeName='solid',toTypeName=["void","weird"],direction="isotropy",value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='void', toTypeName="weird", direction="isotropy",value=1,dual=True)
tileHandler.selfConnectable(typeName=['solid',"void","weird"],value=1)

# Define forward problem.
problem = Elasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns,
                     additional_info=(SigmaInterpreter(typeList=tileHandler.typeList,folderPath="data/C",debug=False),))

# Apply the automatic differentiation wrapper.
# This is a critical step that makes the problem solver differentiable.
fwd_pred = ad_wrapper(problem, solver_options={'petsc_solver': {}}, adjoint_solver_options={'petsc_solver': {}})

# Define the objective function 'J_total(theta)'.
# In the following, 'sol = fwd_pred(params)' basically says U = U(theta).
def J_total(params):
    # J(u(theta), theta)
    sol_list = fwd_pred(params)
    compliance = problem.compute_compliance(sol_list[0])
    return compliance, sol_list


# Output solution files to local disk
outputs = []
def output_sol(params, obj_val,sol_list):
    print(f"\nOutput solution - params.shape:{params.shape}")
    # sol_list = fwd_pred(params)

    sol = sol_list[0]
    vtu_path = os.path.join(data_path, f'vtk/sol_{output_sol.counter:03d}.vtu')
    save_sol(problem.fe, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path, cell_infos=[(f'theta{i}', params[:, i]) for i in range(params.shape[-1])])
    print(f"compliance = {obj_val}")
    outputs.append(obj_val)
    output_sol.counter += 1
output_sol.counter = 0


# Prepare J_total and dJ/d(theta) that are required by the MMA optimizer.
def objectiveHandle(rho):
    # MMA solver requires (J, dJ) as inputs
    # J has shape ()
    # dJ has shape (...) = rho.shape
    (J, sol), dJ = jax.value_and_grad(J_total,has_aux=True)(rho)
    output_sol(rho, J, sol)
    return J, dJ

vf0=0.35
vf1=0.35

# Prepare g and dg/d(theta) that are required by the MMA optimizer.
numConstraints = 2
def consHandle(rho, epoch):
    # MMA solver requires (c, dc) as inputs
    # c should have shape (numConstraints,)
    # dc should have shape (numConstraints, ...)
    def computeGlobalVolumeConstraint(rho):
        g = np.mean(rho)/vf1 - 1.
        return g
    c0, gradc0 = jax.value_and_grad(lambda rho: np.mean(rho[...,0])/vf0 - 1.)(rho)
    c1, gradc1 = jax.value_and_grad(lambda rho: np.mean(rho[...,1])/vf1 - 1.)(rho)

    c=np.array([c0,c1])
    gradc=np.array([gradc0,gradc1])
    print(f"c.shape:{c.shape}")
    print(f"gradc.shape:{gradc.shape}")
    c = c.reshape((-1,))
    return c, gradc

adj=build_hex8_adjacency_with_meshio(mesh=meshio_mesh)
wfc=lambda prob: waveFunctionCollapse(prob, adj, tileHandler)

# Finalize the details of the MMA optimizer, and solve the TO problem.
optimizationParams = {'maxIters':51, 'movelimit':0.1}
rho_ini = np.ones((Nx,Ny,Nz,tileHandler.typeNum),dtype=np.float64).reshape(-1,tileHandler.typeNum)/tileHandler.typeNum
print(f"rho_ini.shape{rho_ini.shape}")
rho_oped,J_list=optimize(problem.fe, rho_ini, optimizationParams, objectiveHandle, consHandle, numConstraints,tileNum=tileHandler.typeNum,WFC=wfc)
np.save("cache/rho_oped",rho_oped)
print(f"As a reminder, compliance = {J_total(np.ones((len(problem.fe.flex_inds), 1)))} for full material")

# Plot the optimization results.
obj = onp.array(outputs)
plt.figure(figsize=(10, 8))
plt.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
plt.xlabel(r"Optimization step", fontsize=20)
plt.ylabel(r"Objective value", fontsize=20)
plt.tick_params(labelsize=20)
plt.tick_params(labelsize=20)
plt.show()
