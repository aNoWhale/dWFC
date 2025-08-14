import os
import sys
from typing import Callable, List
from pathlib import Path
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/WFC 目录下）
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
import jax
import jax.numpy as np
import json
import meshio


from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem.utils import com


# Define constitutive relationship.
class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad)
    # to obtain the 'P_fn' function.
    def get_tensor_map(self):

        def psi(F):
            E = 10.0
            nu = 0.3
            mu = E / (2.0 * (1.0 + nu))
            kappa = E / (3.0 * (1.0 - 2.0 * nu))
            J = np.linalg.det(F)
            Jinv = J ** (-2.0 / 3.0)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.0) * (Jinv * I1 - 3.0) + (kappa / 2.0) * (J - 1.0) ** 2.0
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress



class RVEcritic:
    def __init__(self,problem:Problem,load_fns:List[Callable],dirichlet_bc_fns:List[Callable],ele_type='TET4',vec=3,dim=3,data_path="data/tile"):
        self.ele_type = ele_type
        self.uninstanted_problem = lambda mesh: problem(mesh,vec=vec,dim=dim,ele_type=ele_type,location_fns=load_fns,dirichlet_bc_fns=dirichlet_bc_fns)
        self.data_path = data_path
        folder_path = Path(self.data_path)
        folder_path.mkdir(parents=True, exist_ok=True)  # 自动处理父目录
        
    def saveProperties(self,):
        folder_path = Path(self.data_path)
        folder_path.mkdir(parents=True, exist_ok=True)  # 自动处理父目录

    def critic(self,mesh_file:str,solver_options={"petsc_solver":{},}):
        base_name, ext = os.path.splitext(mesh_file)
        meshio_mesh:meshio.Mesh=meshio.read(mesh_file)
        mesh = Mesh(meshio_mesh.points,meshio_mesh.cells_dict[get_meshio_cell_type(self.ele_type)],self.ele_type)
        del meshio_mesh
        problem = self.uninstanted_problem(mesh)
        sol_list=solver(problem,solver_options=solver_options)
        sol = sol_list[0]
        vtu_path = os.path.join(self.data_path, f'vtk/{base_name}.vtu')
        save_sol(problem.fe,np.hstack((sol,np.zeros((len(sol),1)))),vtu_path,cell_infos=[])

    def postprocess(self,sol,problem:Problem):
        pass

    


if __name__ == "__main__":
    pass

