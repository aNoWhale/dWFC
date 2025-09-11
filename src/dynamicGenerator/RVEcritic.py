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
# Generally, JAX-FEM solves -div.(f(u_grad,alpha_1,alpha_2,...,alpha_N)) = b.
# Here, we have f(u_grad,alpha_1,alpha_2,...,alpha_N) = sigma(u_grad, theta),
# reflected by the function 'stress'. The functions 'custom_init'and 'set_params'
# override base class methods. In particular, set_params sets the design variable theta.
class Elasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    def custom_init(self):
        # Override base class method.
        # Set up 'self.fe.flex_inds' so that location-specific TO can be realized.
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))

        # full_params = np.ones((self.fe.num_cells, params.shape[1]))
        # full_params = full_params.at[self.fe.flex_inds].set(params)
        # thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        # self.full_params = full_params
        # self.internal_vars = [thetas]
        # 注意shape是cell，n_quad,parashape
        
    def get_tensor_map(self):
        def stress(u_grad,id=None):
            # 定义 6×6 刚度矩阵 C（单位：Pa）
            # 示例：正交各向异性材料（可替换为任意 C）
            C = np.array([
                [120e3,  40e3,  40e3,     0,     0,     0],
                [ 40e3, 120e3,  40e3,     0,     0,     0],
                [ 40e3,  40e3, 120e3,     0,     0,     0],
                [     0,     0,     0, 30e3,     0,     0],
                [     0,     0,     0,     0, 30e3,     0],
                [     0,     0,     0,     0,     0, 30e3]
            ])
            # 3D 各向异性弹性本构（Voigt 表示）
            # u_grad: (3, 3) 位移梯度
            epsilon = 0.5 * (u_grad + u_grad.T)  # 应变张量 (3,3)

            # Voigt 向量形式：ε = [ε11, ε22, ε33, 2ε23, 2ε13, 2ε12]
            epsilon_voigt = np.array([
                epsilon[0, 0],
                epsilon[1, 1],
                epsilon[2, 2],
                2 * epsilon[1, 2],
                2 * epsilon[0, 2],
                2 * epsilon[0, 1]
            ])
            sigma_voigt = C @ epsilon_voigt  # (6,)
            # 转回张量形式
            sigma = np.array([
                [sigma_voigt[0], sigma_voigt[5], sigma_voigt[4]],
                [sigma_voigt[5], sigma_voigt[1], sigma_voigt[3]],
                [sigma_voigt[4], sigma_voigt[3], sigma_voigt[2]]
            ])

            return sigma
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 0., 100.])
        return [surface_map]


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

