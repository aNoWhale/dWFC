# Import some useful modules.
import sys

import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as onp

import glob
import matplotlib
# 强制使用有头后端（Tkinter，跨平台兼容性好）
# 如果Tkinter不可用，可以尝试其他后端：Qt5Agg, GTK3Agg, WXAgg等
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
print(f"{matplotlib.get_backend()}")
from pathlib import Path
import meshio
import jax
import jax_smi
jax_smi.initialise_tracking()
# jax.config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)
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

from src.WFC.TileHandler_JAX import TileHandler

from pathlib import Path
def create_directory_if_not_exists(directory_path):
    """
    如果目录不存在则创建目录
    
    Args:
        directory_path: 目录路径（可以是字符串或Path对象）
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"目录 '{directory_path}' 创建成功")
        else:
            print(f"目录 '{directory_path}' 已存在")
    except Exception as e:
        print(f"创建目录 '{directory_path}' 时出错: {e}")


# Do some cleaning work. Remove old solution files.
# data_path = os.path.join(os.path.dirname(__file__), 'data')
print(f"{jax.devices()}")
create_directory_if_not_exists("data")
data_path ='data'

files = glob.glob(os.path.join(data_path, f'vtk/*'))
create_directory_if_not_exists("data/npy")
create_directory_if_not_exists("data/vtk")
create_directory_if_not_exists("data/msh")
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
        self.sigmaInterpreter:SigmaInterpreter = self.additional_info[0]


    def get_tensor_map(self):
        def stress(u_grad, weights):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            return self.sigmaInterpreter(u_grad, weights)
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., -10]) #-0.3
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
    

    def compute_von_mises(self, sol):
        # 停止梯度：后处理无需梯度跟踪
        weights = jax.lax.stop_gradient(self.internal_vars[0])  # (num_cells, num_quads, tiletypes)
        u_grad = jax.lax.stop_gradient(self.fe.sol_to_grad(sol))  # 4维：(num_cells, num_quads, dim, dim)
        u_grad_shape=u_grad.shape
        weights_shape=weights.shape
        sigma = jax.vmap(self.sigmaInterpreter,in_axes=(0,0),out_axes=0)(u_grad.reshape(-1,u_grad_shape[-2],u_grad_shape[-1]), weights.reshape(-1,weights_shape[-1]))  # 4维：(num_cells, num_quads, dim, dim)
        sigma = sigma.reshape(u_grad_shape)
        # 二维形式的 von Mises 应力计算
        # 应力张量 sigma 的形状：(num_cells, num_quads, 2, 2)
        # 提取应力分量
        sigma_xx = sigma[..., 0, 0]  # (num_cells, num_quads)
        sigma_yy = sigma[..., 1, 1]  # (num_cells, num_quads)
        sigma_xy = sigma[..., 0, 1]  # (num_cells, num_quads)
        # 二维平面应力 von Mises 公式
        vm_gauss = np.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3*sigma_xy**2)
        
        # 使用高斯点积分权重计算单元平均值
        cells_JxW = self.JxW[:, 0, :]  # 形状：(num_cells, num_quads)
        cell_volumes = np.sum(cells_JxW, axis=1)  # 形状：(num_cells,)
        cell_vm = np.sum(vm_gauss * cells_JxW, axis=1) / cell_volumes  # 形状：(num_cells,)
        
        # 计算全局平均值
        total_volume = np.sum(cells_JxW)
        avg_vm = np.sum(vm_gauss * cells_JxW) / total_volume  # 标量
        
        return {
            'cell_von_mises': cell_vm,
            'avg_von_mises': avg_vm,
            'gauss_von_mises': vm_gauss
        }




# Specify mesh-related information. We use first-order quadr
# ilateral element.
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 60., 30.
Nx,Ny = 60, 30
kernel_size = 3
create_directory_if_not_exists("data/msh")
mshname=f"L{Lx}{Ly}N{Nx}{Ny}.msh"
mshname4ad=f"L{Lx}{Ly}N{Nx}{Ny}ad.msh"

if not os.path.exists(f"data/msh/{mshname}"):
    meshio_mesh = rectangle_mesh(Nx=Nx*kernel_size, Ny=Ny*kernel_size, domain_x=Lx, domain_y=Ly)
    meshio_mesh4ad = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    meshio.write(mshname,meshio_mesh)
    meshio.write(mshname4ad,meshio_mesh4ad)

else:
    meshio_mesh = meshio.read(f"data/msh/{mshname}")
    meshio_mesh4ad = meshio.read(f"data/msh/{mshname4ad}")

mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
mesh4ad = Mesh(meshio_mesh4ad.points, meshio_mesh4ad.cells_dict[cell_type])

# Define boundary conditions and values.


"""bend"""
def fixed_location(point):
    return np.logical_or(np.isclose(point[0], 0., atol=1+1e-5),
                          np.isclose(point[0], Lx, atol=1+1e-5))

def load_location(point):
    return np.logical_and(np.isclose(point[0], Lx/2, atol=0.1*Lx+1e-5),
                          np.isclose(point[1], Ly, atol=0.1*Ly+1e-5))



def dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]

location_fns = [load_location]

# tileHandler = TileHandler(typeList=['a','b','c','d','e','void'],direction=(('y+',"y-"),("x-","x+"),))
# tileHandler.selfConnectable(typeName="e",direction='isotropy',value=1)

# tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','d','b'],direction='y-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','d','a'],direction='x-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','b','a'],direction='x+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='a',toTypeName='c',direction='y+',value=1,dual=True)

# tileHandler.setConnectiability(fromTypeName='a',toTypeName='b',direction='x-',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='a',toTypeName='d',direction='x+',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='a',toTypeName=['a','b','d','e'],direction='y+',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='a',toTypeName=['a'],direction='y-',value=-1,dual=True)



# tileHandler.setConnectiability(fromTypeName='b',toTypeName=['e','d','a','c'],direction='x-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='b',toTypeName=['e','d','a','b'],direction='y+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='b',toTypeName=['e','d','c','b'],direction='y-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='b',toTypeName='d',direction='x+',value=1,dual=True)

# tileHandler.setConnectiability(fromTypeName='b',toTypeName='c',direction='y+',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='b',toTypeName='c',direction='y-',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='b',toTypeName=['a','b','c','e'],direction='x+',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='b',toTypeName=['b'],direction='x-',value=-1,dual=True)



# tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','d','a','b'],direction='y+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','d','a','c'],direction='x-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','a','b','c'],direction='x+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName='a',direction='y-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName='b',direction='x-',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName='d',direction='x+',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName=['c'],direction='y+',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName=['b','c','d','e'],direction='y-',value=-1,dual=True)

# tileHandler.setConnectiability(fromTypeName='d',toTypeName=['e','c','a','b'],direction='x+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='d',toTypeName=['e','a','b','d'],direction='y+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='d',toTypeName=['e','c','b','d'],direction='y-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='d',toTypeName='b',direction='x-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='d',toTypeName='c',direction='y+',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='d',toTypeName='c',direction='y-',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='d',toTypeName=['d'],direction='x+',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='d',toTypeName=['a','c','d','e'],direction='x-',value=-1,dual=True)

# tileHandler.setConnectiability(fromTypeName='e',toTypeName='a',direction='y+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='e',toTypeName='b',direction='x+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='e',toTypeName='c',direction='y-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='e',toTypeName='d',direction='x-',value=1,dual=True)


# tileHandler.selfConnectable(typeName="void",direction='isotropy',value=1)
# tileHandler.setConnectiability(fromTypeName='void',toTypeName=['a','b','c','d','e'],direction=['x+','x-','y+','y-'],value=1,dual=True)






# tileHandler = TileHandler(typeList=['a','c','e','void'],direction=(('y+',"y-"),("x-","x+"),))
# tileHandler.selfConnectable(typeName="e",direction='isotropy',value=1)

# tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c',],direction='y-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','a'],direction='x-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','a'],direction='x+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='a',toTypeName='c',direction='y+',value=1,dual=True)



# tileHandler.setConnectiability(fromTypeName='a',toTypeName=['a','e'],direction='y+',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='a',toTypeName=['a'],direction='y-',value=-1,dual=True)



# tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','a'],direction='y+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','a','c'],direction='x-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','a','c'],direction='x+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName='a',direction='y-',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName=['c'],direction='y+',value=-1,dual=True)
# tileHandler.setConnectiability(fromTypeName='c',toTypeName=['c','e'],direction='y-',value=-1,dual=True)


# tileHandler.setConnectiability(fromTypeName='e',toTypeName='a',direction='y+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='e',toTypeName='c',direction='y-',value=1,dual=True)


# tileHandler.selfConnectable(typeName="void",direction='isotropy',value=1)
# tileHandler.setConnectiability(fromTypeName='void',toTypeName=['a','c','e'],direction=['x+','x-','y+','y-'],value=1,dual=True)



tileHandler = TileHandler(typeList=['a','c','b','d','void'],direction=(('y+',"y-"),("x-","x+"),))
# tileHandler.selfConnectable(typeName="e",direction='isotropy',value=1)

tileHandler.setConnectiability(fromTypeName='a',toTypeName=['b','c','d'],direction='y-',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='a',toTypeName=['c','a','d'],direction='x-',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='a',toTypeName=['b','c','a'],direction='x+',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='a',toTypeName='c',direction='y+',value=1,dual=True)



tileHandler.setConnectiability(fromTypeName='a',toTypeName=['a','b','d'],direction='y+',value=-1,dual=True)
tileHandler.setConnectiability(fromTypeName='a',toTypeName=['a'],direction='y-',value=-1,dual=True)



tileHandler.setConnectiability(fromTypeName='c',toTypeName=['b','a','d'],direction='y+',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='c',toTypeName=['d','a','c'],direction='x-',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='c',toTypeName=['b','a','c'],direction='x+',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='c',toTypeName='a',direction='y-',value=1,dual=True)

tileHandler.setConnectiability(fromTypeName='c',toTypeName=['c'],direction='y+',value=-1,dual=True)
tileHandler.setConnectiability(fromTypeName='c',toTypeName=['c','b','d'],direction='y-',value=-1,dual=True)

tileHandler.setConnectiability(fromTypeName='b',toTypeName=['d'],direction='x+',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='b',toTypeName=['d','a','c'],direction='x-',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='b',toTypeName=['d','b','c'],direction='y-',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='b',toTypeName=['d','b','a'],direction='y+',value=1,dual=True)

tileHandler.setConnectiability(fromTypeName='b',toTypeName=['b','c','a'],direction='x+',value=-1,dual=True)
tileHandler.setConnectiability(fromTypeName='b',toTypeName=['b'],direction='x-',value=-1,dual=True)

tileHandler.setConnectiability(fromTypeName='d',toTypeName=['b'],direction='x-',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='d',toTypeName=['b','a','c'],direction='x+',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='d',toTypeName=['d','b','c'],direction='y-',value=1,dual=True)
tileHandler.setConnectiability(fromTypeName='d',toTypeName=['d','b','a'],direction='y+',value=1,dual=True)

tileHandler.setConnectiability(fromTypeName='d',toTypeName=['b','c','a'],direction='x-',value=-1,dual=True)
tileHandler.setConnectiability(fromTypeName='d',toTypeName=['d'],direction='x+',value=-1,dual=True)


# tileHandler.setConnectiability(fromTypeName='e',toTypeName='a',direction='y+',value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='e',toTypeName='c',direction='y-',value=1,dual=True)


tileHandler.selfConnectable(typeName="void",direction='isotropy',value=1)
tileHandler.setConnectiability(fromTypeName='void',toTypeName=['a','c','b','d'],direction=['x+','x-','y+','y-'],value=1,dual=True)










from src.WFC.WFCFilter_JAX_Sigma_tau_softmax import preprocess_compatibility,waveFunctionCollapse,preprocess_adjacency,compute_cell_centers
tileHandler.constantlize_compatibility()
tileHandler._compatibility = preprocess_compatibility(tileHandler.compatibility)
print(tileHandler)

from src.fem.SigmaInterpreter_constitutive_2D import SigmaInterpreter
# p=[4,3,3]
# p=[3,3,3,3,4,3]
p=[4,4,4,4,4]

assert len(p)==len(tileHandler.typeList),f"p length {len(p)} must equal to tile types num {len(tileHandler.typeList)}"
sigmaInterpreter=SigmaInterpreter(p=p,) #3,4 445 544 
print(sigmaInterpreter)

# Define forward problem.
problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns,
                     additional_info=(sigmaInterpreter,))

# Apply the automatic differentiation wrapper.
# This is a critical step that makes the problem solver differentiable.
# fwd_pred = ad_wrapper(problem, solver_options={'petsc_solver': {}}, adjoint_solver_options={'petsc_solver': {}})
fwd_pred = ad_wrapper(problem, solver_options={'petsc_solver': {'ksp_type':'tfqmr','pc_type':'lu'}}, adjoint_solver_options={'petsc_solver': {'ksp_type':'tfqmr','pc_type':'lu'}})
# fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})

# Define the objective function 'J_total(theta)'.
# In the following, 'sol = fwd_pred(params)' basically says U = U(theta).
def J_total(params):
    # J(u(theta), theta)
    sol_list = fwd_pred(params)
    compliance = problem.compute_compliance(sol_list[0])
    # avg_poisson_xz, avg_poisson_yz = problem.compute_poissons_ratio(sol_list[0])
    # jax.debug.print("avg_poisson_xz= {a}\navg_poisson_yz= {b}",a=avg_poisson_xz,b=avg_poisson_yz)
    # jax.debug.print("compliance= {c}",c=compliance)
    return compliance, sol_list


# Output solution files to local disk
outputs = []
def output_sol(params, obj_val,sol_list):
    print(f"\nOutput solution - params.shape:{params.shape}")
    # sol_list = fwd_pred(params)

    sol = sol_list[0]
    vtu_path = os.path.join(data_path, f'vtk/sol_{output_sol.counter:03d}.vtu')
    cell_infos = [(f'theta{i}', params[:, i]) for i in range(params.shape[-1])]
    mask = np.max(params, axis=-1) > 0.5
    all = np.where(mask, np.argmax(params,axis=-1), np.nan)
    cell_infos.append( ('all', all) )
    mises = problem.compute_von_mises(sol)
    # cell_infos.extend([(f'{key}', item ) for key,item in mises.items()])
    cell_infos.append(('cell_von_mises', mises["cell_von_mises"] ))
    save_sol(problem.fe, np.hstack((sol, np.zeros((len(sol), 1)))), vtu_path, 
             cell_infos=cell_infos)
    # print(f"compliance = {obj_val}")
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


@jax.jit
def material_selection_loss(rho, alpha=5.0):
    x_safe = np.clip(rho, 1e-8, 1.0 - 1e-8)
    
    # 沿n轴计算每个cell的最大值和总和
    max_vals = np.max(x_safe, axis=1)  # 形状: (cells,)
    sum_vals = np.sum(x_safe, axis=1)  # 形状: (cells,)
    
    # 处理全0情况
    # zero_mask = np.all(x_safe < 1e-6, axis=1)  # 形状: (cells,)
    # concentrations = np.where(zero_mask, 1.0, max_vals / (sum_vals + 1e-8))
    concentrations =  max_vals / (sum_vals + 1e-8)
    
    # 基础损失
    losses = 1.0 - concentrations  # 形状: (cells,)
    
    # 惩罚项（每个cell的惩罚）
    penalties = np.sum(x_safe * (1.0 - x_safe), axis=1)  # 形状: (cells,)
    
    # 总损失
    cell_losses = losses + alpha * 0.01 * penalties
    total_loss = np.mean(cell_losses)
    
    return total_loss




def upper_bound_constraint(rho, index, vf):
    """材料体积上限约束：mean(rho[:,index]) ≤ vf → 均值/vf - 1 ≤ 0"""
    return (np.mean(rho[:, index])) / vf - 1.0

def lower_bound_constraint(rho, index, vr):
    """材料体积下限约束：mean(rho[:,index]) ≥ vr → 1 - (均值/vr) ≤ 0"""
    return 1.0 - (np.mean(rho[:, index])) / vr





# ========== 3. 定义所有约束项（自动遍历的核心） ==========
vt=0.6
# vue=0.35
# vua=0.2
# vub=0.15
# vuc=0.2
# vud=0.15

# va=0.15
# vb=0.1
# vc=0.15
# vd=0.1



constraint_items = [
    # 约束1：总非空材料体积约束（sum(rho[:,0:-1])均值 ≤ vt）
    ("total_non_void_volume_without_void", lambda rho: np.mean(np.sum(rho, axis=-1)-rho[:,-1]) / vt - 1.0),
    # 约束2：材料选择损失（原有逻辑，保留）
    ("material_selection_loss", lambda rho: material_selection_loss(rho)),
    # (f"target_material_upper_bound:{vue}", lambda rho: upper_bound_constraint(rho, 2, vue)),
    # (f"1_upper_bound:{vua}", lambda rho: upper_bound_constraint(rho, 0, vua)),
    # (f"2_upper_bound:{vub}", lambda rho: upper_bound_constraint(rho, 1, vub)),
    # (f"3_upper_bound:{vuc}", lambda rho: upper_bound_constraint(rho, 1, vuc)),
    # (f"4_upper_bound:{vud}", lambda rho: upper_bound_constraint(rho, 3, vud)),

    # (f"1_lower_bound:{va}", lambda rho: lower_bound_constraint(rho, 0, va)),
    # (f"2_lower_bound:{vb}", lambda rho: lower_bound_constraint(rho, 1, vb)),
    # (f"3_lower_bound:{vc}", lambda rho: lower_bound_constraint(rho, 1, vc)),
    # (f"4_lower_bound:{vd}", lambda rho: lower_bound_constraint(rho, 3, vd)),

]

# 自动获取约束数量（无需手动定义numConstraints）
numConstraints = len(constraint_items)

def consHandle(rho, *args):
    """自动化约束计算：遍历约束列表，自动生成c和gradc"""
    c_list = []  # 存储每个约束的c值
    gradc_list = []  # 存储每个约束的梯度
    
    # 遍历所有约束项，逐个计算c和梯度
    for constraint_name, constraint_fun in constraint_items:
        # 计算约束值c_i和梯度dc_i/dρ
        c_i, gradc_i = jax.value_and_grad(constraint_fun)(rho)
        # 打印约束值（可选，调试用）
        # jax.debug.print(f"约束[{constraint_name}]值：{c_i:.6f}")
        # 收集结果
        c_list.append(c_i)
        gradc_list.append(gradc_i)
    
    # 自动拼接c和gradc
    c = np.array(c_list).reshape(-1,)  # 形状：(numConstraints,)
    gradc = np.array(gradc_list)  # 形状：(numConstraints, num_cells, tileNum)
    
    return c, gradc

from src.WFC.adjacencyCSR import build_hex8_adjacency_with_meshio,build_quad4_adjacency_with_meshio
# adj=build_hex8_adjacency_with_meshio(mesh=meshio_mesh)
adj = build_quad4_adjacency_with_meshio(mesh=meshio_mesh4ad)
# from src.WFC.WFCFilter_JAX_log_Sigma_tau_revision import preprocess_adjacency,waveFunctionCollapse,compute_cell_centers


# 预构建邻接矩阵和方向矩阵（仅一次）
A, D = preprocess_adjacency(adj, tileHandler)
cell_centers = jax.lax.stop_gradient(compute_cell_centers(mesh4ad.points[mesh4ad.cells]))
wfc=lambda prob,key: waveFunctionCollapse(prob, A, D, tileHandler.opposite_dir_array, tileHandler.compatibility,key, cell_centers)

# Finalize the details of the MMA optimizer, and solve the TO problem.
optimizationParams = {'maxIters':201, 'movelimit':0.1, 'NxNy':(Nx,Ny),'sensitivity_filtering':"nofilter",'filter_radius':1.2}

key = jax.random.PRNGKey(0)
rho_ini = np.ones((Nx,Ny,tileHandler.typeNum),dtype=np.float64).reshape(-1,tileHandler.typeNum)/tileHandler.typeNum
# rho_ini = rho_ini.at[:,0].set(0.3)
# rho_ini = rho_ini.at[:,1].set(0.3)
# rho_ini = rho_ini.at[:,2].set(0.3)
# rho_ini = rho_ini.at[:,3].set(0.3)


# rho_ini = rho_ini + jax.random.uniform(key,shape=rho_ini.shape)*0.1

# import jax_fem.mma_ori as mo
from jax_fem.mma2D import optimize
from src.fem.TileKernelInterpreter import TileKernelInterpreter
tileKernelInterpreter=TileKernelInterpreter(typeList=tileHandler.typeList,folderPath="data/Kernels")
rho_oped,infos=optimize(problem.fe, rho_ini, optimizationParams, objectiveHandle, consHandle, numConstraints,tileNum=tileHandler.typeNum,WFC=wfc,tileKernelInterpreter=tileKernelInterpreter)
# rho_oped,J_list = mo.optimize(problem.fe, rho_ini, optimizationParams, objectiveHandle, consHandle, numConstraints,)

np.save("data/npy/rho_oped",rho_oped)


# Plot the optimization results.
obj = onp.array(outputs)
create_directory_if_not_exists("data/csv")
onp.savetxt( "data/csv/topo_obj.csv", onp.array(obj), delimiter="," )


fig=plt.figure(figsize=(12, 5))
ax=fig.add_subplot(1,2,1)
ax.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
clear_ratio = []
for index, (epoch, results) in enumerate(infos.items()):
    clear_ratio.append(results["clear_ratio"])
ax=fig.add_subplot(1,2,2)
ax.plot(onp.arange(len(clear_ratio)) + 1, onp.array(clear_ratio), linestyle='-', linewidth=2, color='black')
# ax.xlabel(r"Optimization step", fontsize=20)
# ax.ylabel(r"Objective value", fontsize=20)
# ax.tick_params(labelsize=20)
# ax.tick_params(labelsize=20)

plt.savefig("data/topo_obj.tiff")
# plt.show()



# rho_oped = np.load("/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/vtk++TT0TT180完全约束有filter/npy/rho_oped.npy")
rho_oped = np.load("data/npy/rho_oped.npy")
import src.WFC.classicalWFC as normalWFC

#后处理环节
wfc_classical_end ,max_entropy, collapse_list= jax.lax.stop_gradient(normalWFC.waveFunctionCollapse(rho_oped,adj,tileHandler))
np.save("data/npy/wfc_classical_end.npy",wfc_classical_end)


types_str = ""
for t in tileHandler.typeList:
    types_str += t 
p_str=''
for pi in p:
    p_str += str(pi)

con_lines=[tu[0] for tu in constraint_items]

lines = [
         f"P:{p}\n",
         f"PileHandler:{tileHandler}\n",
         f"Lx,Ly:{Lx},{Ly}\n",
         f"Nx,Ny:{Nx},{Ny}\n",
         f"OptimizationParams:{optimizationParams}\n",
         f"name:f{types_str}{optimizationParams['sensitivity_filtering']}{optimizationParams['filter_radius']}p{p_str}",
         f'Simp',
         f"WFCsigma",
         f"2D",
         ]

with open("data/vtk/parameters.txt", "w", encoding="utf-8") as f:
    try:
        for line in lines:
            f.write(line)  # 写入内容
    except Exception as e:
        print(f"Error writing to file: {e}")
    try:
        f.write("\nconstraints:\n")
        for con_line in con_lines:
            f.write(f"{con_line}\n")
    except Exception as e:
        print(f"Error writing constraints to file: {e}")
