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
from jax_fem.mma import optimize

from src.WFC.TileHandler_JAX import TileHandler
from src.WFC.adjacencyCSR import build_hex8_adjacency_with_meshio

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
            sigma = self.sigmaInterpreter(u_grad,weights)
            return sigma
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 0., -1])
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
    
    def compute_poissons_ratio(self, sol):
        # 获取位移梯度
        u_grad = self.fe.sol_to_grad(sol)
        # 计算应变张量 (num_cells, num_quads, vec, dim)
        epsilon = 0.5 * (u_grad + u_grad.transpose(0, 1, 3, 2))
        # 提取应变分量
        epsilon_xx = epsilon[..., 0, 0]  # x方向正应变
        epsilon_yy = epsilon[..., 1, 1]  # y方向正应变
        epsilon_zz = epsilon[..., 2, 2]  # z方向正应变（压缩方向）
        # 获取积分权重 (num_cells, num_quads)
        cells_JxW = self.JxW[:, 0, :]
        # 计算单元体积 (num_cells,)
        cell_volumes = np.sum(cells_JxW, axis=1)
        # 计算单元平均应变
        cell_eps_xx = np.sum(epsilon_xx * cells_JxW, axis=1) / cell_volumes
        cell_eps_yy = np.sum(epsilon_yy * cells_JxW, axis=1) / cell_volumes
        cell_eps_zz = np.sum(epsilon_zz * cells_JxW, axis=1) / cell_volumes
        # 计算单元级泊松比
        cell_poisson_xz = -cell_eps_xx / cell_eps_zz
        cell_poisson_yz = -cell_eps_yy / cell_eps_zz
        # 计算全局体积加权平均泊松比
        total_volume = np.sum(cells_JxW)
        avg_poisson_xz = -np.sum(epsilon_xx * cells_JxW) / np.sum(epsilon_zz * cells_JxW)
        avg_poisson_yz = -np.sum(epsilon_yy * cells_JxW) / np.sum(epsilon_zz * cells_JxW)
        return avg_poisson_xz, avg_poisson_yz


    def compute_von_mises(self, sol):
        # 停止梯度：后处理无需梯度跟踪
        weights = jax.lax.stop_gradient(self.internal_vars[0])  # (num_cells, num_quads, tiletypes)
        u_grad = jax.lax.stop_gradient(self.fe.sol_to_grad(sol))  # 4维：(num_cells, num_quads, dim, dim)
        sigma = jax.lax.stop_gradient(self.sigmaInterpreter(u_grad, weights))  # 4维：(num_cells, num_quads, dim, dim)
        dim = self.dim  # 3（三维问题）
        sigma_tr = np.trace(sigma, axis1=2, axis2=3)  # 形状：(num_cells, num_quads)
        sigma_spherical = (sigma_tr / dim)[..., None, None] * np.eye(dim)[None, None, :, :]  # 4维：(num_cells, num_quads, dim, dim)
        s_dev = sigma - sigma_spherical  # 4维：(num_cells, num_quads, dim, dim)
        vm_gauss = np.sqrt(3. / 2. * np.sum(s_dev **2, axis=(2, 3)))  # 形状：(num_cells, num_quads)
        cells_JxW = self.JxW[:, 0, :]  # 形状：(num_cells, num_quads)
        cell_volumes = np.sum(cells_JxW, axis=1)  # 形状：(num_cells,)
        cell_vm = np.sum(vm_gauss * cells_JxW, axis=1) / cell_volumes  # 形状：(num_cells,)
        total_volume = np.sum(cells_JxW)
        avg_vm = np.sum(vm_gauss * cells_JxW) / total_volume  # 标量
        return {
            'cell_von_mises': cell_vm,
            'avg_von_mises': avg_vm,
            'gauss_von_mises': vm_gauss
        }
# Specify mesh-related information. We use first-order quadrilateral element.
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
# Lx, Ly, Lz = 60., 10., 30.
# Nx, Ny, Nz = 60, 10, 30
# Lx, Ly, Lz = 10., 2., 5.
# Nx, Ny, Nz = 10, 2, 5
Lx, Ly, Lz = 40., 5., 20.
Nx, Ny, Nz = 40, 5, 20
create_directory_if_not_exists("data/msh")
mshname=f"L{Lx}{Ly}{Lz}N{Nx}{Ny}{Nz}.msh"
if not os.path.exists(f"data/msh/{mshname}"):
    meshio_mesh = box_mesh_gmsh(Nx=Nx,Ny=Ny,Nz=Nz,domain_x=Lx,domain_y=Ly,domain_z=Lz,data_dir="data",ele_type=ele_type,name=mshname)
else:
    meshio_mesh = meshio.read(f"data/msh/{mshname}")
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
# Define boundary conditions and values.
def fixed_location(point):
    return np.isclose(point[0], 0., atol=0.1+1e-5)

def load_location(point):
    return np.logical_and(np.isclose(point[2], 0, atol=0.1*Lz+1e-5),
                          np.isclose(point[0], Lx, atol=0.1*Lx+1e-5))



def dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[fixed_location]*3, [0, 1, 2], [dirichlet_val]*3]

location_fns = [load_location]

tileHandler = TileHandler(typeList=['BCC3', 'cubic1'], 
                          direction=(('back',"front"),("left","right"),("top","bottom")),
                          direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
tileHandler.selfConnectable(typeName=['BCC3', 'cubic1'],value=1)
tileHandler.setConnectiability(fromTypeName='BCC3',toTypeName=[ 'cubic1',],direction="isotropy",value=1,dual=True)



# tileHandler = TileHandler(typeList=['BCC3', 'cubic1', '++'], 
#                           direction=(('back',"front"),("left","right"),("top","bottom")),
#                           direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
# tileHandler.selfConnectable(typeName=['++','BCC3', 'cubic1'],value=1)
# tileHandler.setConnectiability(fromTypeName='++',toTypeName=[ 'BCC3',],direction="isotropy",value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='BCC3',toTypeName=[ 'cubic1',],direction="isotropy",value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='++',toTypeName=[ 'cubic1',],direction="isotropy",value=0,dual=True)
# tileHandler.setConnectiability(fromTypeName='++',toTypeName=[ 'TTx180',],direction="left",value=0,dual=True)


# tileHandler = TileHandler(typeList=['++', 'TTx0', 'TTx180'], 
#                           direction=(('back',"front"),("left","right"),("top","bottom")),
#                           direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
# tileHandler.selfConnectable(typeName=['++','TTx0', 'TTx180'],value=1)
# tileHandler.setConnectiability(fromTypeName='++',toTypeName=[ 'TTx0','TTx180'],direction="isotropy",value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='TTx180',toTypeName=[ 'TTx0',],direction="isotropy",value=1,dual=True)
# tileHandler.setConnectiability(fromTypeName='++',toTypeName=[ 'TTx0',],direction="right",value=0,dual=True)
# tileHandler.setConnectiability(fromTypeName='++',toTypeName=[ 'TTx180',],direction="left",value=0,dual=True)


print(tileHandler)
tileHandler.constantlize_compatibility()

from src.fem.SigmaInterpreter_constitutive import SigmaInterpreter
# sigmaInterpreter=SigmaInterpreter(typeList=tileHandler.typeList,folderPath="data/EVG", p=[3,3,4], debug=False) #3,4
sigmaInterpreter=SigmaInterpreter(typeList=tileHandler.typeList,folderPath="data/EVG", debug=False) #3,4

print(sigmaInterpreter)
# Define forward problem.
problem = Elasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns,
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
    # mises = problem.compute_von_mises(sol)
    # cell_infos.extend([(f'{key}', item ) for key,item in mises.items()])
    # cell_infos.append(('cell_von_mises', mises["cell_von_mises"] ))
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
    zero_mask = np.all(x_safe < 1e-6, axis=1)  # 形状: (cells,)
    concentrations = np.where(zero_mask, 1.0, max_vals / (sum_vals + 1e-8))
    
    # 基础损失
    losses = 1.0 - concentrations  # 形状: (cells,)
    
    # 惩罚项（每个cell的惩罚）
    penalties = np.sum(x_safe * (1.0 - x_safe), axis=1)  # 形状: (cells,)
    
    # 总损失
    cell_losses = losses + alpha * 0.01 * penalties
    total_loss = np.mean(cell_losses)
    
    return total_loss



vt=0.35
vf0 = 0.2
vf1 = 0.2
# vf2 = 0.2
# Prepare g and dg/d(theta) that are required by the MMA optimizer.
numConstraints = 4
def consHandle(rho,*args):
    # MMA solver requires (c, dc) as inputs
    # c should have shape (numConstraints,)
    # dc should have shape (numConstraints, ...)
    # def computeGlobalVolumeConstraint(rho):
    #     g = np.mean(rho)/vf0 - 1.
    #     return g
    def totalVolume(rho):
        t = np.mean(np.sum(rho,axis=-1,keepdims=False))/vt -1 #没用二次形式的时候应该也行
        return t
    ct, gradct = jax.value_and_grad(totalVolume)(rho)
    cm , gradcm = jax.value_and_grad(material_selection_loss)(rho)
    c0, gradc0 = jax.value_and_grad(lambda rho: (np.mean(rho[...,0])/vf0)-1 )(rho)
    c1, gradc1 = jax.value_and_grad(lambda rho: (np.mean(rho[...,1])/vf1)-1 )(rho)
    # c2, gradc2 = jax.value_and_grad(lambda rho: (np.mean(rho[...,2])/vf2)-1 )(rho)
    # c0, gradc0 = jax.value_and_grad(computeGlobalVolumeConstraint)(rho)
    # c=np.array([ ct,c0,c1 ])
    # gradc=np.array([ gradct,gradc0,gradc1])

    c=np.array([ct, cm, c0, c1 ])
    gradc=np.array([gradct, gradcm, gradc0, gradc1 ])
    # print(f"c.shape:{c.shape}")
    # print(f"gradc.shape:{gradc.shape}")
    c = c.reshape((-1,))
    return c, gradc

adj=build_hex8_adjacency_with_meshio(mesh=meshio_mesh)
from src.WFC.WFCFilter_JAX_log_monotonicity import preprocess_adjacency,waveFunctionCollapse


# 预构建邻接矩阵和方向矩阵（仅一次）
A, D = preprocess_adjacency(adj, tileHandler)
wfc=lambda prob: waveFunctionCollapse(prob, A, D, tileHandler.opposite_dir_array, tileHandler.compatibility)

# Finalize the details of the MMA optimizer, and solve the TO problem.
optimizationParams = {'maxIters':101, 'movelimit':0.1, 'NxNyNz':(Nx,Ny,Nz),'sensitivity_filtering':True}

key = jax.random.PRNGKey(0)
rho_ini = np.ones((Nx,Ny,Nz,tileHandler.typeNum),dtype=np.float64).reshape(-1,tileHandler.typeNum)*0.25
rho_ini = rho_ini.at[:,1].set(0.25)
# rho_ini = rho_ini.at[:,2].set(0.20)

# rho_ini = rho_ini + jax.random.uniform(key,shape=rho_ini.shape)*0.1

import jax_fem.mma_ori as mo
# rho_oped,J_list=optimize(problem.fe, rho_ini, optimizationParams, objectiveHandle, consHandle, numConstraints,tileNum=tileHandler.typeNum,WFC=wfc)
# # rho_oped,J_list = mo.optimize(problem.fe, rho_ini, optimizationParams, objectiveHandle, consHandle, numConstraints,)
# create_directory_if_not_exists("data/npy")
# np.save("data/npy/rho_oped",rho_oped)


# # Plot the optimization results.
# obj = onp.array(outputs)
# create_directory_if_not_exists("data/csv")
# onp.savetxt( "data/csv/topo_obj.csv", onp.array(obj), delimiter="," )


# fig=plt.figure(figsize=(12, 5))
# ax=fig.add_subplot(1,2,1)
# ax.plot(onp.arange(len(obj)) + 1, obj, linestyle='-', linewidth=2, color='black')
# # ax.xlabel(r"Optimization step", fontsize=20)
# # ax.ylabel(r"Objective value", fontsize=20)
# # ax.tick_params(labelsize=20)
# # ax.tick_params(labelsize=20)

# ax=fig.add_subplot(1,2,2)

# plt.savefig("data/topo_obj.tiff")
# plt.show()



# rho_oped = np.load("../../data/见到最好的BCCcubic/npy/rho_oped.npy")
rho_oped = np.load("data/见到最好的BCCcubic/npy/rho_oped.npy")
import src.WFC.iterateWaveFunctionCollapse_map as normalWFC
wfc_end ,max_entropy, collapse_list= jax.lax.stop_gradient(normalWFC.waveFunctionCollapse(rho_oped,adj,tileHandler,max_neighbors=8))
np.save("data/npy/wfc_end",wfc_end)


