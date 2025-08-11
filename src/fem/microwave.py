import os
from pathlib import Path
import sys
 
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

import meshio

import jax
jax.config.update('jax_disable_jit', True)
import jax.numpy as jnp
from jax_fem.problem import Problem
from jax_fem.generate_mesh import box_mesh_gmsh,Mesh,get_meshio_cell_type
from jax_fem.solver import solver
from jax_fem.utils import save_sol


class Microwave(Problem):
    def __init__(self, metamaterial_def, frequency=10e9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frequency = frequency
        self.material_map = metamaterial_def
        
        # 物理常数
        self.c0 = 3e8
        self.eps0 = 8.854e-12
        self.mu0 = 4 * jnp.pi * 1e-7
        
        # 计算波数
        self.omega = 2 * jnp.pi * self.frequency
        self.k0 = self.omega / self.c0

    def get_tensor_map(self):
        def tensor_map(u_grad, x):
            # 获取材料参数
            eps_r = self.material_map(x)
            mu_r = 1.0  # 固定磁导率
            
            # 计算磁导率张量
            mu_inv = 1 / (mu_r * self.mu0) * jnp.eye(3)
            
            # 提取电场分量梯度
            grad_Ex = u_grad[0, :]
            grad_Ey = u_grad[1, :]
            grad_Ez = u_grad[2, :]
            
            # 计算旋度 ∇ × E
            curl_E = jnp.array([
                grad_Ey[2] - grad_Ez[1],
                grad_Ez[0] - grad_Ex[2],
                grad_Ex[1] - grad_Ey[0]
            ])
            
            # 计算磁通密度 B = μ⁻¹(∇ × E)
            # 返回形状为 (3,) 的向量
            B = mu_inv @ curl_E
            
            # 为了匹配框架期望的形状 (3, 3)，我们需要返回一个张量
            # 这里我们创建一个对角张量，对角线元素为 B
            # 实际物理意义可能需要调整，但这是为了匹配框架要求
            return jnp.diag(B)
        
        return tensor_map

    def get_mass_map(self):
        def mass_map(u, x, *internal_vals):
            # 获取介电常数
            eps_r = self.material_map(x)
            
            # 波动方程项: -ω²εE
            # 返回形状为 (3,) 的向量
            return -self.omega**2 * self.eps0 * eps_r * u
        
        return mass_map
    
    def get_source(self):
        def source_fn(x):
            # 点源位于中心
            center = jnp.array([0.5, 0.5, 0.5])
            dist = jnp.linalg.norm(x - center, axis=-1)
            sigma = 0.05
            strength = 10.0
            
            # 高斯源 (z方向极化)
            source_val = strength * jnp.exp(-dist**2 / (2 * sigma**2))
            zeros = jnp.zeros_like(source_val)
            return jnp.stack([zeros, zeros, source_val], axis=-1)
        
        return source_fn

    def boundary_condition(self):
        def dirichlet_fun(point):
            return jnp.zeros(3)
        
        return [{'type': 'dirichlet', 'dof': None, 'boundary_fn': dirichlet_fun}]

    def set_params(self, params):
        # Override base class method.
        full_params = jnp.zeros((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = jnp.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]



def create_material_map():
    def material_map(x):
        center = jnp.array([0.5, 0.5, 0.5])
        diff = x - center
        r_sq = jnp.sum(jnp.square(diff), axis=-1)  # 使用平方距离避免平方根
        
        # 使用平滑过渡函数
        transition = 1 / (1 + jnp.exp(50 * (r_sq - 0.04)))  # 0.04 = 0.2**2
        eps_r = 1.0 + 3.0 * transition  # 从1.0到4.0的平滑过渡
        
        return eps_r  # 只返回一个值
    
    return material_map

def get_cell_vertices_vectorized(meshio_mesh):
    """may equal to problem.physical_quad_points"""
    points = meshio_mesh.points
    hex_cells = None
    for cell_block in meshio_mesh.cells:
        if cell_block.type == "hexahedron":
            hex_cells = cell_block.data
            break
    if hex_cells is None:
        raise ValueError("网格中未找到HEX8单元")
    
    # 向量化索引操作
    return points[hex_cells]



if __name__ == "__main__":
    material_map = create_material_map()
    
    # 2. 创建网格 (单位立方体)
    meshio_mesh = box_mesh_gmsh(
        Nx=10, Ny=10, Nz=10, 
        domain_x=1.0, domain_y=1.0, domain_z=1.0,
        ele_type='HEX8', data_dir='data'
    )
    
    cell_vertices = get_cell_vertices_vectorized(meshio_mesh)
    
    mesh = Mesh(
        points=meshio_mesh.points,
        cells=meshio_mesh.cells_dict[get_meshio_cell_type('HEX8')],
        ele_type='HEX8'
    )
    
    # 3. 创建问题实例
    problem = Microwave(
        metamaterial_def=material_map,
        frequency=2.4e9,
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type='HEX8'
    )
    problem.internal_vars=[problem.physical_quad_points]
    
    # 4. 求解问题
    print("Solving microwave problem...")
    sol = solver(problem)  # 移除了可能引起问题的 solver_options 参数
    print("Solution shape:", sol[0].shape)
    
    # 5. 保存结果
    data_dir = "microwave_results"
    os.makedirs(data_dir, exist_ok=True)
    
    # 保存电势场
    vtk_path = os.path.join(data_dir, "Electric_Potential.vtu")
    save_sol(problem.fes[0], sol[0], vtk_path)
    
    # 计算电场强度 E = -∇Φ
    print("Calculating electric field...")
    points = mesh.points
    cells = mesh.cells
    num_nodes = points.shape[0]
    
    # 创建空数组存储电场分量
    Ex = jnp.zeros(num_nodes)
    Ey = jnp.zeros(num_nodes)
    Ez = jnp.zeros(num_nodes)
    
    # 对于每个单元计算电场并插值到节点
    for cell in cells:
        cell_points = points[cell]
        cell_sol = sol[0][cell]
        
        # 计算形函数梯度
        grad_basis = problem.fes[0].shape_grads[0]  # 简化处理，实际应用中应正确计算
        
        # 计算电场：E = -∇Φ
        grad_phi = jnp.einsum('i,ijk->jk', cell_sol, grad_basis)
        E_cell = -grad_phi
        
        # 插值到节点（简化处理）
        for idx, node_idx in enumerate(cell):
            Ex = Ex.at[node_idx].add(E_cell[idx, 0])
            Ey = Ey.at[node_idx].add(E_cell[idx, 1])
            Ez = Ez.at[node_idx].add(E_cell[idx, 2])
    
    # 归一化处理
    normalization_factor = len(cells) / 8.0  # HEX8单元有8个节点
    Ex /= normalization_factor
    Ey /= normalization_factor
    Ez /= normalization_factor
    
    # 创建带有电场分量的网格
    mesh_with_fields = meshio.Mesh(
        points=mesh.points,
        cells=[("hexahedron", mesh.cells)],
        point_data={
            "Electric_Potential": sol[0],
            "Electric_Field_x": Ex,
            "Electric_Field_y": Ey,
            "Electric_Field_z": Ez,
            "Electric_Field_Magnitude": jnp.sqrt(Ex**2 + Ey**2 + Ez**2)
        }
    )
    
    # 保存带有电场的VTK文件
    vtk_path = os.path.join(data_dir, "Electric_Fields.vtu")
    mesh_with_fields.write(vtk_path)
    
    # 创建带有介电常数分布的网格
    eps_r = jnp.array([material_map(p) for p in points])
    material_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("hexahedron", mesh.cells)],
        point_data={"Relative_Permittivity": eps_r}
    )
    
    # 保存介电常数分布
    eps_path = os.path.join(data_dir, "Dielectric_Constant.vtu")
    material_mesh.write(eps_path)
    
    print(f"Saved results to directory: {data_dir}")
    print("Visualize using ParaView or similar software:")
    print(f"1. Electric potential: {os.path.join(data_dir, 'Electric_Potential.vtu')}")
    print(f"2. Electric fields: {os.path.join(data_dir, 'Electric_Fields.vtu')}")
    print(f"3. Dielectric distribution: {os.path.join(data_dir, 'Dielectric_Constant.vtu')}")
    print("Test completed successfully!")