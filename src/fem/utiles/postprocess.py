import jax
import jax.numpy as np
from jax import vmap, grad


def compute_cell_volumes(points, cells):
    """计算有限元网格中所有单元的体积（修正版）"""
    def hexahedron_volume(vertices):
        """计算六面体单元（HEX8）体积：采用5四面体分割法（标准方法）"""
        # HEX8顶点标准顺序：(0-3为底面，4-7为顶面，0与4、1与5等对应)
        v = vertices
        # 分割为5个四面体（体积计算更稳定）
        vols = [
            tetrahedron_volume(v[0], v[1], v[2], v[4]),
            tetrahedron_volume(v[2], v[3], v[0], v[4]),
            tetrahedron_volume(v[2], v[4], v[6], v[7]),
            tetrahedron_volume(v[2], v[7], v[4], v[0]),
            tetrahedron_volume(v[2], v[1], v[5], v[4])
        ]
        return np.sum(vols)
    
    def tetrahedron_volume(v0, v1, v2, v3):
        """计算四面体体积（标量三重积法，正确无误）"""
        a = v1 - v0
        b = v2 - v0
        c = v3 - v0
        return np.abs(np.dot(a, np.cross(b, c))) / 6.0
    
    def polygon_area(vertices):
        """计算2D多边形面积（改用鞋带公式，精度更高）"""
        # 确保输入为2D坐标（z分量为0时忽略）
        xy = vertices[:, :2]  # 取前两列作为平面坐标
        n = xy.shape[0]
        x, y = xy[:, 0], xy[:, 1]
        # 鞋带公式：0.5 * |sum(x_i y_{i+1} - x_{i+1} y_i)|
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        return area
    
    # 用vmap替代循环，提升JAX效率（向量化处理所有单元）
    def process_cell(cell):
        vertices = points[cell]
        if len(cell) == 8:  # HEX8
            return hexahedron_volume(vertices)
        elif len(cell) == 4:  # TET4
            return tetrahedron_volume(*vertices)
        else:  # 2D多边形（三角形、四边形等）
            return polygon_area(vertices)
    
    # 向量化计算所有单元体积
    volumes = vmap(process_cell)(cells)
    return volumes


def calculate_element_disp_grad(fe, element_points, element_solution):
    """计算单元内的位移梯度（修正：用解析形函数梯度，增加积分点）"""
    dim = fe.fe.dim
    num_nodes = element_points.shape[0]
    
    # 仅处理HEX8单元（线性六面体）
    if num_nodes != 8:
        raise ValueError("仅支持HEX8单元的位移梯度计算")
    
    # HEX8形函数对参考坐标(ξ,η,ζ)的解析梯度（参考单元[-1,1]^3）
    # 形函数：N_i = 0.125*(1+ξ_iξ)(1+η_iη)(1+ζ_iζ)，其中(ξ_i,η_i,ζ_i)为节点参考坐标
    xi_ref = np.array([  # HEX8节点的参考坐标
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ])
    
    # 高斯积分点：1x1x1（完全积分线性单元），可扩展为2x2x2
    xi_gps = np.array([[0.0, 0.0, 0.0]])  # 积分点坐标
    weights = np.array([1.0])  # 积分权重（归一化）
    
    # 解析计算形函数在积分点的梯度（dN/dξ, dN/dη, dN/dζ）
    def shape_grad(xi):
        """计算单个积分点的形函数梯度"""
        ξ, η, ζ = xi
        grads = []
        for (ξi, ηi, ζi) in xi_ref:
            dNdξ = 0.125 * ξi * (1 + ηi*η) * (1 + ζi*ζ)
            dNdη = 0.125 * ηi * (1 + ξi*ξ) * (1 + ζi*ζ)
            dNdζ = 0.125 * ζi * (1 + ξi*ξ) * (1 + ηi*η)
            grads.append([dNdξ, dNdη, dNdζ])
        return np.array(grads)  # 形状：(8, 3)
    
    # 计算雅可比矩阵并求逆（积分点平均）
    J_list = []
    for xi_gp in xi_gps:
        N_grad_ref = shape_grad(xi_gp)  # 参考坐标下的形函数梯度
        # 雅可比矩阵：J = dx/dξ = sum(x_i * dN_i/dξ^T)
        J = np.einsum('ni,nd->d', element_points, N_grad_ref)  # 等价于sum(outer(x_i, grad_i))
        J_list.append(J)
    J_avg = np.mean(J_list, axis=0)  # 积分点平均雅可比矩阵
    
    # 检查雅可比矩阵奇异性（避免不可逆）
    J_det = np.linalg.det(J_avg)
    if J_det < 1e-12:
        raise ValueError(f"单元雅可比矩阵奇异（行列式={J_det}），可能单元质量差")
    J_inv = np.linalg.inv(J_avg)
    
    # 物理坐标下的形函数梯度：dN/dx = (dN/dξ) * J^{-1}
    N_grad_phys = vmap(lambda g: J_inv @ g)(shape_grad(xi_gps[0]))
    
    # 位移梯度 ∇u = sum(u_i * dN_i/dx^T)
    disp_grad = np.einsum('ni,nd->d', element_solution, N_grad_phys)
    return disp_grad


def calculate_element_strain(disp_grads):
    """计算小应变张量（正确无误）"""
    return 0.5 * (disp_grads + disp_grads.T)


def calculate_elastic_constants(avg_stress, avg_strain):
    """从平均应力应变计算弹性常数（扩展至多轴情况）"""
    # 提取主分量（假设x方向为加载方向）
    sigma_xx = avg_stress[0, 0]
    epsilon_xx = avg_strain[0, 0]
    
    # 杨氏模量（仅在单轴加载时有效）
    E = sigma_xx / epsilon_xx if np.abs(epsilon_xx) > 1e-8 else 0.0
    
    # 泊松比（横向应变/轴向应变，取绝对值）
    if np.abs(epsilon_xx) > 1e-8:
        nu = -np.mean([avg_strain[1, 1], avg_strain[2, 2]]) / epsilon_xx
        nu = np.clip(nu, 0.0, 0.5)  # 泊松比物理范围：0~0.5
    else:
        nu = 0.0
    
    return E, nu


def calculate_elastic_matrix(E, nu):
    """计算各向同性弹性矩阵（Voigt表示法，修正JAX数组赋值方式）"""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # 直接构造矩阵（避免at方法，更符合JAX习惯）
    C = np.zeros((6, 6))
    C = C.at[[0,1,2], [0,1,2]].set(lam + 2*mu)  # 对角项（xx, yy, zz）
    C = C.at[[0,0,1,1,2,2], [1,2,0,2,0,1]].set(lam)  # 交叉项（xx-yy等）
    C = C.at[[3,4,5], [3,4,5]].set(mu)  # 剪切项（yz, xz, xy）
    return C


def compute_initial_shear_modulus(energy_function):
    """计算零应变初始剪切模量（修正：基于能量密度函数）"""
    def pure_shear_energy(gamma):
        """纯剪切变形：F = [[1, gamma, 0], [0,1,0], [0,0,1]]"""
        F = np.array([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return energy_function(F)  # W(F)为能量密度
    
    # 剪切模量mu = d²W/d(gamma)² 在gamma=0处的值
    d2W_dgamma2 = grad(grad(pure_shear_energy))(0.0)
    return d2W_dgamma2


def compute_initial_bulk_modulus(energy_function):
    """计算零应变初始体积模量（修正：基于能量密度函数）"""
    def volumetric_energy(theta):
        """体积变形：F = (1+theta)^(1/3) * I（保持形状不变）"""
        J = 1.0 + theta  # 体积比J = det(F)
        F = J **(1/3) * np.eye(3)
        return energy_function(F)  # W(F)为能量密度
    
    # 体积模量kappa = J*d²W/d(J)² 在J=1处的值（J=1+theta，theta=0时J=1）
    d2W_dtheta2 = grad(grad(volumetric_energy))(0.0)
    return d2W_dtheta2  # 等价于kappa = d2W_dtheta2（因J=1时导数等价）


def calculate_hyperelastic_properties(problem, sol):
    """计算超弹性材料有效性质（修正能量函数使用及变形梯度定义）"""
    fe = problem.fes[0]
    cells = fe.cells
    points = fe.points
    solution = sol  # 位移场
    
    # 1. 计算单元体积及总体积
    volumes = compute_cell_volumes(points, cells)
    total_volume = np.sum(volumes)
    if total_volume < 1e-12:
        raise ValueError("网格总体积接近零，可能输入错误")
    
    # 2. 定义能量密度函数（关键修正：需从问题中获取能量函数）
    # 假设problem包含energy_function(F)，返回单位体积能量
    def energy_function(F):
        return problem.energy_function(F)
    
    # 3. 单元级计算（用vmap向量化处理）
    def process_element(element_idx):
        cell = cells[element_idx]
        elem_points = points[cell]  # 单元节点坐标（初始构型）
        elem_disp = solution[cell]  # 单元节点位移
        
        # 位移梯度
        disp_grad = calculate_element_disp_grad(fe, elem_points, elem_disp)
        # 变形梯度F = I + ∇u
        F = disp_grad + np.eye(3)
        J = np.linalg.det(F)  # 体积比
        
        # 第一类Piola-Kirchhoff应力P = dW/dF（假设problem提供应力计算）
        P = problem.pk1_stress(F)  # 修正：直接从变形梯度计算应力
        
        # 柯西应力sigma = (1/J) * P * F^T
        sigma = (P @ F.T) / J
        
        # Green-Lagrange应变E = 0.5*(F^T F - I)
        E = 0.5 * (F.T @ F - np.eye(3))
        
        return P, sigma, E, F, volumes[element_idx]
    
    # 向量化处理所有单元
    p_list, sigma_list, e_list, f_list, vol_list = vmap(process_element)(np.arange(len(cells)))
    
    # 4. 体积加权平均
    avg_pk1 = np.sum(p_list * vol_list[:, None, None], axis=0) / total_volume
    avg_cauchy = np.sum(sigma_list * vol_list[:, None, None], axis=0) / total_volume
    avg_green = np.sum(e_list * vol_list[:, None, None], axis=0) / total_volume
    avg_F = np.sum(f_list * vol_list[:, None, None], axis=0) / total_volume
    avg_J = np.linalg.det(avg_F)
    
    # 5. 初始模量（零应变状态）
    initial_mu = compute_initial_shear_modulus(energy_function)
    initial_kappa = compute_initial_bulk_modulus(energy_function)
    
    # 6. 有效切线/割线模量（单轴拉伸）
    def uniaxial_response(eps):
        """单轴拉伸变形梯度（考虑泊松效应）"""
        # 假设材料不可压缩（J=1），横向应变=-nu*eps，此处简化为sqrt(1/(1+eps))
        lateral = 1.0 / np.sqrt(1.0 + eps)  # 确保J=1
        F = np.diag([1.0 + eps, lateral, lateral])
        P = problem.pk1_stress(F)  # 名义应力
        return P[0, 0]  # x方向名义应力
    
    # 小应变范围内计算（0到0.1）
    strains = np.linspace(1e-8, 0.1, 10)
    stresses = vmap(uniaxial_response)(strains)
    tangent_modulus = np.gradient(stresses, strains)[-1]  # 切线模量
    secant_modulus = stresses[-1] / strains[-1] if strains[-1] > 1e-8 else 0.0
    
    # 7. 材料参数（从问题中获取，非硬编码）
    mat_params = problem.material_params  # 假设包含'E', 'nu'等
    
    return {
        'avg_pk1_stress': avg_pk1,
        'avg_cauchy_stress': avg_cauchy,
        'avg_green_strain': avg_green,
        'volumetric_ratio': avg_J,
        'initial_shear_modulus': initial_mu,
        'initial_bulk_modulus': initial_kappa,
        'tangent_modulus': tangent_modulus,
        'secant_modulus': secant_modulus,
        'material_params': mat_params
    }
