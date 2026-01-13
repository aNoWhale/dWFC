"""
网格驱动的点阵结构生成器
基于HEX8网格单元，填充不同类型的点阵单元
"""

import gmsh
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict, field
import meshio
from pathlib import Path

@dataclass
class LatticeUnit:
    """点阵单元定义"""
    stp_file: str
    label: str
    dimensions: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # 默认尺寸
    
@dataclass
class LatticeConfig:
    """点阵配置"""
    hex_mesh_file: str
    unit_assignments: np.ndarray  # 每个单元的类型索引
    lattice_units: List[LatticeUnit]  # 可用的点阵单元
    output_dir: str = "output"
    mesh_controls: Dict = field(default_factory=lambda: {
        'min_size': 0.1,
        'max_size': 1.0,
        'mesh_algorithm': 6,
        'optimize': True
    })
    verbose: bool = True

class HexMeshDrivenLatticeGenerator:
    """基于HEX8网格的点阵结构生成器"""
    
    def __init__(self, config: LatticeConfig):
        """
        初始化生成器
        
        参数:
        config: 点阵配置
        """
        self.config = config
        self.verbose = config.verbose
        self.hex_mesh = None
        self.lattice_units = {}
        self.instances = []
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1 if self.verbose else 0)
        
    # def load_hex_mesh(self):
    #     """加载HEX8网格"""
    #     if self.verbose:
    #         print(f"加载HEX8网格: {self.config.hex_mesh_file}")
        
    #     # 支持多种网格格式
    #     mesh_ext = Path(self.config.hex_mesh_file).suffix.lower()
        
    #     if mesh_ext in ['.msh', '.mesh']:
    #         # 使用Gmsh加载
    #         gmsh.open(self.config.hex_mesh_file)
            
    #         # 获取节点和单元
    #         node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    #         element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()
            
    #         # 查找HEX8单元 (Gmsh中HEX8的类型是5)
    #         hex8_type = 5
    #         hex_elements = None
    #         for elem_type, elem_tags, elem_nodes in zip(element_types, element_tags, element_node_tags):
    #             if elem_type == hex8_type:
    #                 hex_elements = {
    #                     'tags': elem_tags,
    #                     'nodes': elem_nodes.reshape(-1, 8)  # 重塑为(N, 8)
    #                 }
    #                 break
            
    #         if hex_elements is None:
    #             raise ValueError("未在网格中找到HEX8单元")
            
    #         # 重新组织节点坐标
    #         nodes = node_coords.reshape(-1, 3)
            
    #         # 创建节点映射
    #         node_dict = {tag: coord for tag, coord in zip(node_tags, nodes)}
            
    #         # 获取单元节点坐标
    #         cell_nodes = []
    #         for elem_nodes in hex_elements['nodes']:
    #             cell_coords = np.array([node_dict[node] for node in elem_nodes])
    #             cell_nodes.append(cell_coords)
            
    #         cell_nodes = np.array(cell_nodes)
            
    #         self.hex_mesh = {
    #             'nodes': nodes,
    #             'node_tags': node_tags,
    #             'elements': hex_elements['tags'],
    #             'element_nodes': hex_elements['nodes'],
    #             'cell_nodes': cell_nodes,
    #             'n_cells': len(hex_elements['tags'])
    #         }
            
    #         if self.verbose:
    #             print(f"  节点数: {len(node_tags)}")
    #             print(f"  HEX8单元数: {len(hex_elements['tags'])}")
            
    #         # 清除Gmsh中的网格，为后续几何创建腾出空间
    #         gmsh.clear()
            
    #     elif mesh_ext in ['.vtk', '.vtu', '.xdmf', '.h5']:
    #         # 使用meshio加载
    #         import meshio
    #         mesh = meshio.read(self.config.hex_mesh_file)
            
    #         # 查找HEX8单元
    #         hex8_cells = None
    #         for cell_block in mesh.cells:
    #             if cell_block.type == "hexahedron":
    #                 hex8_cells = cell_block
    #                 break
            
    #         if hex8_cells is None:
    #             raise ValueError("未在网格中找到六面体单元")
            
    #         self.hex_mesh = {
    #             'nodes': mesh.points,
    #             'elements': np.arange(len(hex8_cells.data)),
    #             'element_nodes': hex8_cells.data + 1,  # Gmsh使用1-based索引
    #             'cell_nodes': mesh.points[hex8_cells.data],
    #             'n_cells': len(hex8_cells.data)
    #         }
            
    #         if self.verbose:
    #             print(f"  节点数: {len(mesh.points)}")
    #             print(f"  HEX8单元数: {len(hex8_cells.data)}")
    #     else:
    #         raise ValueError(f"不支持的网格格式: {mesh_ext}")
        
    #     # 检查单元分配数组
    #     if len(self.config.unit_assignments) != self.hex_mesh['n_cells']:
    #         raise ValueError(f"单元分配数组长度({len(self.config.unit_assignments)})与单元数({self.hex_mesh['n_cells']})不匹配")
    
    
    def load_from_meshio(self, meshio_mesh):
        """从meshio网格对象加载"""
        if self.verbose:
            print("从meshio网格对象加载HEX8网格...")
        
        # 查找HEX8单元
        hex8_cells = None
        for cell_block in meshio_mesh.cells:
            if cell_block.type == "hexahedron":
                hex8_cells = cell_block
                break
        
        if hex8_cells is None:
            raise ValueError("未在网格中找到六面体单元")
        
        # 获取数据
        nodes = meshio_mesh.points
        connectivity = hex8_cells.data
        n_cells = len(connectivity)
        
        # 计算每个单元的顶点坐标
        cell_nodes = nodes[connectivity]  # 使用numpy高级索引
        
        self.hex_mesh = {
            'nodes': nodes,
            'node_tags': np.arange(1, len(nodes) + 1),
            'elements': np.arange(1, n_cells + 1),
            'element_nodes': connectivity + 1,  # 转换为1-based索引
            'cell_nodes': cell_nodes,
            'n_cells': n_cells
        }
        
        if self.verbose:
            print(f"  节点数: {len(nodes)}")
            print(f"  HEX8单元数: {n_cells}")


    def load_lattice_units(self):
        """加载点阵单元STP文件"""
        if self.verbose:
            print("加载点阵单元...")
        
        for i, unit in enumerate(self.config.lattice_units):
            if not os.path.exists(unit.stp_file):
                raise FileNotFoundError(f"STP文件不存在: {unit.stp_file}")
            
            # 导入STP文件
            entities = gmsh.model.occ.importShapes(unit.stp_file)
            gmsh.model.occ.synchronize()
            
            # 获取几何边界框，用于尺寸归一化
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            original_size = (xmax - xmin, ymax - ymin, zmax - zmin)
            
            # 归一化到单位尺寸
            if original_size != (1.0, 1.0, 1.0):
                scale_x = unit.dimensions[0] / original_size[0]
                scale_y = unit.dimensions[1] / original_size[1]
                scale_z = unit.dimensions[2] / original_size[2]
                
                # 移动到原点
                gmsh.model.occ.translate(entities, -xmin, -ymin, -zmin)
                # 缩放到目标尺寸
                gmsh.model.occ.dilate(entities, 0, 0, 0, scale_x, scale_y, scale_z)
                gmsh.model.occ.synchronize()
            
            # 获取几何实体
            dim_tags = gmsh.model.getEntities()
            volumes = [tag for dim, tag in dim_tags if dim == 3]
            surfaces = [tag for dim, tag in dim_tags if dim == 2]
            
            # 存储点阵单元信息
            self.lattice_units[unit.label] = {
                'stp_file': unit.stp_file,
                'volumes': volumes,
                'surfaces': surfaces,
                'dimensions': unit.dimensions,
                'index': i
            }
            
            if self.verbose:
                print(f"  点阵单元 '{unit.label}': {unit.stp_file}")
                print(f"    尺寸: {unit.dimensions}")
                print(f"    包含 {len(volumes)} 个体, {len(surfaces)} 个面")
    
    def compute_cell_transform(self, cell_nodes: np.ndarray) -> Dict:
        """
        计算从单位立方体到六面体单元的仿射变换
        
        参数:
        cell_nodes: 六面体单元的8个节点坐标，形状(8, 3)
                  节点顺序应符合Gmsh的HEX8顺序
        
        返回:
        变换参数
        """
        # 六面体单元的8个节点
        # Gmsh HEX8节点顺序: [0,1,2,3,4,5,6,7]
        # 对应单位立方体: (0,0,0), (1,0,0), (1,1,0), (0,1,0), 
        #                (0,0,1), (1,0,1), (1,1,1), (0,1,1)
        
        # 提取关键点
        P0 = cell_nodes[0]  # (0,0,0)
        P1 = cell_nodes[1]  # (1,0,0)
        P3 = cell_nodes[3]  # (0,1,0)
        P4 = cell_nodes[4]  # (0,0,1)
        
        # 计算局部坐标系
        u = P1 - P0  # X方向
        v = P3 - P0  # Y方向
        w = P4 - P0  # Z方向
        
        # 计算中心点
        center = np.mean(cell_nodes, axis=0)
        
        # 计算缩放
        scale_x = np.linalg.norm(u)
        scale_y = np.linalg.norm(v)
        scale_z = np.linalg.norm(z)
        
        # 计算旋转
        # 归一化方向向量
        u_norm = u / (scale_x + 1e-10)
        v_norm = v / (scale_y + 1e-10)
        w_norm = w / (scale_z + 1e-10)
        
        # 构建旋转矩阵
        R = np.column_stack([u_norm, v_norm, w_norm])
        
        return {
            'translation': P0,
            'rotation_matrix': R,
            'scaling': (scale_x, scale_y, scale_z),
            'center': center,
            'cell_nodes': cell_nodes
        }
    
    def apply_transform(self, volumes: List[int], transform: Dict) -> List[Tuple[int, int]]:
        """
        对几何体应用变换
        
        参数:
        volumes: 体积标签列表
        transform: 变换参数
        
        返回:
        变换后的体积标签列表
        """
        # 复制几何体
        new_volumes = gmsh.model.occ.copy([(3, vol) for vol in volumes])
        
        # 应用变换
        translation = transform['translation']
        R = transform['rotation_matrix']
        scaling = transform['scaling']
        
        # 1. 缩放到单位立方体
        gmsh.model.occ.dilate(new_volumes, 0, 0, 0, scaling[0], scaling[1], scaling[2])
        
        # 2. 应用旋转（通过旋转矩阵）
        # 计算旋转轴和角度
        # 注意：这里简化处理，实际旋转可能更复杂
        # 对于任意旋转矩阵，可以分解为绕坐标轴的旋转
        # 这里使用欧拉角ZYX顺序
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x_angle = np.arctan2(R[2, 1], R[2, 2])
            y_angle = np.arctan2(-R[2, 0], sy)
            z_angle = np.arctan2(R[1, 0], R[0, 0])
        else:
            x_angle = np.arctan2(-R[1, 2], R[1, 1])
            y_angle = np.arctan2(-R[2, 0], sy)
            z_angle = 0
        
        # 转换为角度
        x_angle_deg = np.degrees(x_angle)
        y_angle_deg = np.degrees(y_angle)
        z_angle_deg = np.degrees(z_angle)
        
        # 应用旋转（顺序：Z, Y, X）
        if abs(z_angle_deg) > 1e-6:
            gmsh.model.occ.rotate(new_volumes, 0, 0, 0, 0, 0, 1, z_angle_deg)
        if abs(y_angle_deg) > 1e-6:
            gmsh.model.occ.rotate(new_volumes, 0, 0, 0, 0, 1, 0, y_angle_deg)
        if abs(x_angle_deg) > 1e-6:
            gmsh.model.occ.rotate(new_volumes, 0, 0, 0, 1, 0, 0, x_angle_deg)
        
        # 3. 平移
        gmsh.model.occ.translate(new_volumes, translation[0], translation[1], translation[2])
        
        return new_volumes
    
    def generate_lattice(self, progress_interval: int = 100):
        """
        生成点阵结构
        
        参数:
        progress_interval: 进度显示间隔
        """
        if self.hex_mesh is None:
            self.load_hex_mesh()
        
        if not self.lattice_units:
            self.load_lattice_units()
        
        if self.verbose:
            print(f"生成点阵结构 ({self.hex_mesh['n_cells']} 个单元)...")
        
        n_cells = self.hex_mesh['n_cells']
        cell_nodes = self.hex_mesh['cell_nodes']
        unit_assignments = self.config.unit_assignments
        
        # 获取点阵单元标签列表
        unit_labels = list(self.lattice_units.keys())
        
        for i in range(n_cells):
            # 显示进度
            if self.verbose and (i + 1) % progress_interval == 0:
                print(f"  处理单元 {i+1}/{n_cells}")
            
            # 获取当前单元的点阵单元类型
            unit_idx = unit_assignments[i]
            if unit_idx >= len(unit_labels):
                if self.verbose:
                    print(f"  警告: 单元 {i} 的索引 {unit_idx} 超出范围，使用默认单元0")
                unit_label = unit_labels[0]
            else:
                unit_label = unit_labels[unit_idx]
            
            # 获取单元节点
            nodes = cell_nodes[i]
            
            # 计算变换
            transform = self.compute_cell_transform(nodes)
            
            # 获取点阵单元体积
            unit_volumes = self.lattice_units[unit_label]['volumes']
            
            # 应用变换并放置点阵单元
            new_volumes = self.apply_transform(unit_volumes, transform)
            
            # 存储实例信息
            self.instances.append({
                'cell_index': i,
                'unit_label': unit_label,
                'unit_index': unit_idx,
                'transform': transform,
                'volumes': [v[1] for v in new_volumes]
            })
        
        # 同步几何
        gmsh.model.occ.synchronize()
        
        if self.verbose:
            print(f"点阵结构生成完成，共 {len(self.instances)} 个实例")
    
    def set_mesh_controls(self, **kwargs):
        """设置网格控制参数"""
        if kwargs:
            self.config.mesh_controls.update(kwargs)
        
        controls = self.config.mesh_controls
        
        # 设置Gmsh网格参数
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", controls['min_size'])
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", controls['max_size'])
        gmsh.option.setNumber("Mesh.Algorithm", controls['mesh_algorithm'])
        gmsh.option.setNumber("Mesh.Optimize", 1 if controls.get('optimize', True) else 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if controls.get('optimize', True) else 0)
        
        # 高级参数
        if 'quality' in controls:
            gmsh.option.setNumber("Mesh.QualityType", controls['quality'])
        if 'optimize_threshold' in controls:
            gmsh.option.setNumber("Mesh.OptimizeThreshold", controls['optimize_threshold'])
        
        if self.verbose:
            print(f"设置网格参数:")
            print(f"  最小尺寸: {controls['min_size']}")
            print(f"  最大尺寸: {controls['max_size']}")
            print(f"  网格算法: {controls['mesh_algorithm']}")
            print(f"  优化: {controls.get('optimize', True)}")
    
    def define_physical_groups(self, 
                             by_unit_type: bool = True,
                             by_cell_group: bool = False,
                             cell_groups: Optional[np.ndarray] = None):
        """
        定义物理组
        
        参数:
        by_unit_type: 是否按点阵单元类型分组
        by_cell_group: 是否按单元组分组
        cell_groups: 单元分组数组，长度等于单元数
        """
        gmsh.model.occ.synchronize()
        
        if self.verbose:
            print("定义物理组...")
        
        # 获取所有体积
        all_volumes = gmsh.model.getEntities(3)
        
        if not all_volumes:
            print("警告: 未找到任何体积")
            return
        
        if by_unit_type:
            # 按点阵单元类型分组
            type_groups = {}
            for instance in self.instances:
                unit_label = instance['unit_label']
                if unit_label not in type_groups:
                    type_groups[unit_label] = []
                type_groups[unit_label].extend(instance['volumes'])
            
            # 为每种类型创建物理组
            for i, (unit_label, volumes) in enumerate(type_groups.items()):
                if volumes:
                    gmsh.model.addPhysicalGroup(3, volumes, i+1)
                    gmsh.model.setPhysicalName(3, i+1, f"unit_{unit_label}")
                    if self.verbose:
                        print(f"  物理组 {i+1}: 单元类型 '{unit_label}' ({len(volumes)} 个体积)")
        
        if by_cell_group and cell_groups is not None:
            # 按单元组分组
            group_groups = {}
            for instance in self.instances:
                cell_idx = instance['cell_index']
                group = cell_groups[cell_idx]
                if group not in group_groups:
                    group_groups[group] = []
                group_groups[group].extend(instance['volumes'])
            
            # 为每组创建物理组
            start_idx = len(type_groups) if by_unit_type else 1
            for i, (group, volumes) in enumerate(group_groups.items()):
                if volumes:
                    gmsh.model.addPhysicalGroup(3, volumes, start_idx + i)
                    gmsh.model.setPhysicalName(3, start_idx + i, f"group_{group}")
                    if self.verbose:
                        print(f"  物理组 {start_idx + i}: 单元组 {group} ({len(volumes)} 个体积)")
        
        # 定义外表面物理组
        all_surfaces = []
        for dim, tag in all_volumes:
            boundary = gmsh.model.getBoundary([(dim, tag)], oriented=False, recursive=False)
            all_surfaces.extend([surf[1] for surf in boundary if surf[0] == 2])
        
        unique_surfaces = list(set(all_surfaces))
        
        if unique_surfaces:
            surf_start_idx = 1000
            gmsh.model.addPhysicalGroup(2, unique_surfaces, surf_start_idx)
            gmsh.model.setPhysicalName(2, surf_start_idx, "external_surfaces")
            
            if self.verbose:
                print(f"  物理组 {surf_start_idx}: 外表面 ({len(unique_surfaces)} 个面)")
    
    def generate_mesh(self, 
                     dimension: int = 3,
                     order: int = 1,
                     refinement: int = 0,
                     save_path: Optional[str] = None) -> str:
        """
        生成体网格
        
        参数:
        dimension: 网格维度
        order: 单元阶数
        refinement: 细化次数
        save_path: 保存路径
        
        返回:
        网格文件路径
        """
        if self.verbose:
            print("生成网格...")
        
        # 同步几何
        gmsh.model.occ.synchronize()
        
        # 设置单元阶数
        gmsh.model.mesh.setOrder(order)
        
        # 生成网格
        for i in range(refinement + 1):
            if self.verbose and i > 0:
                print(f"  第 {i+1} 次细化...")
            gmsh.model.mesh.generate(dimension)
        
        # 优化网格
        if self.config.mesh_controls.get('optimize', True):
            if self.verbose:
                print("  优化网格...")
            gmsh.model.mesh.optimize("Netgen")
        
        # 生成默认保存路径
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.config.output_dir, f"lattice_mesh_{timestamp}.msh")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存网格
        gmsh.write(save_path)
        
        if self.verbose:
            print(f"网格已保存到: {save_path}")
        
        # 保存网格统计
        self.save_mesh_statistics(save_path)
        
        return save_path
    
    def save_mesh_statistics(self, mesh_path: str):
        """保存网格统计信息"""
        # 获取网格统计
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()
        
        stats = {
            'n_nodes': len(node_tags),
            'n_elements': sum(len(tags) for tags in elem_tags),
            'element_types': {},
            'instances': len(self.instances),
            'unit_types': {}
        }
        
        # 统计单元类型
        for elem_type, tags in zip(elem_types, elem_tags):
            type_name = gmsh.model.mesh.getElementProperties(elem_type)[0]
            stats['element_types'][type_name] = len(tags)
        
        # 统计点阵单元类型
        unit_counts = {}
        for instance in self.instances:
            unit_label = instance['unit_label']
            unit_counts[unit_label] = unit_counts.get(unit_label, 0) + 1
        stats['unit_counts'] = unit_counts
        
        # 保存统计到JSON
        stats_path = os.path.splitext(mesh_path)[0] + "_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        if self.verbose:
            print(f"网格统计:")
            print(f"  节点数: {stats['n_nodes']}")
            print(f"  单元总数: {stats['n_elements']}")
            for elem_type, count in stats['element_types'].items():
                print(f"  {elem_type}: {count}")
            print(f"  实例数: {stats['instances']}")
            for unit_type, count in unit_counts.items():
                print(f"  {unit_type}: {count}")
        
        return stats
    
    def export_geometry(self, filename: str):
        """导出几何到STEP格式"""
        gmsh.model.occ.synchronize()
        gmsh.write(filename)
        
        if self.verbose:
            print(f"几何已导出到: {filename}")
    
    def export_stl(self, filename: str):
        """导出表面网格到STL格式"""
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)  # 生成表面网格
        gmsh.write(filename)
        
        if self.verbose:
            print(f"STL表面网格已导出到: {filename}")
    
    def visualize(self, show_mesh: bool = False, interactive: bool = True):
        """可视化"""
        gmsh.model.occ.synchronize()
        
        if not show_mesh:
            gmsh.option.setNumber("Mesh.Visible", 0)
        
        if interactive:
            gmsh.fltk.run()
        else:
            # 保存截图
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(self.config.output_dir, f"visualization_{timestamp}.png")
            gmsh.fltk.initialize()
            gmsh.write(image_path)
            gmsh.fltk.finalize()
            print(f"可视化截图已保存到: {image_path}")
        
        gmsh.option.setNumber("Mesh.Visible", 1)
    
    def save_config(self, filename: str):
        """保存配置"""
        config_dict = {
            'hex_mesh_file': self.config.hex_mesh_file,
            'unit_assignments': self.config.unit_assignments.tolist() if isinstance(self.config.unit_assignments, np.ndarray) else self.config.unit_assignments,
            'lattice_units': [asdict(unit) for unit in self.config.lattice_units],
            'output_dir': self.config.output_dir,
            'mesh_controls': self.config.mesh_controls,
            'verbose': self.config.verbose
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        if self.verbose:
            print(f"配置已保存到: {filename}")
    
    def cleanup(self):
        """清理资源"""
        gmsh.finalize()


# ============================================================================
# 高级功能：网格处理工具
# ============================================================================

class HexMeshProcessor:
    """HEX8网格处理器"""
    
    @staticmethod
    def create_test_hex_mesh(nx: int = 3, ny: int = 3, nz: int = 3, 
                            dx: float = 10.0, dy: float = 10.0, dz: float = 10.0,
                            save_path: str = "test_hex_mesh.msh"):
        """
        创建测试用的HEX8网格
        
        参数:
        nx, ny, nz: 各方向单元数
        dx, dy, dz: 各方向尺寸
        save_path: 保存路径
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # 创建节点
        nodes = []
        node_id = 1
        node_map = {}
        
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    x = i * dx
                    y = j * dy
                    z = k * dz
                    nodes.append((node_id, x, y, z))
                    node_map[(i, j, k)] = node_id
                    node_id += 1
        
        # 创建立方体单元
        elements = []
        elem_id = 1
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # HEX8节点顺序
                    n1 = node_map[(i, j, k)]
                    n2 = node_map[(i+1, j, k)]
                    n3 = node_map[(i+1, j+1, k)]
                    n4 = node_map[(i, j+1, k)]
                    n5 = node_map[(i, j, k+1)]
                    n6 = node_map[(i+1, j, k+1)]
                    n7 = node_map[(i+1, j+1, k+1)]
                    n8 = node_map[(i, j+1, k+1)]
                    
                    elements.append((elem_id, n1, n2, n3, n4, n5, n6, n7, n8))
                    elem_id += 1
        
        # 写入Gmsh格式
        with open(save_path, 'w') as f:
            f.write("$MeshFormat\n")
            f.write("2.2 0 8\n")
            f.write("$EndMeshFormat\n")
            f.write("$Nodes\n")
            f.write(f"{len(nodes)}\n")
            for nid, x, y, z in nodes:
                f.write(f"{nid} {x} {y} {z}\n")
            f.write("$EndNodes\n")
            f.write("$Elements\n")
            f.write(f"{len(elements)}\n")
            for eid, *nids in elements:
                f.write(f"{eid} 5 2 0 0 {nids[0]} {nids[1]} {nids[2]} {nids[3]} {nids[4]} {nids[5]} {nids[6]} {nids[7]}\n")
            f.write("$EndElements\n")
        
        gmsh.finalize()
        
        print(f"测试HEX8网格已创建: {save_path}")
        print(f"  单元数: {nx * ny * nz}")
        print(f"  节点数: {(nx + 1) * (ny + 1) * (nz + 1)}")
        
        return save_path
    
    @staticmethod
    def load_hex_mesh_info(mesh_file: str) -> Dict:
        """加载HEX8网格信息"""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        gmsh.open(mesh_file)
        
        # 获取节点
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = node_coords.reshape(-1, 3)
        
        # 获取HEX8单元
        hex8_type = 5
        element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()
        
        hex_elements = None
        for elem_type, elem_tags, elem_nodes in zip(element_types, element_tags, element_node_tags):
            if elem_type == hex8_type:
                hex_elements = {
                    'tags': elem_tags,
                    'nodes': elem_nodes.reshape(-1, 8)
                }
                break
        
        gmsh.finalize()
        
        if hex_elements is None:
            raise ValueError("未找到HEX8单元")
        
        info = {
            'nodes': nodes,
            'node_tags': node_tags,
            'elements': hex_elements['tags'],
            'element_nodes': hex_elements['nodes'],
            'n_cells': len(hex_elements['tags'])
        }
        
        return info


# ============================================================================
# 使用示例
# ============================================================================

def example_basic():
    """基本使用示例"""
    import numpy as np
    
    print("=" * 60)
    print("示例: HEX8网格驱动的点阵结构生成")
    print("=" * 60)
    
    # 1. 创建测试HEX8网格
    print("\n1. 创建测试HEX8网格...")
    hex_mesh_file = HexMeshProcessor.create_test_hex_mesh(
        nx=4, ny=3, nz=2,
        dx=8.0, dy=8.0, dz=8.0,
        save_path="test_mesh.hex8.msh"
    )
    
    # 2. 创建点阵单元配置
    lattice_units = [
        LatticeUnit(stp_file="models/truss_unit.stp", label="truss", dimensions=(8.0, 8.0, 8.0)),
        LatticeUnit(stp_file="models/gyroid_unit.stp", label="gyroid", dimensions=(8.0, 8.0, 8.0)),
        LatticeUnit(stp_file="models/bcc_unit.stp", label="bcc", dimensions=(8.0, 8.0, 8.0)),
        LatticeUnit(stp_file="models/fcc_unit.stp", label="fcc", dimensions=(8.0, 8.0, 8.0)),
        LatticeUnit(stp_file="models/diamond_unit.stp", label="diamond", dimensions=(8.0, 8.0, 8.0))
    ]
    
    # 3. 创建单元分配数组
    # 假设我们有4x3x2=24个单元
    n_cells = 4 * 3 * 2
    # 随机分配点阵单元类型 (0-4)
    np.random.seed(42)  # 固定随机种子以便重现
    unit_assignments = np.random.randint(0, 5, n_cells)
    
    # 也可以创建特定模式
    # unit_assignments = np.array([
    #     0, 1, 2, 3, 4, 0, 1, 2,
    #     3, 4, 0, 1, 2, 3, 4, 0,
    #     1, 2, 3, 4, 0, 1, 2, 3
    # ])
    
    # 4. 创建配置
    config = LatticeConfig(
        hex_mesh_file=hex_mesh_file,
        unit_assignments=unit_assignments,
        lattice_units=lattice_units,
        output_dir="output/basic_example",
        mesh_controls={
            'min_size': 0.5,
            'max_size': 2.0,
            'mesh_algorithm': 6,
            'optimize': True,
            'optimize_threshold': 0.3
        },
        verbose=True
    )
    
    # 5. 创建生成器并生成点阵
    print("\n2. 生成点阵结构...")
    generator = HexMeshDrivenLatticeGenerator(config)
    
    # 如果STP文件不存在，创建测试几何
    for unit in lattice_units:
        if not os.path.exists(unit.stp_file):
            print(f"创建测试STP文件: {unit.stp_file}")
            os.makedirs(os.path.dirname(unit.stp_file), exist_ok=True)
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            
            # 创建不同的测试几何
            if "truss" in unit.label:
                # 桁架结构
                gmsh.model.occ.addBox(1, 1, 1, 6, 6, 6)
                gmsh.model.occ.addCylinder(1, 1, 1, 6, 0, 0, 0.5)
                gmsh.model.occ.addCylinder(1, 1, 1, 0, 6, 0, 0.5)
                gmsh.model.occ.addCylinder(1, 1, 1, 0, 0, 6, 0.5)
                gmsh.model.occ.fuse([(3,1)], [(3,2), (3,3), (3,4)])
            elif "gyroid" in unit.label:
                # 简化螺旋结构
                gmsh.model.occ.addSphere(4, 4, 4, 3.5)
                gmsh.model.occ.cut([(3,1)], [(3,2)])
            elif "bcc" in unit.label:
                # 体心立方
                gmsh.model.occ.addBox(1, 1, 1, 6, 6, 6)
                gmsh.model.occ.addSphere(4, 4, 4, 2)
                gmsh.model.occ.fuse([(3,1)], [(3,2)])
            elif "fcc" in unit.label:
                # 面心立方
                gmsh.model.occ.addBox(1, 1, 1, 6, 6, 6)
                gmsh.model.occ.addSphere(4, 4, 1, 1.5)
                gmsh.model.occ.addSphere(4, 1, 4, 1.5)
                gmsh.model.occ.addSphere(1, 4, 4, 1.5)
                gmsh.model.occ.fuse([(3,1)], [(3,2), (3,3), (3,4)])
            elif "diamond" in unit.label:
                # 金刚石结构
                gmsh.model.occ.addBox(2, 2, 2, 4, 4, 4)
                gmsh.model.occ.addSphere(4, 4, 4, 1.5)
                gmsh.model.occ.addSphere(4, 4, 2, 1.5)
                gmsh.model.occ.addSphere(4, 2, 4, 1.5)
                gmsh.model.occ.addSphere(2, 4, 4, 1.5)
                gmsh.model.occ.fuse([(3,1)], [(3,2), (3,3), (3,4), (3,5)])
            
            gmsh.model.occ.synchronize()
            gmsh.write(unit.stp_file)
            gmsh.finalize()
    
    # 生成点阵结构
    generator.load_lattice_units()
    generator.generate_lattice(progress_interval=10)
    
    # 6. 定义物理组
    print("\n3. 定义物理组...")
    generator.define_physical_groups(by_unit_type=True)
    
    # 7. 生成网格
    print("\n4. 生成体网格...")
    mesh_file = generator.generate_mesh(
        dimension=3,
        order=1,
        save_path=os.path.join(config.output_dir, "lattice_mesh.msh")
    )
    
    # 8. 导出几何
    print("\n5. 导出几何和网格...")
    generator.export_geometry(os.path.join(config.output_dir, "lattice_geometry.step"))
    generator.export_stl(os.path.join(config.output_dir, "lattice_surface.stl"))
    
    # 9. 保存配置
    generator.save_config(os.path.join(config.output_dir, "lattice_config.json"))
    
    # 10. 清理
    generator.cleanup()
    
    print("\n" + "=" * 60)
    print("生成完成!")
    print(f"网格文件: {mesh_file}")
    print("=" * 60)
    
    return mesh_file


def example_with_custom_mesh():
    """使用自定义HEX8网格的示例"""
    import numpy as np
    
    print("=" * 60)
    print("示例: 自定义HEX8网格点阵结构")
    print("=" * 60)
    
    # 1. 创建不规则HEX8网格
    print("\n1. 创建不规则HEX8网格...")
    
    # 这里创建一个简单的2x2x2网格，但节点位置不规则
    nodes = np.array([
        [0, 0, 0],    # 节点1
        [10, 0, 0],   # 节点2
        [0, 12, 0],   # 节点3
        [10, 12, 0],  # 节点4
        [0, 0, 8],    # 节点5
        [10, 0, 8],   # 节点6
        [0, 12, 8],   # 节点7
        [10, 12, 8],  # 节点8
        [5, 6, 4],    # 节点9 (扭曲的网格)
    ])
    
    # HEX8单元连接 (1-based索引)
    # 单元1: 节点1-2-4-3-5-6-8-7
    # 单元2: 节点2-9-7-4-6-5-8-6 (示例，实际应根据拓扑)
    
    # 由于创建复杂网格需要更多代码，这里使用简单的测试网格
    hex_mesh_file = "custom_hex_mesh.msh"
    
    # 2. 定义点阵单元
    lattice_units = [
        LatticeUnit(stp_file="models/simple_cube.stp", label="cube", dimensions=(5.0, 5.0, 5.0)),
        LatticeUnit(stp_file="models/simple_sphere.stp", label="sphere", dimensions=(5.0, 5.0, 5.0)),
    ]
    
    # 3. 单元分配
    unit_assignments = np.array([0, 1, 0, 1])  # 交替分配
    
    # 4. 创建配置
    config = LatticeConfig(
        hex_mesh_file=hex_mesh_file,
        unit_assignments=unit_assignments,
        lattice_units=lattice_units,
        output_dir="output/custom_example",
        mesh_controls={
            'min_size': 0.3,
            'max_size': 1.5,
            'mesh_algorithm': 5
        },
        verbose=True
    )
    
    # 5. 生成
    generator = HexMeshDrivenLatticeGenerator(config)
    
    # 创建测试STP文件
    for unit in lattice_units:
        if not os.path.exists(unit.stp_file):
            os.makedirs(os.path.dirname(unit.stp_file), exist_ok=True)
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            
            if "cube" in unit.label:
                gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
            else:  # sphere
                gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.5)
            
            gmsh.model.occ.synchronize()
            gmsh.write(unit.stp_file)
            gmsh.finalize()
    
    generator.load_lattice_units()
    generator.generate_lattice()
    generator.define_physical_groups(by_unit_type=True)
    
    mesh_file = generator.generate_mesh(
        save_path=os.path.join(config.output_dir, "custom_lattice.msh")
    )
    
    generator.save_config(os.path.join(config.output_dir, "custom_config.json"))
    generator.cleanup()
    
    return mesh_file


def example_from_vtk_mesh():
    """从VTK网格文件加载的示例"""
    import numpy as np
    import meshio
    
    print("=" * 60)
    print("示例: 从VTK网格文件生成点阵")
    print("=" * 60)
    
    # 创建VTK格式的测试网格
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1