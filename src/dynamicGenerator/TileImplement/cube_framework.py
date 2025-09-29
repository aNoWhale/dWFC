import cadquery as cq
import numpy as np
from typing import List, Tuple
import time
from tqdm import tqdm
import sys



def create_cube_framework(vertices: List[Tuple[float, float, float]], 
                         cylinder_radius: float = 0.05, 
                         cylinder_segments: int = 16,
                         add_vertex_spheres: bool = True,
                         sphere_radius: float = None) -> cq.Workplane:
    """
    根据立方体的8个顶点坐标生成由12根圆柱体构成的立方体框架，并可选择在顶点添加小球
    
    Args:
        vertices: 立方体的8个顶点坐标列表 [(x,y,z), ...]
        cylinder_radius: 圆柱体半径
        cylinder_segments: 圆柱体分段数
        add_vertex_spheres: 是否在顶点添加小球
        sphere_radius: 小球半径，如果为None则使用cylinder_radius
        
    Returns:
        CadQuery Workplane对象
    """
    if len(vertices) != 8:
        raise ValueError("必须提供8个顶点坐标")
    
    # 如果未指定球半径，使用圆柱体半径
    if sphere_radius is None:
        sphere_radius = cylinder_radius
    
    # 立方体的12条边的连接关系
    edges = [
        # 底面4条边
        (0, 1), (1, 2), (2, 3), (3, 0),
        # 顶面4条边  
        (4, 5), (5, 6), (6, 7), (7, 4),
        # 垂直4条边
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # 创建工作平面
    result = cq.Workplane("XY")
    
    # 为每条边创建圆柱体
    for i, (start_idx, end_idx) in enumerate(edges):
        start_point = vertices[start_idx]
        end_point = vertices[end_idx]
        
        # 计算圆柱体长度和方向
        vector = np.array(end_point) - np.array(start_point)
        length = np.linalg.norm(vector)
        
        if length == 0:
            continue
            
        # 计算中点作为圆柱体中心
        center = (np.array(start_point) + np.array(end_point)) / 2
        
        # 创建圆柱体并定位（减少分段数提升性能）
        cylinder = (cq.Workplane("XY")
                   .transformed(offset=cq.Vector(*center))
                   .cylinder(length, cylinder_radius, centered=True))
        
        # 计算旋转角度使圆柱体沿着边的方向
        direction = vector / length
        z_axis = np.array([0, 0, 1])
        
        # 如果不平行于z轴，需要旋转
        if not np.allclose(np.abs(np.dot(direction, z_axis)), 1.0):
            # 计算旋转轴（叉积）
            rotation_axis = np.cross(z_axis, direction)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                # 计算旋转角度
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                angle_deg = np.degrees(angle)
                
                # 应用旋转
                cylinder = cylinder.rotate(
                    axisStartPoint=cq.Vector(*center),
                    axisEndPoint=cq.Vector(*(center + rotation_axis)),
                    angleDegrees=angle_deg
                )
        
        # 合并到结果中
        if i == 0:
            result = cylinder
        else:
            result = result.union(cylinder)
    
    # 在每个顶点添加小球
    if add_vertex_spheres:
        for vertex in vertices:
            sphere = (cq.Workplane("XY")
                     .transformed(offset=cq.Vector(*vertex))
                     .sphere(sphere_radius))
            result = result.union(sphere)
    
    return result


def create_standard_cube_framework(size: float = 1.0, 
                                 cylinder_radius: float = 0.05,
                                 add_vertex_spheres: bool = True,
                                 sphere_radius: float = None) -> cq.Workplane:
    """
    创建标准立方体框架（边长为size的正立方体）
    
    Args:
        size: 立方体边长
        cylinder_radius: 圆柱体半径
        add_vertex_spheres: 是否在顶点添加小球
        sphere_radius: 小球半径，如果为None则使用cylinder_radius
        
    Returns:
        CadQuery Workplane对象
    """
    half_size = size / 2
    
    # 标准立方体的8个顶点
    vertices = [
        (-half_size, -half_size, -half_size),  # 0: 左下后
        (half_size, -half_size, -half_size),   # 1: 右下后
        (half_size, half_size, -half_size),    # 2: 右上后
        (-half_size, half_size, -half_size),   # 3: 左上后
        (-half_size, -half_size, half_size),   # 4: 左下前
        (half_size, -half_size, half_size),    # 5: 右下前
        (half_size, half_size, half_size),     # 6: 右上前
        (-half_size, half_size, half_size),    # 7: 左上前
    ]
    
    return create_cube_framework(vertices, cylinder_radius, 16, add_vertex_spheres, sphere_radius)


def show_progress_bar(current: int, total: int, bar_length: int = 50, prefix: str = "Progress"):
    """
    显示进度条
    
    Args:
        current: 当前进度
        total: 总数
        bar_length: 进度条长度
        prefix: 进度条前缀文本
    """
    percent = float(current) * 100 / total
    arrow = '█' * int(percent / 100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f'\r{prefix}: [{arrow + spaces}] {percent:.1f}% ({current}/{total})')
    sys.stdout.flush()
    
    if current == total:
        print()  # 完成后换行


def create_shared_cube_grid_optimized(grid_size: Tuple[int, int, int] = (100, 100, 100),
                                     cube_size: float = 1.0,
                                     cylinder_radius: float = 0.05,
                                     cylinder_segments: int = 8,  # 减少分段数提升性能
                                     add_vertex_spheres: bool = True,
                                     sphere_radius: float = None,
                                     batch_size: int = 1000,
                                     show_progress: bool = True) -> cq.Workplane:
    """
    高性能创建大规模共享框架的立方体网格
    优化策略：批量创建、减少union操作、预分配内存
    
    Args:
        grid_size: 网格尺寸 (x, y, z)，支持大规模如(100, 100, 100)
        cube_size: 每个立方体的边长
        cylinder_radius: 圆杆半径
        add_vertex_spheres: 是否在顶点添加小球
        sphere_radius: 小球半径，如果为None则使用cylinder_radius
        batch_size: 批量处理大小，用于控制内存使用
        show_progress: 是否显示进度条
        
    Returns:
        合并后的CadQuery Workplane对象
    """
    if sphere_radius is None:
        sphere_radius = cylinder_radius
        
    nx, ny, nz = grid_size
    
    if show_progress:
        print(f"正在生成 {nx}x{ny}x{nz} 大规模共享框架立方体网格...")
        print(f"预计顶点数: {(nx+1)*(ny+1)*(nz+1):,}")
    
    # 使用更高效的方法计算所有唯一的顶点
    vertices = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                x = (i - nx / 2) * cube_size
                y = (j - ny / 2) * cube_size  
                z = (k - nz / 2) * cube_size
                vertices.append((x, y, z))
    
    if show_progress:
        print(f"计算得到 {len(vertices):,} 个唯一顶点")
    
    # 使用更高效的边计算方法
    edges = set()
    
    if show_progress:
        print("计算唯一边...")
        cube_iter = tqdm(range(nx * ny * nz), desc="分析立方体", unit="cube")
    else:
        cube_iter = range(nx * ny * nz)
    
    # 预计算边的模式以提高效率
    edge_patterns = [
        # 底面4条边 (相对偏移)
        ((0, 0, 0), (1, 0, 0)),  # x方向
        ((1, 0, 0), (1, 1, 0)),  # y方向  
        ((1, 1, 0), (0, 1, 0)),  # -x方向
        ((0, 1, 0), (0, 0, 0)),  # -y方向
        # 顶面4条边
        ((0, 0, 1), (1, 0, 1)),
        ((1, 0, 1), (1, 1, 1)),
        ((1, 1, 1), (0, 1, 1)),
        ((0, 1, 1), (0, 0, 1)),
        # 垂直4条边
        ((0, 0, 0), (0, 0, 1)),
        ((1, 0, 0), (1, 0, 1)),
        ((1, 1, 0), (1, 1, 1)),
        ((0, 1, 0), (0, 1, 1)),
    ]
    
    for cube_idx in cube_iter:
        i = cube_idx // (ny * nz)
        j = (cube_idx % (ny * nz)) // nz
        k = cube_idx % nz
        
        # 当前立方体的基础坐标
        x_base = (i - nx / 2) * cube_size
        y_base = (j - ny / 2) * cube_size
        z_base = (k - nz / 2) * cube_size
        
        # 使用预定义模式快速生成边
        for (dx1, dy1, dz1), (dx2, dy2, dz2) in edge_patterns:
            v1 = (x_base + dx1 * cube_size, y_base + dy1 * cube_size, z_base + dz1 * cube_size)
            v2 = (x_base + dx2 * cube_size, y_base + dy2 * cube_size, z_base + dz2 * cube_size)
            # 确保边的顺序一致
            if v1 > v2:
                v1, v2 = v2, v1
            edges.add((v1, v2))
    
    edges = list(edges)
    
    if show_progress:
        print(f"计算得到 {len(edges):,} 条唯一边")
    
    # 批量创建圆杆以提高性能
    start_time = time.time()
    cylinders = []
    
    if show_progress:
        pbar = tqdm(edges, desc="创建圆杆", unit="杆", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = edges
    
    # 批量创建圆杆
    for edge in pbar:
        start_point, end_point = edge
        
        vector = np.array(end_point) - np.array(start_point)
        length = np.linalg.norm(vector)
        
        if length == 0:
            continue
            
        center = (np.array(start_point) + np.array(end_point)) / 2
        
        # 创建圆柱体（减少分段数提升性能）
        cylinder = (cq.Workplane("XY")
                   .transformed(offset=cq.Vector(*center))
                   .cylinder(length, cylinder_radius, centered=True))
        
        # 计算旋转
        direction = vector / length
        z_axis = np.array([0, 0, 1])
        
        if not np.allclose(np.abs(np.dot(direction, z_axis)), 1.0):
            rotation_axis = np.cross(z_axis, direction)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                angle_deg = np.degrees(angle)
                
                cylinder = cylinder.rotate(
                    axisStartPoint=cq.Vector(*center),
                    axisEndPoint=cq.Vector(*(center + rotation_axis)),
                    angleDegrees=angle_deg
                )
        
        cylinders.append(cylinder)
    
    # 批量合并圆杆以提高性能
    if show_progress:
        print("批量合并圆杆...")
        
    result = None
    for i in tqdm(range(0, len(cylinders), batch_size), desc="合并批次", disable=not show_progress):
        batch = cylinders[i:i + batch_size]
        
        # 在批次内部使用树形合并算法
        current_batch = batch.copy()
        while len(current_batch) > 1:
            next_batch = []
            for i in range(0, len(current_batch), 2):
                if i + 1 < len(current_batch):
                    merged = current_batch[i].union(current_batch[i + 1])
                    next_batch.append(merged)
                else:
                    next_batch.append(current_batch[i])
            current_batch = next_batch
        batch_result = current_batch[0]
        
        # 将批次结果合并到总结果
        if result is None:
            result = batch_result
        else:
            result = result.union(batch_result)
    
    # 批量添加顶点球
    if add_vertex_spheres:
        if show_progress:
            print("批量创建顶点球...")
            
        spheres = []
        for vertex in tqdm(vertices, desc="创建顶点球", disable=not show_progress):
            sphere = (cq.Workplane("XY")
                     .transformed(offset=cq.Vector(*vertex))
                     .sphere(sphere_radius))
            spheres.append(sphere)
        
        # 批量合并球体
        if show_progress:
            print("批量合并顶点球...")
            
        for i in tqdm(range(0, len(spheres), batch_size), desc="合并球体批次", disable=not show_progress):
            batch = spheres[i:i + batch_size]
            
            # 球体批次也使用树形合并
            current_batch = batch.copy()
            while len(current_batch) > 1:
                next_batch = []
                for i in range(0, len(current_batch), 2):
                    if i + 1 < len(current_batch):
                        merged = current_batch[i].union(current_batch[i + 1])
                        next_batch.append(merged)
                    else:
                        next_batch.append(current_batch[i])
                current_batch = next_batch
            batch_result = current_batch[0]
            
            result = result.union(batch_result)
    
    if show_progress:
        total_time = time.time() - start_time
        total_operations = len(edges) + (len(vertices) if add_vertex_spheres else 0)
        ops_per_sec = total_operations / total_time if total_time > 0 else 0
        print(f"大规模共享框架网格创建完成！")
        print(f"总耗时: {total_time:.2f}秒, 平均速度: {ops_per_sec:.1f} 结构/秒")
        print(f"内存批处理大小: {batch_size}")
    
    return result


def export_model(workplane: cq.Workplane, filename: str, format: str = "step"):
    """
    导出模型到文件
    
    Args:
        workplane: CadQuery工作平面对象
        filename: 文件名（不含扩展名）
        format: 导出格式 ("step", "stl", "3mf")
    """
    if format.lower() == "step":
        cq.exporters.export(workplane, f"{filename}.step")
    elif format.lower() == "stl":
        cq.exporters.export(workplane, f"{filename}.stl")
    elif format.lower() == "3mf":
        cq.exporters.export(workplane, f"{filename}.3mf")
    else:
        raise ValueError(f"不支持的格式: {format}")


if __name__ == "__main__":
    # 可调节的网格规模参数
    GRID_SIZE = (4, 4, 4)  # 可修改为任意规模，如 (100, 100, 100)
    CUBE_SIZE = 1.0
    CYLINDER_RADIUS = 0.02
    ADD_VERTEX_SPHERES = False  # 大规模时建议设为False
    SPHERE_RADIUS = None
    BATCH_SIZE = 50  # 批处理大小，可根据内存调整
    
    print(f"正在创建 {GRID_SIZE[0]}x{GRID_SIZE[1]}x{GRID_SIZE[2]} 大规模共享框架立方体网格...")
    print(f"配置参数:")
    print(f"- 网格规模: {GRID_SIZE}")
    print(f"- 立方体尺寸: {CUBE_SIZE}")
    print(f"- 圆杆半径: {CYLINDER_RADIUS}")
    print(f"- 顶点球: {'开启' if ADD_VERTEX_SPHERES else '关闭'}")
    print(f"- 批处理大小: {BATCH_SIZE}")
    
    # 预估规模
    total_cubes = GRID_SIZE[0] * GRID_SIZE[1] * GRID_SIZE[2]
    total_vertices = (GRID_SIZE[0] + 1) * (GRID_SIZE[1] + 1) * (GRID_SIZE[2] + 1)
    estimated_edges = total_cubes * 12 // 2  # 粗略估计
    
    print(f"\n预估规模:")
    print(f"- 立方体数量: {total_cubes:,}")
    print(f"- 顶点数量: {total_vertices:,}")
    print(f"- 预估边数: {estimated_edges:,}")
    
    # 内存和性能建议
    if total_cubes > 1000000:  # 100万+
        print(f"\n⚠️  大规模警告: 超过100万个立方体")
        print("建议:")
        print("- 确保有足够内存 (16GB+)")
        print("- 处理时间可能需要数小时")
        print("- 考虑关闭顶点球以节省内存")
        print("- 可以增大批处理大小到5000+")
    elif total_cubes > 100000:  # 10万+
        print(f"\n💡 中等规模: 约{total_cubes//1000}K个立方体")
        print("预计处理时间: 10-60分钟")
    else:
        print(f"\n✅ 小规模: {total_cubes}个立方体，处理应该很快")
    
    # 创建共享框架立方体网格
    shared_cube_grid = create_shared_cube_grid_optimized(
        grid_size=GRID_SIZE,
        cube_size=CUBE_SIZE,
        cylinder_radius=CYLINDER_RADIUS,
        cylinder_segments=8,  # 减少分段数提升性能
        add_vertex_spheres=ADD_VERTEX_SPHERES,
        sphere_radius=SPHERE_RADIUS,
        batch_size=BATCH_SIZE,
        show_progress=True
    )
    
    # 生成文件名
    filename = f"{GRID_SIZE[0]}x{GRID_SIZE[1]}x{GRID_SIZE[2]}_shared_framework_grid"
    
    # 导出模型
    print(f"\n开始导出模型到 {filename}.step...")
    export_model(shared_cube_grid, filename, "step")
    
    print(f"\n🎉 {GRID_SIZE[0]}x{GRID_SIZE[1]}x{GRID_SIZE[2]} 共享框架网格模型已创建并导出完成！")
    print(f"文件: {filename}.step")
    
    print(f"\n📊 最终统计:")
    print(f"- 成功生成了 {total_cubes:,} 个立方体的共享框架网格")
    print("- 相邻立方体之间的圆杆是重合的，避免了重复建模")
    print(f"- 使用了批量处理策略，批次大小: {BATCH_SIZE}")
    
    print(f"\n🔧 如需修改规模，请编辑代码中的 GRID_SIZE 参数")
    print("常用规模建议:")
    print("- 测试: (10, 10, 10)")
    print("- 中等: (50, 50, 50)")
    print("- 大型: (100, 100, 100)")
    print("- 超大: (200, 200, 200)")
    
    # 可视化建议
    if total_cubes <= 8000:  # 20x20x20以下
        try:
            from jupyter_cadquery import show
            print(f"\n正在显示 {GRID_SIZE[0]}x{GRID_SIZE[1]}x{GRID_SIZE[2]} 共享框架网格...")
            show(shared_cube_grid)
        except ImportError:
            print("要显示模型，请安装jupyter-cadquery: pip install jupyter-cadquery")
    else:
        print(f"\n⚠️  模型规模较大，跳过可视化显示以节省内存")