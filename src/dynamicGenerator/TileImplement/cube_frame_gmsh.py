import gmsh
import numpy as np

# 初始化Gmsh
gmsh.initialize()
gmsh.model.add("cube_frame")

# 参数设置
cube_size = 1.0        # 立方体边长
cylinder_radius = 0.05  # 圆杆半径
sphere_radius = 0.08    # 球体半径
mesh_size = 0.02        # 网格尺寸

# 定义立方体的8个顶点坐标
vertices = [
    [0, 0, 0],
    [cube_size, 0, 0],
    [cube_size, cube_size, 0],
    [0, cube_size, 0],
    [0, 0, cube_size],
    [cube_size, 0, cube_size],
    [cube_size, cube_size, cube_size],
    [0, cube_size, cube_size]
]

# 定义立方体的12条边（通过顶点索引）
edges = [
    # 底面的4条边
    [0, 1], [1, 2], [2, 3], [3, 0],
    # 顶面的4条边
    [4, 5], [5, 6], [6, 7], [7, 4],
    # 垂直的4条边
    [0, 4], [1, 5], [2, 6], [3, 7]
]

# 存储所有几何体的标签
sphere_tags = []
cylinder_tags = []

# 在每个顶点创建球体
for i, vertex in enumerate(vertices):
    sphere_tag = gmsh.model.occ.addSphere(vertex[0], vertex[1], vertex[2], sphere_radius)
    sphere_tags.append((3, sphere_tag))  # 3表示三维体

# 创建12条边（圆柱体）
for edge in edges:
    start = vertices[edge[0]]
    end = vertices[edge[1]]
    
    # 计算圆柱体的方向和长度
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    
    # 创建圆柱体
    cylinder_tag = gmsh.model.occ.addCylinder(
        start[0], start[1], start[2],  # 起点
        direction[0], direction[1], direction[2],  # 方向向量
        cylinder_radius  # 半径
    )
    cylinder_tags.append((3, cylinder_tag))

# 合并所有几何体（布尔并集）
all_volumes = sphere_tags + cylinder_tags
result, result_map = gmsh.model.occ.fuse(all_volumes[:1], all_volumes[1:])

# 同步几何模型
gmsh.model.occ.synchronize()

# 添加物理组
gmsh.model.addPhysicalGroup(3, [tag for dim, tag in result], name="CubeFrame")

# 设置网格尺寸
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

# 生成3D网格
gmsh.model.mesh.generate(3)

# 可视化（可选）
gmsh.fltk.run()

# 保存网格文件（可选）
gmsh.write("cube_frame.msh")
gmsh.write("cube_frame.step")  # 也可以保存为STL格式

# 获取网格信息
print(f"网格节点数: {len(gmsh.model.mesh.getNodes()[0])}")
print(f"网格单元数: {len(gmsh.model.mesh.getElements()[1])}")

# 清理
gmsh.finalize()