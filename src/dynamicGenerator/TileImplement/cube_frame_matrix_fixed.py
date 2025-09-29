import gmsh
import numpy as np
import time

# 初始化Gmsh
gmsh.initialize()
gmsh.model.add("cube_frame_matrix")

# 参数设置
nx, ny, nz = 10, 10, 10  # 立方体数量
cube_size = 1.0        # 单个立方体边长
cylinder_radius = 0.05  # 圆杆半径
sphere_radius = 0.08    # 球体半径
mesh_size = 0.1         # 网格尺寸

# 设置容差以避免几何问题
gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)

print(f"开始生成 {nx}x{ny}x{nz} 立方体框架矩阵...")
print(f"将创建 {(nx+1)*(ny+1)*(nz+1)} 个球体（顶点）")
print(f"将创建 {nx*(ny+1)*(nz+1) + ny*(nx+1)*(nz+1) + nz*(nx+1)*(ny+1)} 根圆杆（边）")

start_time = time.time()

sphere_tags = []
cylinder_tags = []

# 创建所有顶点的球体
print("创建球体...")
for i in range(nx + 1):
    for j in range(ny + 1):
        for k in range(nz + 1):
            x = i * cube_size
            y = j * cube_size
            z = k * cube_size
            
            sphere_tag = gmsh.model.occ.addSphere(x, y, z, sphere_radius)
            sphere_tags.append((3, sphere_tag))

print(f"球体创建完成，共 {len(sphere_tags)} 个")

# 创建所有的圆杆（边）
print("创建圆杆...")

# X方向的边
for i in range(nx):
    for j in range(ny + 1):
        for k in range(nz + 1):
            start_x = i * cube_size
            start_y = j * cube_size
            start_z = k * cube_size
            
            cylinder_tag = gmsh.model.occ.addCylinder(
                start_x, start_y, start_z,
                cube_size, 0, 0,
                cylinder_radius
            )
            cylinder_tags.append((3, cylinder_tag))

# Y方向的边
for i in range(nx + 1):
    for j in range(ny):
        for k in range(nz + 1):
            start_x = i * cube_size
            start_y = j * cube_size
            start_z = k * cube_size
            
            cylinder_tag = gmsh.model.occ.addCylinder(
                start_x, start_y, start_z,
                0, cube_size, 0,
                cylinder_radius
            )
            cylinder_tags.append((3, cylinder_tag))

# Z方向的边
for i in range(nx + 1):
    for j in range(ny + 1):
        for k in range(nz):
            start_x = i * cube_size
            start_y = j * cube_size
            start_z = k * cube_size
            
            cylinder_tag = gmsh.model.occ.addCylinder(
                start_x, start_y, start_z,
                0, 0, cube_size,
                cylinder_radius
            )
            cylinder_tags.append((3, cylinder_tag))

print(f"圆杆创建完成，共 {len(cylinder_tags)} 根")

# 合并所有几何体 - 使用fragment代替fuse以避免周期性表面问题
print("合并几何体...")
all_volumes = sphere_tags + cylinder_tags

try:
    # 方法1：使用fragment（推荐，避免周期性表面）
    print("使用fragment方法合并...")
    result, result_map = gmsh.model.occ.fragment(all_volumes, [])
    print("几何体合并完成（fragment）")
except:
    # 方法2：如果fragment失败，尝试分批fuse
    print("Fragment失败，尝试fuse方法...")
    batch_size = 50
    merged = [all_volumes[0]]
    
    for i in range(1, len(all_volumes), batch_size):
        batch = all_volumes[i:min(i+batch_size, len(all_volumes))]
        if batch:
            try:
                result, _ = gmsh.model.occ.fuse(merged[-1:], batch)
                merged = result
                print(f"  已合并 {min(i+batch_size, len(all_volumes))}/{len(all_volumes)} 个几何体...")
            except Exception as e:
                print(f"  合并警告: {e}")
                # 继续处理
                pass
    result = merged

# 移除重复的体
gmsh.model.occ.removeAllDuplicates()

# 同步几何模型
print("同步几何模型...")
gmsh.model.occ.synchronize()

# 获取所有体
volumes = gmsh.model.getEntities(3)
if volumes:
    # 添加物理组
    volume_tags = [tag for dim, tag in volumes]
    gmsh.model.addPhysicalGroup(3, volume_tags, tag=1, name="CubeFrameMatrix")
    print(f"添加物理组，包含 {len(volume_tags)} 个体")

# 设置网格选项以避免周期性表面问题
print("设置网格参数...")
gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT算法，更稳定
gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * 2)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

# 设置所有点的网格尺寸
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

# 尝试生成网格
print("生成3D网格...")
try:
    # 先生成1D网格
    gmsh.model.mesh.generate(1)
    print("  1D网格生成完成")
    
    # 再生成2D网格
    gmsh.model.mesh.generate(2)
    print("  2D网格生成完成")
    
    # 最后生成3D网格
    gmsh.model.mesh.generate(3)
    print("  3D网格生成完成")
    
    # 优化网格
    gmsh.model.mesh.optimize("Netgen")
    
except Exception as e:
    print(f"网格生成出错: {e}")
    print("尝试替代方案...")
    
    # 替代方案：重新设置网格参数
    gmsh.option.setNumber("Mesh.Algorithm", 1)  # MeshAdapt
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 3)
    
    try:
        gmsh.model.mesh.generate(3)
        print("使用替代算法生成网格成功")
    except Exception as e2:
        print(f"替代方案也失败: {e2}")
        print("将只保存几何体，不生成网格")

end_time = time.time()
print(f"生成完成！用时: {end_time - start_time:.2f} 秒")

# 获取网格信息
try:
    nodes = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements()
    if len(nodes[0]) > 0:
        print(f"网格节点数: {len(nodes[0])}")
        print(f"网格单元数: {sum(len(e) for e in elements[1])}")
except:
    print("无法获取网格信息")

# 保存文件
print("保存文件...")
try:
    # 保存几何文件（始终可以保存）
    gmsh.write(f"cube_frame_matrix_{nx}x{ny}x{nz}.brep")
    print(f"  几何文件已保存: cube_frame_matrix_{nx}x{ny}x{nz}.brep")
    
    # 尝试保存网格文件
    gmsh.write(f"cube_frame_matrix_{nx}x{ny}x{nz}.msh")
    print(f"  网格文件已保存: cube_frame_matrix_{nx}x{ny}x{nz}.msh")
    
    # 对于小模型，也保存STL
    if nx * ny * nz <= 27:
        gmsh.write(f"cube_frame_matrix_{nx}x{ny}x{nz}.stl")
        print(f"  STL文件已保存: cube_frame_matrix_{nx}x{ny}x{nz}.stl")
except Exception as e:
    print(f"保存文件时出错: {e}")

# 可视化（对于小型模型）
if nx <= 3 and ny <= 3 and nz <= 3:
    print("启动可视化...")
    try:
        gmsh.fltk.run()
    except:
        print("可视化失败，但文件已保存")
else:
    print("模型较大，跳过可视化。可以用Gmsh GUI打开保存的文件查看")

# 清理
gmsh.finalize()
print("完成！")