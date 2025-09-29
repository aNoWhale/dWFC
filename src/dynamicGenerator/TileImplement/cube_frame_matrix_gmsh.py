import gmsh
import numpy as np
import time

# 初始化Gmsh
gmsh.initialize()
gmsh.model.add("cube_frame_matrix")

# 参数设置
nx, ny, nz = 3, 3, 3  # 立方体数量（可以改为100, 100, 100，但会非常慢）
cube_size = 1.0        # 单个立方体边长
cylinder_radius = 0.05  # 圆杆半径
sphere_radius = 0.08    # 球体半径
mesh_size = 0.5         # 网格尺寸（越大越粗糙但越快）

# 优化开关
use_optimization = True  # 是否使用优化（避免重复创建几何体）
show_progress = True     # 是否显示进度

print(f"开始生成 {nx}x{ny}x{nz} 立方体框架矩阵...")
print(f"将创建 {(nx+1)*(ny+1)*(nz+1)} 个球体（顶点）")
print(f"将创建 {nx*(ny+1)*(nz+1) + ny*(nx+1)*(nz+1) + nz*(nx+1)*(ny+1)} 根圆杆（边）")

start_time = time.time()

# 存储已创建的几何体，避免重复创建
created_spheres = {}  # 键：(i,j,k)坐标，值：几何体标签
created_cylinders = {}  # 键：((i1,j1,k1), (i2,j2,k2))坐标对，值：几何体标签

sphere_tags = []
cylinder_tags = []

# 创建所有顶点的球体
print("创建球体...")
total_spheres = (nx+1) * (ny+1) * (nz+1)
sphere_count = 0

for i in range(nx + 1):
    for j in range(ny + 1):
        for k in range(nz + 1):
            x = i * cube_size
            y = j * cube_size
            z = k * cube_size
            
            # 创建球体
            sphere_tag = gmsh.model.occ.addSphere(x, y, z, sphere_radius)
            sphere_tags.append((3, sphere_tag))
            created_spheres[(i, j, k)] = sphere_tag
            
            sphere_count += 1
            if show_progress and sphere_count % 100 == 0:
                print(f"  已创建 {sphere_count}/{total_spheres} 个球体...")

print(f"球体创建完成，共 {len(sphere_tags)} 个")

# 创建所有的圆杆（边）
print("创建圆杆...")
cylinder_count = 0

# X方向的边（平行于X轴）
for i in range(nx):
    for j in range(ny + 1):
        for k in range(nz + 1):
            start = (i * cube_size, j * cube_size, k * cube_size)
            end = ((i + 1) * cube_size, j * cube_size, k * cube_size)
            
            # 创建圆柱体
            cylinder_tag = gmsh.model.occ.addCylinder(
                start[0], start[1], start[2],
                cube_size, 0, 0,  # X方向
                cylinder_radius
            )
            cylinder_tags.append((3, cylinder_tag))
            
            cylinder_count += 1
            if show_progress and cylinder_count % 100 == 0:
                print(f"  已创建 {cylinder_count} 根圆杆...")

# Y方向的边（平行于Y轴）
for i in range(nx + 1):
    for j in range(ny):
        for k in range(nz + 1):
            start = (i * cube_size, j * cube_size, k * cube_size)
            end = (i * cube_size, (j + 1) * cube_size, k * cube_size)
            
            # 创建圆柱体
            cylinder_tag = gmsh.model.occ.addCylinder(
                start[0], start[1], start[2],
                0, cube_size, 0,  # Y方向
                cylinder_radius
            )
            cylinder_tags.append((3, cylinder_tag))
            
            cylinder_count += 1
            if show_progress and cylinder_count % 100 == 0:
                print(f"  已创建 {cylinder_count} 根圆杆...")

# Z方向的边（平行于Z轴）
for i in range(nx + 1):
    for j in range(ny + 1):
        for k in range(nz):
            start = (i * cube_size, j * cube_size, k * cube_size)
            end = (i * cube_size, j * cube_size, (k + 1) * cube_size)
            
            # 创建圆柱体
            cylinder_tag = gmsh.model.occ.addCylinder(
                start[0], start[1], start[2],
                0, 0, cube_size,  # Z方向
                cylinder_radius
            )
            cylinder_tags.append((3, cylinder_tag))
            
            cylinder_count += 1
            if show_progress and cylinder_count % 100 == 0:
                print(f"  已创建 {cylinder_count} 根圆杆...")

print(f"圆杆创建完成，共 {len(cylinder_tags)} 根")

# 合并所有几何体
print("合并几何体（布尔运算）...")
all_volumes = sphere_tags + cylinder_tags

# 分批合并以提高效率
batch_size = 100
if len(all_volumes) > batch_size:
    print(f"  分批合并，每批 {batch_size} 个...")
    merged = [all_volumes[0]]
    
    for i in range(1, len(all_volumes), batch_size):
        batch = all_volumes[i:min(i+batch_size, len(all_volumes))]
        if batch:
            result, _ = gmsh.model.occ.fuse(merged[-1:], batch)
            merged = result
            if show_progress:
                print(f"  已合并 {min(i+batch_size, len(all_volumes))}/{len(all_volumes)} 个几何体...")
    
    final_result = merged
else:
    final_result, _ = gmsh.model.occ.fuse(all_volumes[:1], all_volumes[1:])

print("几何体合并完成")

# 同步几何模型
print("同步几何模型...")
gmsh.model.occ.synchronize()

# 添加物理组
gmsh.model.addPhysicalGroup(3, [tag for dim, tag in final_result], name="CubeFrameMatrix")

# 设置网格尺寸
print("设置网格参数...")
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

# 设置网格算法（使用更快的算法）
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay算法
gmsh.option.setNumber("Mesh.Optimize", 0)     # 关闭优化以加快速度

# 生成3D网格
print("生成3D网格...")
gmsh.model.mesh.generate(3)

end_time = time.time()
print(f"生成完成！用时: {end_time - start_time:.2f} 秒")

# 获取网格信息
nodes = gmsh.model.mesh.getNodes()
elements = gmsh.model.mesh.getElements()
print(f"网格节点数: {len(nodes[0])}")
print(f"网格单元数: {sum(len(e) for e in elements[1])}")

# 保存网格文件
print("保存文件...")
gmsh.write(f"cube_frame_matrix_{nx}x{ny}x{nz}.msh")
gmsh.write(f"cube_frame_matrix_{nx}x{ny}x{nz}.stl")
print(f"文件已保存: cube_frame_matrix_{nx}x{ny}x{nz}.msh/stl")

# 可视化（对于大型模型可能会很慢）
if nx <= 5 and ny <= 5 and nz <= 5:
    print("启动可视化...")
    gmsh.fltk.run()
else:
    print("模型太大，跳过可视化。可以用Gmsh GUI打开.msh文件查看")

# 清理
gmsh.finalize()