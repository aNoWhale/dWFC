import gmsh
import numpy as np
import time
import gc
import sys

def create_cube_frame_ultra_fast(nx, ny, nz, 
                                 cube_size=1.0,
                                 cylinder_radius=0.05,
                                 sphere_radius=0.08,
                                 output_file=None):
    """
    超快速版本：不进行布尔运算，直接输出原始几何体
    适用于大规模矩阵（如100x100x100）
    """
    
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("cube_frame_fast")
    
    # 优化设置
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Geometry.OCCParallel", 1)
    gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
    
    print(f"超快速生成 {nx}×{ny}×{nz} 立方体框架矩阵")
    print("注意：此模式不进行布尔运算，几何体会有重叠")
    
    start_time = time.time()
    
    # 计算总数
    total_spheres = (nx+1) * (ny+1) * (nz+1)
    total_cylinders = nx*(ny+1)*(nz+1) + ny*(nx+1)*(nz+1) + nz*(nx+1)*(ny+1)
    
    print(f"预计创建 {total_spheres} 个球体，{total_cylinders} 根圆杆")
    
    # 批量创建，不保存标签以节省内存
    count = 0
    
    # 创建球体
    print("\n创建球体...")
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                x = i * cube_size
                y = j * cube_size
                z = k * cube_size
                gmsh.model.occ.addSphere(x, y, z, sphere_radius)
                
                count += 1
                if count % 10000 == 0:
                    print(f"  已创建 {count}/{total_spheres} 个球体...")
                    gc.collect()  # 定期清理内存
    
    print(f"✓ 球体创建完成: {count} 个")
    
    # 创建圆杆
    cylinder_count = 0
    
    # X方向
    print("\n创建X方向圆杆...")
    for i in range(nx):
        for j in range(ny + 1):
            for k in range(nz + 1):
                gmsh.model.occ.addCylinder(
                    i * cube_size, j * cube_size, k * cube_size,
                    cube_size, 0, 0, cylinder_radius
                )
                cylinder_count += 1
                if cylinder_count % 10000 == 0:
                    print(f"  已创建 {cylinder_count}/{total_cylinders} 根...")
                    gc.collect()
    
    # Y方向
    print("创建Y方向圆杆...")
    for i in range(nx + 1):
        for j in range(ny):
            for k in range(nz + 1):
                gmsh.model.occ.addCylinder(
                    i * cube_size, j * cube_size, k * cube_size,
                    0, cube_size, 0, cylinder_radius
                )
                cylinder_count += 1
                if cylinder_count % 10000 == 0:
                    print(f"  已创建 {cylinder_count}/{total_cylinders} 根...")
                    gc.collect()
    
    # Z方向
    print("创建Z方向圆杆...")
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz):
                gmsh.model.occ.addCylinder(
                    i * cube_size, j * cube_size, k * cube_size,
                    0, 0, cube_size, cylinder_radius
                )
                cylinder_count += 1
                if cylinder_count % 10000 == 0:
                    print(f"  已创建 {cylinder_count}/{total_cylinders} 根...")
                    gc.collect()
    
    print(f"✓ 圆杆创建完成: {cylinder_count} 根")
    
    # 同步
    print("\n同步几何模型...")
    gmsh.model.occ.synchronize()
    
    # 保存
    if output_file is None:
        output_file = f"cube_frame_{nx}x{ny}x{nz}_fast.brep"
    
    print(f"保存到 {output_file}...")
    gmsh.write(output_file)
    
    end_time = time.time()
    print(f"\n✓ 完成！用时: {end_time - start_time:.2f} 秒")
    
    gmsh.finalize()
    return output_file


def create_cube_frame_script_mode(nx, ny, nz, output_file="cube_frame.geo"):
    """
    生成Gmsh脚本文件，而不是直接创建几何体
    这种方式最快，且内存占用最小
    """
    
    print(f"生成Gmsh脚本文件 {output_file}")
    
    cube_size = 1.0
    cylinder_radius = 0.05
    sphere_radius = 0.08
    
    with open(output_file, 'w') as f:
        f.write("// Gmsh script for cube frame matrix\n")
        f.write(f"// Size: {nx}x{ny}x{nz}\n\n")
        
        # 设置参数
        f.write("SetFactory(\"OpenCASCADE\");\n")
        f.write(f"cube_size = {cube_size};\n")
        f.write(f"cylinder_radius = {cylinder_radius};\n")
        f.write(f"sphere_radius = {sphere_radius};\n\n")
        
        # 球体
        print(f"写入球体定义...")
        sphere_id = 1
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    x = i * cube_size
                    y = j * cube_size
                    z = k * cube_size
                    f.write(f"Sphere({sphere_id}) = {{{x}, {y}, {z}, sphere_radius}};\n")
                    sphere_id += 1
        
        # 圆柱体
        print(f"写入圆柱体定义...")
        cylinder_id = sphere_id
        
        # X方向
        for i in range(nx):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    x = i * cube_size
                    y = j * cube_size
                    z = k * cube_size
                    f.write(f"Cylinder({cylinder_id}) = {{{x}, {y}, {z}, ")
                    f.write(f"{x+cube_size}, {y}, {z}, cylinder_radius}};\n")
                    cylinder_id += 1
        
        # Y方向
        for i in range(nx + 1):
            for j in range(ny):
                for k in range(nz + 1):
                    x = i * cube_size
                    y = j * cube_size
                    z = k * cube_size
                    f.write(f"Cylinder({cylinder_id}) = {{{x}, {y}, {z}, ")
                    f.write(f"{x}, {y+cube_size}, {z}, cylinder_radius}};\n")
                    cylinder_id += 1
        
        # Z方向
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz):
                    x = i * cube_size
                    y = j * cube_size
                    z = k * cube_size
                    f.write(f"Cylinder({cylinder_id}) = {{{x}, {y}, {z}, ")
                    f.write(f"{x}, {y}, {z+cube_size}, cylinder_radius}};\n")
                    cylinder_id += 1
        
        # 可选：添加布尔运算（对于小模型）
        if nx * ny * nz <= 8:
            f.write("\n// Boolean union for small models\n")
            f.write(f"BooleanUnion{{ Volume{{1:{sphere_id-1}}}; Delete; }}")
            f.write(f"{{ Volume{{{sphere_id}:{cylinder_id-1}}}; Delete; }}\n")
    
    print(f"✓ 脚本文件已生成: {output_file}")
    print(f"使用方法: gmsh {output_file}")
    return output_file


def create_cube_frame_stl_direct(nx, ny, nz, output_file="cube_frame.stl"):
    """
    直接生成STL文件，跳过Gmsh
    最快的方法，适用于不需要布尔运算的情况
    """
    
    from stl import mesh
    
    print(f"直接生成STL文件（{nx}x{ny}x{nz}）")
    
    cube_size = 1.0
    cylinder_radius = 0.05
    sphere_radius = 0.08
    
    # 创建网格顶点和面的列表
    vertices = []
    faces = []
    
    def add_sphere(cx, cy, cz, radius, resolution=8):
        """添加球体的三角面片"""
        base_idx = len(vertices)
        
        # 生成球体顶点
        for i in range(resolution + 1):
            lat = np.pi * i / resolution
            for j in range(resolution):
                lon = 2 * np.pi * j / resolution
                x = cx + radius * np.sin(lat) * np.cos(lon)
                y = cy + radius * np.sin(lat) * np.sin(lon)
                z = cz + radius * np.cos(lat)
                vertices.append([x, y, z])
        
        # 生成三角面片
        for i in range(resolution):
            for j in range(resolution):
                if i < resolution - 1:
                    idx1 = base_idx + i * resolution + j
                    idx2 = base_idx + i * resolution + (j + 1) % resolution
                    idx3 = base_idx + (i + 1) * resolution + j
                    idx4 = base_idx + (i + 1) * resolution + (j + 1) % resolution
                    
                    faces.append([idx1, idx2, idx3])
                    faces.append([idx2, idx4, idx3])
    
    def add_cylinder(x1, y1, z1, x2, y2, z2, radius, resolution=8):
        """添加圆柱体的三角面片"""
        base_idx = len(vertices)
        
        # 计算圆柱体的方向
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        dx, dy, dz = dx/length, dy/length, dz/length
        
        # 找到两个垂直向量
        if abs(dx) < 0.9:
            perp1 = np.cross([dx, dy, dz], [1, 0, 0])
        else:
            perp1 = np.cross([dx, dy, dz], [0, 1, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross([dx, dy, dz], perp1)
        
        # 生成圆柱体顶点
        for i in range(2):
            t = i * length
            cx = x1 + t * dx
            cy = y1 + t * dy
            cz = z1 + t * dz
            
            for j in range(resolution):
                angle = 2 * np.pi * j / resolution
                offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                vertices.append([cx + offset[0], cy + offset[1], cz + offset[2]])
        
        # 生成侧面三角面片
        for j in range(resolution):
            idx1 = base_idx + j
            idx2 = base_idx + (j + 1) % resolution
            idx3 = base_idx + resolution + j
            idx4 = base_idx + resolution + (j + 1) % resolution
            
            faces.append([idx1, idx3, idx2])
            faces.append([idx2, idx3, idx4])
    
    # 生成所有球体
    print("生成球体...")
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                add_sphere(i * cube_size, j * cube_size, k * cube_size, sphere_radius)
    
    # 生成所有圆柱体
    print("生成圆柱体...")
    # X方向
    for i in range(nx):
        for j in range(ny + 1):
            for k in range(nz + 1):
                add_cylinder(i * cube_size, j * cube_size, k * cube_size,
                           (i + 1) * cube_size, j * cube_size, k * cube_size,
                           cylinder_radius)
    
    # Y方向
    for i in range(nx + 1):
        for j in range(ny):
            for k in range(nz + 1):
                add_cylinder(i * cube_size, j * cube_size, k * cube_size,
                           i * cube_size, (j + 1) * cube_size, k * cube_size,
                           cylinder_radius)
    
    # Z方向
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz):
                add_cylinder(i * cube_size, j * cube_size, k * cube_size,
                           i * cube_size, j * cube_size, (k + 1) * cube_size,
                           cylinder_radius)
    
    # 创建STL网格
    print("创建STL网格...")
    stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[face[j]]
    
    # 保存STL文件
    stl_mesh.save(output_file)
    print(f"✓ STL文件已保存: {output_file}")
    
    return output_file


if __name__ == "__main__":
    print("立方体框架矩阵生成器（优化版）")
    print("=" * 50)
    
    print("\n选择生成方式:")
    print("1. 超快速模式（无布尔运算，适合大规模）")
    print("2. 脚本模式（生成.geo文件，最省内存）")
    print("3. 直接STL模式（需要numpy-stl库）")
    print("4. 测试模式（3x3x3）")
    
    choice = input("\n请选择 (1-4): ")
    
    if choice == '1':
        nx = int(input("X方向数量 (建议≤50): "))
        ny = int(input("Y方向数量 (建议≤50): "))
        nz = int(input("Z方向数量 (建议≤50): "))
        create_cube_frame_ultra_fast(nx, ny, nz)
        
    elif choice == '2':
        nx = int(input("X方向数量: "))
        ny = int(input("Y方向数量: "))
        nz = int(input("Z方向数量: "))
        output_file = f"cube_frame_{nx}x{ny}x{nz}.geo"
        create_cube_frame_script_mode(nx, ny, nz, output_file)
        print(f"\n可以使用以下命令在Gmsh中打开:")
        print(f"  gmsh {output_file}")
        
    elif choice == '3':
        try:
            from stl import mesh
            nx = int(input("X方向数量 (建议≤20): "))
            ny = int(input("Y方向数量 (建议≤20): "))
            nz = int(input("Z方向数量 (建议≤20): "))
            create_cube_frame_stl_direct(nx, ny, nz)
        except ImportError:
            print("需要安装numpy-stl: pip install numpy-stl")
            
    else:
        # 测试模式
        create_cube_frame_ultra_fast(3, 3, 3)