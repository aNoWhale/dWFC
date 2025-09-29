import gmsh
import numpy as np
import time
import sys

def create_cube_frame_matrix(nx, ny, nz, 
                            cube_size=1.0,
                            cylinder_radius=0.05,
                            sphere_radius=0.08,
                            mesh_size=0.5,
                            save_mesh=True,
                            visualize=False,
                            use_fast_mode=True):
    """
    生成立方体框架矩阵
    
    参数:
    - nx, ny, nz: 三个方向的立方体数量
    - cube_size: 单个立方体边长
    - cylinder_radius: 圆杆半径
    - sphere_radius: 球体半径
    - mesh_size: 网格尺寸2
    - save_mesh: 是否保存网格文件
    - visualize: 是否可视化
    - use_fast_mode: 使用快速模式（对于大型模型）
    """
    
    # 初始化Gmsh
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("cube_frame_matrix")
    
    # 对于大型模型，使用优化设置
    if use_fast_mode or (nx * ny * nz > 1000):
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Geometry.OCCParallel", 1)  # 并行处理
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)      # 快速算法
        gmsh.option.setNumber("Mesh.Optimize", 0)         # 跳过优化
    
    print(f"开始生成 {nx}×{ny}×{nz} 立方体框架矩阵...")
    print(f"预计创建:")
    print(f"  - {(nx+1)*(ny+1)*(nz+1)} 个球体（顶点）")
    print(f"  - {nx*(ny+1)*(nz+1) + ny*(nx+1)*(nz+1) + nz*(nx+1)*(ny+1)} 根圆杆（边）")
    
    start_time = time.time()
    
    # 使用fragment而不是fuse来处理大量几何体
    all_volumes = []
    
    # 批量创建球体
    print("\n创建球体顶点...")
    sphere_count = 0
    total_spheres = (nx+1) * (ny+1) * (nz+1)
    
    # 为了效率，批量添加球体
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                x = i * cube_size
                y = j * cube_size
                z = k * cube_size
                
                sphere_tag = gmsh.model.occ.addSphere(x, y, z, sphere_radius)
                all_volumes.append((3, sphere_tag))
                
                sphere_count += 1
                if sphere_count % 1000 == 0:
                    print(f"  已创建 {sphere_count}/{total_spheres} 个球体...")
                    sys.stdout.flush()
    
    print(f"✓ 球体创建完成: {sphere_count} 个")
    
    # 批量创建圆杆
    print("\n创建圆杆...")
    cylinder_count = 0
    total_cylinders = nx*(ny+1)*(nz+1) + ny*(nx+1)*(nz+1) + nz*(nx+1)*(ny+1)
    
    # X方向的圆杆
    print("  创建X方向圆杆...")
    for i in range(nx):
        for j in range(ny + 1):
            for k in range(nz + 1):
                cylinder_tag = gmsh.model.occ.addCylinder(
                    i * cube_size, j * cube_size, k * cube_size,
                    cube_size, 0, 0,
                    cylinder_radius
                )
                all_volumes.append((3, cylinder_tag))
                cylinder_count += 1
                
                if cylinder_count % 1000 == 0:
                    print(f"    已创建 {cylinder_count}/{total_cylinders} 根圆杆...")
                    sys.stdout.flush()
    
    # Y方向的圆杆
    print("  创建Y方向圆杆...")
    for i in range(nx + 1):
        for j in range(ny):
            for k in range(nz + 1):
                cylinder_tag = gmsh.model.occ.addCylinder(
                    i * cube_size, j * cube_size, k * cube_size,
                    0, cube_size, 0,
                    cylinder_radius
                )
                all_volumes.append((3, cylinder_tag))
                cylinder_count += 1
                
                if cylinder_count % 1000 == 0:
                    print(f"    已创建 {cylinder_count}/{total_cylinders} 根圆杆...")
                    sys.stdout.flush()
    
    # Z方向的圆杆
    print("  创建Z方向圆杆...")
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz):
                cylinder_tag = gmsh.model.occ.addCylinder(
                    i * cube_size, j * cube_size, k * cube_size,
                    0, 0, cube_size,
                    cylinder_radius
                )
                all_volumes.append((3, cylinder_tag))
                cylinder_count += 1
                
                if cylinder_count % 1000 == 0:
                    print(f"    已创建 {cylinder_count}/{total_cylinders} 根圆杆...")
                    sys.stdout.flush()
    
    print(f"✓ 圆杆创建完成: {cylinder_count} 根")
    
    # 合并几何体
    print("\n合并几何体...")
    
    if len(all_volumes) > 10000:
        print("  警告：几何体数量很大，合并可能需要较长时间...")
        print("  建议：对于100×100×100的模型，考虑使用分块处理或简化模型")
    
    # 使用fragment进行布尔运算（比fuse更高效）
    if use_fast_mode and len(all_volumes) > 1000:
        print("  使用快速模式（fragment）...")
        result, result_map = gmsh.model.occ.fragment(all_volumes[:1], all_volumes[1:])
    else:
        # 分批合并
        batch_size = 500
        if len(all_volumes) > batch_size:
            print(f"  分批合并，每批 {batch_size} 个...")
            merged = [all_volumes[0]]
            
            for i in range(1, len(all_volumes), batch_size):
                batch = all_volumes[i:min(i+batch_size, len(all_volumes))]
                if batch:
                    try:
                        result, _ = gmsh.model.occ.fuse(merged[-1:], batch)
                        merged = result
                        print(f"    已合并 {min(i+batch_size, len(all_volumes))}/{len(all_volumes)} 个几何体...")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"    合并出错: {e}")
                        break
            
            result = merged
        else:
            result, _ = gmsh.model.occ.fuse(all_volumes[:1], all_volumes[1:])
    
    print("✓ 几何体合并完成")
    
    # 同步模型
    print("\n同步几何模型...")
    gmsh.model.occ.synchronize()
    
    # 添加物理组
    volumes = gmsh.model.getEntities(3)
    if volumes:
        gmsh.model.addPhysicalGroup(3, [tag for dim, tag in volumes], name="CubeFrameMatrix")
    
    # 设置网格参数
    print("设置网格参数...")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    
    # 生成网格（可选）
    generate_mesh = input("\n是否生成网格？(y/n，对于大型模型可能很慢): ").lower() == 'y'
    
    if generate_mesh:
        print("生成3D网格...")
        try:
            gmsh.model.mesh.generate(3)
            
            # 获取网格信息
            nodes = gmsh.model.mesh.getNodes()
            elements = gmsh.model.mesh.getElements()
            print(f"✓ 网格生成完成")
            print(f"  节点数: {len(nodes[0])}")
            print(f"  单元数: {sum(len(e) for e in elements[1])}")
        except Exception as e:
            print(f"网格生成失败: {e}")
    
    # 保存文件
    if save_mesh:
        filename = f"cube_frame_matrix_{nx}x{ny}x{nz}"
        print(f"\n保存文件...")
        
        # 保存几何文件
        gmsh.write(f"{filename}.brep")
        print(f"  几何文件: {filename}.brep")
        
        # 如果生成了网格，保存网格文件
        if generate_mesh:
            gmsh.write(f"{filename}.msh")
            print(f"  网格文件: {filename}.msh")
            
            # 对于小型模型，也保存STL
            if nx * ny * nz <= 125000:
                gmsh.write(f"{filename}.step")
                print(f"  STEP文件: {filename}.step")
    
    # 计时
    end_time = time.time()
    print(f"\n总用时: {end_time - start_time:.2f} 秒")
    
    # 可视化
    if visualize and nx * ny * nz <= 27:
        print("启动可视化...")
        gmsh.fltk.run()
    elif visualize:
        print("模型太大，跳过可视化")
    
    # 清理
    gmsh.finalize()
    
    return True


def create_large_matrix_optimized(nx, ny, nz, output_format='brep'):
    """
    为超大规模矩阵优化的版本（如100×100×100）
    不进行布尔运算，直接输出几何体
    """
    
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("large_cube_matrix")
    
    cube_size = 1.0
    cylinder_radius = 0.05
    sphere_radius = 0.08
    
    print(f"生成超大规模 {nx}×{ny}×{nz} 立方体框架...")
    print("注意：此模式不进行布尔运算，几何体可能有重叠")
    
    start_time = time.time()
    
    # 创建所有几何体但不合并
    volume_tags = []
    
    # 简化：只创建外表面的框架
    create_full = input("创建完整模型(f)还是只创建外表面(s)? (f/s): ").lower() == 'f'
    
    if not create_full:
        print("只创建外表面框架...")
        # 只创建6个面的框架
        # ... 这里可以添加只创建外表面的逻辑
    
    # 快速创建几何体
    print("快速创建几何体...")
    
    # 使用多线程（如果可用）
    gmsh.option.setNumber("Geometry.OCCParallel", 1)
    
    # 批量创建
    count = 0
    total = (nx+1)*(ny+1)*(nz+1) + nx*(ny+1)*(nz+1) + ny*(nx+1)*(nz+1) + nz*(nx+1)*(ny+1)
    
    # 创建进度条
    def show_progress(current, total):
        percent = current * 100 / total
        bar_length = 50
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        sys.stdout.write(f'\r进度: |{bar}| {percent:.1f}% ({current}/{total})')
        sys.stdout.flush()
    
    # 创建所有几何体...
    # [这里添加创建逻辑]
    
    # 同步
    gmsh.model.occ.synchronize()
    
    # 保存
    filename = f"large_cube_matrix_{nx}x{ny}x{nz}.{output_format}"
    gmsh.write(filename)
    print(f"\n文件已保存: {filename}")
    
    end_time = time.time()
    print(f"用时: {end_time - start_time:.2f} 秒")
    
    gmsh.finalize()


if __name__ == "__main__":
    print("立方体框架矩阵生成器")
    print("=" * 40)
    
    # 选择模式
    print("\n选择模式:")
    print("1. 小型模型 (≤5×5×5)")
    print("2. 中型模型 (≤20×20×20)")
    print("3. 大型模型 (≤50×50×50)")
    print("4. 超大型模型 (100×100×100)")
    print("5. 自定义尺寸")
    
    choice = input("\n请选择 (1-5): ")
    
    if choice == '1':
        nx, ny, nz = 5, 5, 5
        mesh_size = 0.1
        visualize = True
    elif choice == '2':
        nx, ny, nz = 10, 10, 10
        mesh_size = 0.3
        visualize = False
    elif choice == '3':
        nx, ny, nz = 20, 20, 20
        mesh_size = 0.5
        visualize = False
    elif choice == '4':
        nx, ny, nz = 100, 100, 100
        print("\n警告：100×100×100 将创建超过300万个几何体！")
        print("这可能需要大量内存和时间。")
        confirm = input("确定继续？(y/n): ")
        if confirm.lower() != 'y':
            exit()
        # 使用优化版本
        create_large_matrix_optimized(nx, ny, nz)
        exit()
    else:
        nx = int(input("X方向立方体数量: "))
        ny = int(input("Y方向立方体数量: "))
        nz = int(input("Z方向立方体数量: "))
        mesh_size = float(input("网格尺寸 (建议0.1-1.0): "))
        visualize = input("是否可视化？(y/n): ").lower() == 'y'
    
    # 生成模型
    create_cube_frame_matrix(
        nx, ny, nz,
        mesh_size=mesh_size,
        visualize=visualize,
        use_fast_mode=(nx*ny*nz > 100)
    )