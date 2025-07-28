def generate_cube_hex_mesh(
    cube_size=1.0, element_size=0.1, output_file="cube_mesh.msh"
):
    """
    生成具有HEX8六面体单元的结构化立方体网格
    """
    import gmsh
    import os

    # 初始化Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # 启用详细输出
    gmsh.model.add("structured_cube")

    # 创建立方体几何模型
    box = gmsh.model.occ.addBox(0, 0, 0, cube_size, cube_size, cube_size)
    gmsh.model.occ.synchronize()

    # 计算分割数
    n = max(1, int(round(cube_size / element_size)))

    # 对每个维度设置分割数
    gmsh.model.mesh.setTransfiniteAutomatic(recombine=True)

    # 获取所有曲线并设置分割数
    curves = gmsh.model.getEntities(1)
    for curve in curves:
        gmsh.model.mesh.setTransfiniteCurve(curve[1], n + 1)

    # 获取所有表面并设置结构化网格
    surfaces = gmsh.model.getEntities(2)
    for surface in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(surface[1])
        gmsh.model.mesh.setRecombine(2, surface[1])

    # 设置体积结构化网格
    volumes = gmsh.model.getEntities(3)
    for volume in volumes:
        gmsh.model.mesh.setTransfiniteVolume(volume[1])
        gmsh.model.mesh.setRecombine(3, volume[1])

    # 设置网格算法为结构化
    # 关键修复：使用更适合的算法
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Delaunay for quads
    gmsh.option.setNumber(
        "Mesh.Algorithm3D", 1
    )  # Delaunay 3D算法，更适合处理四边形边界

    # 设置重组选项
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # 简单重组
    gmsh.option.setNumber("Mesh.RecombineAll", 1)  # 对所有面重组

    # 生成网格 - 关键修复：先生成2D网格再生成3D网格
    gmsh.model.mesh.generate(2)  # 先生成表面网格
    gmsh.model.mesh.recombine()  # 重组表面网格为四边形
    gmsh.model.mesh.generate(3)  # 再生成体积网格
    gmsh.model.mesh.recombine()  # 重组体积网格为六面体

    # 添加物理组确保所有元素可访问
    surfaces = gmsh.model.getEntities(2)
    surface_tags = [s[1] for s in surfaces]
    gmsh.model.addPhysicalGroup(2, surface_tags, name="Boundary")

    volumes = gmsh.model.getEntities(3)
    volume_tags = [v[1] for v in volumes]
    gmsh.model.addPhysicalGroup(3, volume_tags, name="Volume")

    # 确保生成HEX8元素
    # 获取所有3D元素类型
    element_types, element_tags, _ = gmsh.model.mesh.getElements(3)

    # 检查是否包含HEX8元素（类型16）
    if 16 not in element_types:
        print("警告: 未生成HEX8元素，尝试强制重组...")
        gmsh.model.mesh.recombine()
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.recombine()

    # 保存网格文件
    gmsh.write(output_file)

    # 结束Gmsh
    gmsh.finalize()

    return os.path.abspath(output_file)
