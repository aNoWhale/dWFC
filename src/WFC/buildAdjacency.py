import gmsh
from collections import defaultdict
import jax.numpy as jnp
import meshio

def build_hex8_adjacency(model):
    """
    为非结构化六面体(HEX8)网格构建邻接关系
    """
    # 1. 获取所有3D元素
    element_types, element_tags, node_tags = model.mesh.getElements(3)  # 3D元素

    # 查找HEX8元素类型
    hex8_index = None
    for i, elem_type in enumerate(element_types):
        if elem_type == 16:  # HEX8元素类型
            hex8_index = i
            break

    if hex8_index is None:
        # 尝试通过名称查找
        for i, elem_type in enumerate(element_types):
            try:
                elem_name = gmsh.model.mesh.getElementTypeName(elem_type)
            except AttributeError:
                try:
                    elem_name = gmsh.model.mesh.getElementName(elem_type)
                except:
                    elem_name = ""
            if "hexa" in elem_name.lower() or "hex8" in elem_name.lower():
                hex8_index = i
                break

    if hex8_index is None:
        raise ValueError("No HEX8 elements found in the model")

    hex8_elements = element_tags[hex8_index]
    hex8_node_tags = node_tags[hex8_index]

    num_elements = len(hex8_elements)

    # 2. 创建面到单元的映射
    face_to_cells = defaultdict(set)

    # 定义六面体的面节点顺序
    hex8_faces = [
        [0, 1, 2, 3],  # 底面
        [4, 5, 6, 7],  # 顶面
        [0, 1, 5, 4],  # 前面
        [1, 2, 6, 5],  # 右面
        [2, 3, 7, 6],  # 后面
        [3, 0, 4, 7],  # 左面,
    ]

    # 3. 遍历所有六面体单元
    for idx, elem_tag in enumerate(hex8_elements):
        # 获取单元的节点标签
        start_idx = idx * 8  # 每个HEX8有8个节点
        elem_nodes = hex8_node_tags[start_idx : start_idx + 8]

        # 处理每个面
        for face_nodes_idx in hex8_faces:
            # 获取面的节点标签
            face_nodes = [elem_nodes[i] for i in face_nodes_idx]

            # 对节点标签排序以创建一致的键
            face_key = tuple(sorted(face_nodes))

            # 添加面到单元的映射
            face_to_cells[face_key].add(elem_tag)

    # 4. 构建邻接关系
    adj_dict = defaultdict(set)  # 每个单元的邻居集合

    # 通过共享面连接单元
    for face, cells in face_to_cells.items():
        if len(cells) > 1:  # 只考虑内部面(共享面)
            cell_list = list(cells)
            for i in range(len(cell_list)):
                for j in range(i + 1, len(cell_list)):
                    adj_dict[cell_list[i]].add(cell_list[j])
                    adj_dict[cell_list[j]].add(cell_list[i])

    # 5. 创建标签到索引的映射
    tag_to_idx = {tag: idx for idx, tag in enumerate(hex8_elements)}

    # 6. 准备CSR数据结构
    row_ptr = [0]
    col_idx = []

    # 填充邻接矩阵
    for tag in hex8_elements:
        neighbors = list(adj_dict.get(tag, set()))
        col_idx.extend([tag_to_idx[n] for n in neighbors if n in tag_to_idx])
        row_ptr.append(row_ptr[-1] + len(neighbors))

    # 7. 转换为JAX数组
    return {
        "row_ptr": jnp.array(row_ptr, dtype=jnp.int32),
        "col_idx": jnp.array(col_idx, dtype=jnp.int32),
        "data": jnp.ones(len(col_idx)),  # 所有边权重为1
        "num_elements": num_elements,
    }


def build_adjacency_gmsh(mesh_file):
    """
    从Gmsh网格文件构建邻接关系
    """
    gmsh.initialize()
    gmsh.open(mesh_file)

    # 确保模型拓扑正确
    try:
        gmsh.model.mesh.createTopology()
    except:
        pass

    # 构建邻接关系
    adj_info = build_hex8_adjacency(gmsh.model)
    gmsh.finalize()

    return adj_info


def build_hex8_adjacency_with_meshio(mesh_file):
    """
    使用meshio读取.msh文件并构建HEX8网格的邻接关系
    """
    # 读取网格文件
    mesh = meshio.read(mesh_file)

    # 提取HEX8单元
    hex_cells = None
    for cell_block in mesh.cells:
        if cell_block.type == "hexahedron":
            hex_cells = cell_block.data
            break

    if hex_cells is None:
        raise ValueError("No HEX8 elements found in the mesh")

    num_elements = hex_cells.shape[0]

    # 1. 创建面到单元的映射
    face_to_cells = defaultdict(set)

    # 定义六面体的面节点顺序
    hex8_faces = [
        [0, 1, 2, 3],  # 底面
        [4, 5, 6, 7],  # 顶面
        [0, 1, 5, 4],  # 前面
        [1, 2, 6, 5],  # 右面
        [2, 3, 7, 6],  # 后面
        [3, 0, 4, 7],  # 左面
    ]

    # 2. 遍历所有六面体单元
    for elem_idx, elem_nodes in enumerate(hex_cells):
        # 处理每个面
        for face_nodes_idx in hex8_faces:
            # 获取面的节点标签
            face_nodes = elem_nodes[face_nodes_idx]

            # 对节点标签排序以创建一致的键
            face_key = tuple(sorted(face_nodes))

            # 添加面到单元的映射
            face_to_cells[face_key].add(elem_idx)

    # 3. 构建邻接关系
    adj_dict = defaultdict(set)  # 每个单元的邻居集合

    # 通过共享面连接单元
    for face, cells in face_to_cells.items():
        if len(cells) > 1:  # 只考虑内部面(共享面)
            cell_list = list(cells)
            for i in range(len(cell_list)):
                for j in range(i + 1, len(cell_list)):
                    adj_dict[cell_list[i]].add(cell_list[j])
                    adj_dict[cell_list[j]].add(cell_list[i])

    # 4. 准备CSR数据结构
    row_ptr = [0]
    col_idx = []

    # 填充邻接矩阵
    for i in range(num_elements):
        neighbors = list(adj_dict.get(i, set()))
        col_idx.extend(neighbors)
        row_ptr.append(row_ptr[-1] + len(neighbors))

    # 5. 转换为JAX数组
    return {
        "row_ptr": jnp.array(row_ptr, dtype=jnp.int32),
        "col_idx": jnp.array(col_idx, dtype=jnp.int32),
        "data": jnp.ones(len(col_idx)),  # 所有边权重为1
        "num_elements": num_elements,
    }
