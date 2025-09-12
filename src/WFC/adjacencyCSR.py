import gmsh
from collections import defaultdict
import jax.numpy as jnp
import meshio

# Hexahedron:             Hexahedron20:          Hexahedron27:

#        v
# 3----------2            3----13----2           3----13----2
# |\     ^   |\           |\         |\          |\         |\
# | \    |   | \          | 15       | 14        |15    24  | 14
# |  \   |   |  \         9  \       11 \        9  \ 20    11 \
# |   7------+---6        |   7----19+---6       |   7----19+---6
# |   |  +-- |-- | -> u   |   |      |   |       |22 |  26  | 23|
# 0---+---\--1   |        0---+-8----1   |       0---+-8----1   |
#  \  |    \  \  |         \  17      \  18       \ 17    25 \  18
#   \ |     \  \ |         10 |        12|        10 |  21    12|
#    \|      w  \|           \|         \|          \|         \|
#     4----------5            4----16----5           4----16----5

# w-front u-right v-top
# def build_hex8_adjacency_with_meshio(mesh_file):
#     """
#     使用meshio读取.msh文件并构建HEX8网格的邻接关系
#     """
#     # 读取网格文件
#     mesh = meshio.read(mesh_file)

#     # 提取HEX8单元
#     hex_cells = None
#     for cell_block in mesh.cells:
#         if cell_block.type == "hexahedron":
#             hex_cells = cell_block.data
#             break

#     if hex_cells is None:
#         raise ValueError("No HEX8 elements found in the mesh")

#     num_elements = hex_cells.shape[0]

#     # 1. 创建面到单元的映射
#     face_to_cells = defaultdict(set)

#     # 定义六面体的面节点顺序
#     hex8_faces = [
#         [0, 1, 2, 3],  # 底面
#         [4, 5, 6, 7],  # 顶面
#         [0, 1, 5, 4],  # 前面
#         [1, 2, 6, 5],  # 右面
#         [2, 3, 7, 6],  # 后面
#         [3, 0, 4, 7],  # 左面
#     ]

#     # 2. 遍历所有六面体单元
#     for elem_idx, elem_nodes in enumerate(hex_cells):
#         # 处理每个面
#         for face_nodes_idx in hex8_faces:
#             # 获取面的节点标签
#             face_nodes = elem_nodes[face_nodes_idx]

#             # 对节点标签排序以创建一致的键
#             face_key = tuple(sorted(face_nodes))

#             # 添加面到单元的映射
#             face_to_cells[face_key].add(elem_idx)

#     # 3. 构建邻接关系
#     adj_dict = defaultdict(set)  # 每个单元的邻居集合

#     # 通过共享面连接单元
#     for face, cells in face_to_cells.items():
#         if len(cells) > 1:  # 只考虑内部面(共享面)
#             cell_list = list(cells)
#             for i in range(len(cell_list)):
#                 for j in range(i + 1, len(cell_list)):
#                     adj_dict[cell_list[i]].add(cell_list[j])
#                     adj_dict[cell_list[j]].add(cell_list[i])

#     # 4. 准备CSR数据结构
#     row_ptr = [0]
#     col_idx = []

#     # 填充邻接矩阵
#     for i in range(num_elements):
#         neighbors = list(adj_dict.get(i, set()))
#         col_idx.extend(neighbors)
#         row_ptr.append(row_ptr[-1] + len(neighbors))

#     # 5. 转换为JAX数组
#     return {
#         "row_ptr": jnp.array(row_ptr, dtype=jnp.int32),
#         "col_idx": jnp.array(col_idx, dtype=jnp.int32),
#         "data": jnp.ones(len(col_idx)),  # 所有边权重为1
#         "num_elements": num_elements,
#     }


def build_hex8_adjacency_with_meshio(mesh:meshio.Mesh=None,mesh_file=None):
    if mesh_file is None and mesh is None:
        raise ValueError("need mesh")
    # 读取网格文件
    if mesh_file is not None:
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

    # 定义六面体的面节点顺序和对应的方向标签
    # 格式: [面节点索引], "方向标签"
    hex8_faces = [
        ([0, 1, 2, 3], "back"),
        ([4, 5, 6, 7], "front"),
        ([0, 1, 5, 4], "bottom"),
        ([1, 2, 6, 5], "right"),
        ([2, 3, 7, 6], "top"),
        ([3, 0, 4, 7], "left"),
    ]

    # 修改1: 记录面方向信息
    face_info = defaultdict(list)  # {排序后面键: [(单元索引, 方向)]}
    
    # 2. 遍历所有六面体单元
    for elem_idx, elem_nodes in enumerate(hex_cells):
        # 处理每个面
        for face_nodes_idx, direction in hex8_faces:
            face_nodes = elem_nodes[face_nodes_idx]
            face_key = tuple(sorted(face_nodes))
            # 记录单元和该面对应的方向
            face_info[face_key].append((elem_idx, direction))

    # 修改2: 创建包含方向信息的邻接字典
    adj_dict = defaultdict(dict)  # 单元索引: {邻居: 方向}
    
    # 通过共享面连接单元并记录方向
    for face, info_list in face_info.items():
        if len(info_list) > 1:  # 只考虑内部面
            # 遍历所有可能的单元对
            for i in range(len(info_list)):
                elem_i, direction_i = info_list[i]
                for j in range(i + 1, len(info_list)):
                    elem_j, direction_j = info_list[j]
                    # 记录单元i的方向和邻居j
                    adj_dict[elem_i][elem_j] = direction_i
                    # 记录单元j的方向和邻居i
                    adj_dict[elem_j][elem_i] = direction_j

    # 3. 准备CSR数据结构
    row_ptr = [0]
    col_idx = []
    directions = []  # 新增: 存储方向信息
    
    # 按单元索引顺序填充
    for i in range(num_elements):
        neighbors_data = adj_dict.get(i, {})
        # 按邻居索引排序以保持一致性
        sorted_neighbors = sorted(neighbors_data.items(), key=lambda x: x[0])
        
        for neighbor_idx, direction in sorted_neighbors:
            col_idx.append(neighbor_idx)
            directions.append(direction)  # 存储方向
        
        row_ptr.append(row_ptr[-1] + len(sorted_neighbors))

    # 4. 转换为JAX数组
    return {
        "row_ptr": jnp.array(row_ptr, dtype=jnp.int32),
        "col_idx": jnp.array(col_idx, dtype=jnp.int32),
        "data": jnp.ones(len(col_idx)),  # 所有边权重为1
        "directions": directions,        # 新增方向信息 (保持为Python列表)
        "num_elements": num_elements,
    }

if __name__ == "__main__":
    result = build_hex8_adjacency_with_meshio("/home/sck/metaOptimization/src/WFC/test.msh")

    # 获取第5个单元的邻居信息
    start = result["row_ptr"][5]
    end = result["row_ptr"][6]
    neighbors = result["col_idx"][start:end]
    neighbor_dirs = result["directions"][start:end]

    print(f"单元5的邻居:")
    for n, d in zip(neighbors, neighbor_dirs):
        print(f"  单元{n}在{d}方向")