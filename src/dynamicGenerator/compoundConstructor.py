"""
this file is used to translate TO result to structure, that is why called constructor

"""
import os
import sys
import tempfile
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/dy 目录下）
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.WFC.TileHandler import TileHandler
from src.dynamicGenerator.meshTools import generate_mesh_from_stp
from jax_fem.generate_mesh import Mesh

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs,STEPControl_Reader
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Builder
from OCC.Core.BRep import BRep_Builder

import numpy as np
import tqdm
import pathlib
from pathlib import Path

# def export_cell_structures(mesh:Mesh, rho:np.ndarray ,tileHandle:TileHandler, output_filename,sum_threshold=0.4,batch_size=100):
#     """
#     导出每个单元为独立结构到STEP文件
    
#     参数:
#         points: (n, 3) float array, 顶点坐标
#         cells: list of lists, 每个单元包含的顶点索引
#         cell_type_ids: (n_cells,) int array, 每个单元的类型ID
#         tileHandle: 包含类型映射和构造方法的对象
#         output_filename: 输出的STEP文件名
#     """
#     points = mesh.points
#     cells = mesh.cells
#     rho_sum = np.sum(rho, axis=-1)
#     mask = rho_sum > sum_threshold
#     cell_type_ids:np.ndarray=np.argmax(rho,axis=-1,keepdims=False).tolist()
#     cell_type_ids = np.where(mask, cell_type_ids, -1)
#     n_total_cells = len(cells)
#     # 按batch_size分块迭代
#     i=0
#     with tempfile.TemporaryDirectory(prefix='constructor_temp_') as tmpdir:
#         print("临时目录路径：", tmpdir)  # 例如：/tmp/my_temp_12345
#         # 在临时目录中创建文件
#         filenames=[]
#         for batch_idx in tqdm.tqdm(range(0, n_total_cells, batch_size),desc="blocks"):
#             # if i>3:
#             #     break
#             # 计算当前批次的起止索引
#             batch_start = batch_idx
#             batch_end = min(batch_idx + batch_size, n_total_cells)
            
#             # 1. 分块单元列表
#             cells_batch = cells[batch_start:batch_end]
            
#             # 2. 分块单元类型ID（保持原有数组结构）
#             cell_type_ids_batch = cell_type_ids[batch_start:batch_end]

#             # for i, (cell_block, type_block) in enumerate(zip(cells_batch, cell_type_ids_batch)):
#             fn=f"{tmpdir}/{i}.stp"
#             write2stp(cells_batch,cell_type_ids_batch,points,tileHandle,output_filename=fn)
#             filenames.append(fn)
#             i+=1
#         # step_writer = STEPControl_Writer()
#         # Interface_Static.SetCVal("write.step.schema", "AP203")
#         # Interface_Static.SetIVal("write.step.verbose", 0)
#         compound = TopoDS_Compound()
#         # 2. 初始化构建器
#         builder = TopoDS_Builder()
#         # 3. 初始化复合容器（关键：先"打开"容器准备添加子形状）
#         builder.MakeCompound(compound)
#         # builder=BRep_Builder()
#         # total_compound = builder.MakeCompound()
#         for file in filenames:
#             reader = STEPControl_Reader()
#             # 读取单个STP文件
#             if reader.ReadFile(file) != IFSelect_RetDone:
#                 print(f"跳过无效文件: {file}")
#                 continue
#             # 转换文件的顶层根结构（不分解内部实体）
#             # 通常每个STP文件只有1个顶层根结构（包含所有实体）
#             if reader.TransferRoot(1):  # 只取第1个顶层根结构（绝大多数STP文件只有1个）
#                 top_shape = reader.Shape()
#                 if not top_shape.IsNull():
#                     # 将当前文件的顶层结构添加到总复合结构中
#                     builder.Add(compound,top_shape)
#                     print(f"已添加文件整体结构: {file}")

        
#         # 生成最终的总复合结构
#         # merged_shape = compound.Shape()
#         if compound.IsNull():
#             print("没有有效结构可合并")
#             return
#         # 写入合并后的STP文件
#         writer = STEPControl_Writer()
#         writer.SetTolerance(1e-5)
#         # 将总复合结构写入新文件（保持8个原文件的顶层结构层级）
#         writer.Transfer(compound, STEPControl_AsIs)
        
#         if writer.Write(output_filename) == IFSelect_RetDone:
#             print(f"合并完成: {output_filename}（包含{len(filenames)}个原文件的整体结构）")
#         else:
#             print("合并失败")


def export_cell_structures(mesh: Mesh, rho: np.ndarray, tileHandle: TileHandler, output_filename, sum_threshold=0.4, batch_size=100):
    points = mesh.points
    cells = mesh.cells
    rho_sum = np.sum(rho, axis=-1)
    mask = rho_sum > sum_threshold
    cell_type_ids: np.ndarray = np.argmax(rho, axis=-1, keepdims=False).tolist()
    cell_type_ids = np.where(mask, cell_type_ids, -1)
    n_total_cells = len(cells)
    i = 0

    with tempfile.TemporaryDirectory(prefix='constructor_temp_') as tmpdir:
        print("临时目录路径：", tmpdir)
        filenames = []

        # 1. 分批次生成临时STP文件（不变）
        for batch_idx in tqdm.tqdm(range(0, n_total_cells, batch_size), desc="blocks"):
            batch_start = batch_idx
            batch_end = min(batch_idx + batch_size, n_total_cells)
            cells_batch = cells[batch_start:batch_end]
            cell_type_ids_batch = cell_type_ids[batch_start:batch_end]

            fn = f"{tmpdir}/{i}.stp"
            write2stp(cells_batch, cell_type_ids_batch, points, tileHandle, output_filename=fn)
            filenames.append(fn)
            i += 1


        total_compound = TopoDS_Compound()
        total_builder = BRep_Builder()
        total_builder.MakeCompound(total_compound)

        for file in filenames:
            reader = STEPControl_Reader()
            if reader.ReadFile(file) != IFSelect_RetDone:
                print(f"跳过无效文件: {file}")
                continue

            batch_compound = TopoDS_Compound()
            # 关键修改2：批次复合结构也用BRep_Builder
            batch_builder = BRep_Builder()
            batch_builder.MakeCompound(batch_compound)

            # 读取当前批次所有根实体（不变，确保不丢失）
            num_roots = reader.NbRootsForTransfer()
            for root_idx in range(1, num_roots + 1):
                if reader.TransferRoot(root_idx):
                    shape = reader.Shape()
                    if not shape.IsNull():
                        # 关键：直接添加原始shape，保留其TopLoc_Location位置信息
                        batch_builder.Add(batch_compound, shape)

            if not batch_compound.IsNull():
                total_builder.Add(total_compound, batch_compound)
                print(f"已添加批次结构: {file}（包含{num_roots}个实体，位置保留）")

        # 3. 导出最终文件（不变，确保schema和精度）
        if total_compound.IsNull():
            print("没有有效结构可合并")
            return

        writer = STEPControl_Writer()
        Interface_Static.SetCVal("write.step.schema", "AP214IS")
        writer.SetTolerance(1e-5)
        # 关键：保持AsIs模式，不修改任何形状和位置
        writer.Transfer(total_compound, STEPControl_AsIs)

        if writer.Write(output_filename) == IFSelect_RetDone:
            print(f"合并完成: {output_filename}")
            print(f"最终根数量 = {len(filenames)}，零件位置与临时文件一致")
        else:
            print("合并失败")




def write2stp(cells,cell_type_ids,points,tileHandle:TileHandler,output_filename):
    # 创建STEP写入器
    step_writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP214IS")
    Interface_Static.SetIVal("write.step.verbose", 0)
    # 遍历所有单元
    # for i, (cell_indices, type_id) in enumerate(zip(cells, cell_type_ids)):
    for i, (cell_indices, type_id) in enumerate(tqdm.tqdm(zip(cells, cell_type_ids), total=len(cells), desc="constructing",mininterval=1.0,)):
        if type_id != -1:
            # 获取单元类型名称
            type_name = tileHandle.typeList[type_id]
            
            # 获取单元顶点坐标
            cell_points:np.ndarray = points[cell_indices]
            
            # 获取构造方法并创建几何结构
            constructor = tileHandle.typeMethod[type_name].build
            shape = constructor(cell_points.tolist())
            shape.Checked(True)
            # 将结构添加到STEP文件（作为独立实体）
            step_writer.Transfer(shape, STEPControl_AsIs)
    
    # 写入文件
    status = step_writer.Write(output_filename)
    
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP文件写入失败，错误代码: {status}")
    else:
        print(f"成功导出 {len(cells)} 个结构到 {output_filename}")



def create_directory_if_not_exists(directory_path):
    """
    如果目录不存在则创建目录
    
    Args:
        directory_path: 目录路径（可以是字符串或Path对象）
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"目录 '{directory_path}' 创建成功")
        else:
            print(f"目录 '{directory_path}' 已存在")
    except Exception as e:
        print(f"创建目录 '{directory_path}' 时出错: {e}")



if __name__ == "__main__":
    from src.dynamicGenerator.TileImplement.CubeSTP import STPtile
    # (x_min, y_min, z_min, x_max, y_max, z_max,
    #         center_x, center_y, center_z,
    #         size_x, size_y, size_z)
    # Cubic=STPtile("data/stp/cubic.stp",(-0.01,-0.01,-0.01,0.01,0.01,0.01,0.,0.,0.,0.02,0.02,0.02))
    # BCC=STPtile("data/stp/BCC.stp",(-0.02,-0.02,-0.02,0.02,0.02,0.02,0.,0.,0.,0.04,0.04,0.04))
    # pp=STPtile("data/stp/++.stp",(-0.01,0.,-0.01,
    #                               0.01,0.02,0.01,
    #                               0.,0.01,0.,
    #                               0.02,0.02,0.02))
    # tileHandler = TileHandler(typeList=['BCC','cubic','pp'], direction=(('back',"front"),("left","right"),("top","bottom")), direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    # tileHandler.register(['BCC','cubic','pp'],[BCC,Cubic,pp])
    
    ZCYS=STPtile("data/stp/ZCYS.stp",(-5,-5,-5,
                                  5,5,5,
                                  0.,0.,0.,
                                  10,10,10))
    ZCYSx0=STPtile("data/stp/ZCYSx0.stp",(-5,-5,-5,
                                  5,5,5,
                                  0.,0.,0.,
                                  10,10,10))
    ZCYSx180=STPtile("data/stp/ZCYSx180.stp",(-5,-5,-5,
                                  5,5,5,
                                  0.,0.,0.,
                                  10,10,10))
    tileHandler = TileHandler(typeList=['ZCYS','ZCYSx0','ZCYSx180'], direction=(('back',"front"),("left","right"),("top","bottom")), direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    tileHandler.register(['ZCYS','ZCYSx0','ZCYSx180'],[ZCYS,ZCYSx0,ZCYSx180])
    
    
    
    # pp=STPtile("data/stp/++.stp",(-0.01,0.,-0.01,
    #                               0.01,0.02,0.01,
    #                               0.,0.01,0.,
    #                               0.02,0.02,0.02))
    # TTx0=STPtile("data/stp/TTx0.stp",(-0.01,0.,-0.01,
    #                                   0.01,0.02,0.01,
    #                                   0.,0.01,0.,
    #                                   0.02,0.02,0.02))
    # TTx180=STPtile("data/stp/TTx180.stp",(-0.01,0.,-0.01,
    #                                       0.01,0.02,0.01,
    #                                       0.,0.01,0.,
    #                                       0.02,0.02,0.02))
    # tileHandler = TileHandler(typeList=['pp','TTx0','TTx180'], direction=(('back',"front"),("left","right"),("top","bottom")), direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    # tileHandler.register(['pp','TTx0','TTx180'],[pp,TTx0,TTx180])
    from jax_fem.generate_mesh import get_meshio_cell_type,box_mesh_gmsh
    import meshio
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    # Lx, Ly, Lz = 60., 10., 30.
    # Nx, Ny, Nz = 60, 10, 30
    # Lx, Ly, Lz = 10., 2., 5.
    # Nx, Ny, Nz = 10, 2, 5
    Lx, Ly, Lz = 40., 5., 20.
    Nx, Ny, Nz = 40, 5, 20
    create_directory_if_not_exists("data/msh")
    mshname=f"L{Lx}{Ly}{Lz}N{Nx}{Ny}{Nz}.msh"
    if not os.path.exists(f"data/msh/{mshname}"):
        meshio_mesh = box_mesh_gmsh(Nx=Nx,Ny=Ny,Nz=Nz,domain_x=Lx,domain_y=Ly,domain_z=Lz,data_dir="data",ele_type=ele_type,name=mshname)
    else:
        meshio_mesh = meshio.read(f"data/msh/{mshname}")
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    # rho=np.load("data/npy/wfc_end.npy").reshape(-1,tileHandler.typeNum)
    toConstuct=np.load("/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/vtkZCYSZCYSx0ZCYSx180common1.8p544/npy/wfc_classical_end.npy").reshape(-1,tileHandler.typeNum)

    # import src.WFC.classicalWFC as normalWFC
    # wfc_classical_end ,max_entropy, collapse_list= normalWFC.waveFunctionCollapse(rho_oped,adj,tileHandler)
    # np.save("/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/vtk更清晰++TT0TT180/npy/wfc_classical_end.npy",wfc_classical_end)
    export_cell_structures(mesh,toConstuct,tileHandler,"/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/vtkZCYSZCYSx0ZCYSx180common1.8p544/wfc_classical_end.stp")