import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.WFC.TileHandler import TileHandler
from src.dynamicGenerator.meshTools import generate_mesh_from_stp
from jax_fem.generate_mesh import Mesh


from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer

import numpy as np
import tqdm
from pathlib import Path


def export_cell_structures_stl(mesh: Mesh, rho: np.ndarray, tileHandle: TileHandler, output_filename, 
                             threshold=0.4, deflection=0.1):
    """导出每个单元为STL格式（使用TopoDS_Compound + BRep_Builder实现）"""
    points = mesh.points
    cells = mesh.cells
    # rho_sum = np.sum(rho, axis=-1)
    mask = np.max(rho,axis=-1) > threshold
    # mask = rho_sum > sum_threshold
    cell_type_ids = np.argmax(rho, axis=-1, keepdims=False)
    cell_type_ids = np.where(mask, cell_type_ids, -1)

    # 按要求创建复合形状（核心修改部分）
    total_compound = TopoDS_Compound()
    total_builder = BRep_Builder()
    total_builder.MakeCompound(total_compound)

    # 遍历所有单元，添加到复合形状
    for i, (cell_indices, type_id) in enumerate(tqdm.tqdm(zip(cells, cell_type_ids), 
                                                          total=len(cells), desc="Constructing STL")):
        if type_id != -1:
            type_name = tileHandle.typeList[type_id]
            cell_points = points[cell_indices]
            # 生成单元几何形状（沿用原build方法）
            constructor = tileHandle.typeMethod[type_name].build
            shape = constructor(cell_points.tolist())
            shape.Checked(True)
            # 添加形状到复合对象
            total_builder.Add(total_compound, shape)

    # 对复合形状进行网格化（一次性处理所有单元，加速关键）
    mesher = BRepMesh_IncrementalMesh(total_compound, deflection)
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("网格生成失败，请调整deflection参数（增大值可降低精度但提高速度）")

    # 写入STL文件
    stl_writer = StlAPI_Writer()
    status = stl_writer.Write(total_compound, output_filename)
    if not status:
        raise RuntimeError(f"STL文件写入失败: {output_filename}")
    print(f"成功导出 {np.sum(mask)} 个单元到STL文件: {output_filename}")


def create_directory_if_not_exists(directory_path):
    try:
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"创建目录失败: {e}")


if __name__ == "__main__":
    from src.dynamicGenerator.TileImplement.CubeSTP import STPtile

    # 初始化tileHandler（与原代码一致）
    # ZCYS = STPtile("data/stp/ZCYS.stp", (-5,-5,-5,5,5,5,0.,0.,0.,10,10,10))
    # ZCYSx0 = STPtile("data/stp/ZCYSx0.stp", (-5,-5,-5,5,5,5,0.,0.,0.,10,10,10))
    # ZCYSx180 = STPtile("data/stp/ZCYSx180.stp", (-5,-5,-5,5,5,5,0.,0.,0.,10,10,10))
    # tileHandler = TileHandler(
    #     typeList=['ZCYS','ZCYSx0','ZCYSx180'],
    #     direction=(('back',"front"),("left","right"),("top","bottom")),
    #     direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5}
    # )
    # tileHandler.register(['ZCYS','ZCYSx0','ZCYSx180'], [ZCYS, ZCYSx0, ZCYSx180])

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

    pp=STPtile("data/stp/++.stp",(-0.01,0.,-0.01,
                                  0.01,0.02,0.01,
                                  0.,0.01,0.,
                                  0.02,0.02,0.02))
    tileHandler = TileHandler(typeList=['pp'], direction=(('back',"front"),("left","right"),("top","bottom")), direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    tileHandler.register(['pp'],[pp])
    # 加载网格和结果（与原代码一致）
    from jax_fem.generate_mesh import get_meshio_cell_type, box_mesh_gmsh
    import meshio
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly, Lz = 40., 5., 20.
    Nx, Ny, Nz = 40, 5, 20
    create_directory_if_not_exists("data/msh")
    mshname = f"L{Lx}{Ly}{Lz}N{Nx}{Ny}{Nz}.msh"
    if not os.path.exists(f"data/msh/{mshname}"):
        meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz, 
                                   data_dir="data", ele_type=ele_type, name=mshname)
    else:
        meshio_mesh = meshio.read(f"data/msh/{mshname}")
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    pathname= 'vtk++multi1.8p5TO'
    toConstruct = np.load(f"/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/{pathname}/npy/wfc_classical_end.npy").reshape(-1, tileHandler.typeNum)
    # 导出为STL（加速效果：比STP快10-20倍）
    export_cell_structures_stl(
        mesh=mesh,
        rho=toConstruct,
        tileHandle=tileHandler,
        output_filename=f"/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/{pathname}/{pathname}.stl",
        threshold=0.4,
        deflection=0.5  # 可调整：0.1（高精度慢）→ 1.0（低精度快）
    )
