"""
this file is used to translate TO result to structure, that is why called constructor

"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设当前文件在 src/dy 目录下）
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.WFC.TileHandler import TileHandler
from src.dynamicGenerator.meshTools import generate_mesh_from_stp
from jax_fem.generate_mesh import Mesh

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone

import numpy as np
import tqdm
import pathlib
from pathlib import Path

def export_cell_structures(mesh:Mesh, rho:np.ndarray ,tileHandle:TileHandler, output_filename,sum_threshold=0.4):
    """
    导出每个单元为独立结构到STEP文件
    
    参数:
        points: (n, 3) float array, 顶点坐标
        cells: list of lists, 每个单元包含的顶点索引
        cell_type_ids: (n_cells,) int array, 每个单元的类型ID
        tileHandle: 包含类型映射和构造方法的对象
        output_filename: 输出的STEP文件名
    """
    points = mesh.points
    cells = mesh.cells
    rho_sum = np.sum(rho, axis=-1)
    mask = rho_sum > sum_threshold
    cell_type_ids:np.ndarray=np.argmax(rho,axis=-1,keepdims=False).tolist()
    cell_type_ids = np.where(mask, cell_type_ids, -1)
    # 创建STEP写入器
    step_writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP203")
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
    # tileHandler = TileHandler(typeList=['BCC','cubic'], direction=(('back',"front"),("left","right"),("top","bottom")), direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    # tileHandler.register(['BCC','cubic'],[BCC,Cubic])
    pp=STPtile("data/stp/++.stp",(-0.01,0.,-0.01,
                                  0.01,0.02,0.01,
                                  0.,0.01,0.,
                                  0.02,0.02,0.02))
    TTx0=STPtile("data/stp/TTx0.stp",(-0.01,0.,-0.01,
                                      0.01,0.02,0.01,
                                      0.,0.01,0.,
                                      0.02,0.02,0.02))
    TTx180=STPtile("data/stp/TTx180.stp",(-0.01,0.,-0.01,
                                          0.01,0.02,0.01,
                                          0.,0.01,0.,
                                          0.02,0.02,0.02))
    tileHandler = TileHandler(typeList=['pp','TTx0','TTx180'], direction=(('back',"front"),("left","right"),("top","bottom")), direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    tileHandler.register(['pp','TTx0','TTx180'],[pp,TTx0,TTx180])
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
    toConstuct=np.load("/home/sck/metaOptimization/data/vtk++TT0TT180完全约束有filter/npy/wfc_classical_end.npy").reshape(-1,tileHandler.typeNum)

    # import src.WFC.classicalWFC as normalWFC
    # wfc_classical_end ,max_entropy, collapse_list= normalWFC.waveFunctionCollapse(rho_oped,adj,tileHandler)
    # np.save("/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/vtk更清晰++TT0TT180/npy/wfc_classical_end.npy",wfc_classical_end)
    export_cell_structures(mesh,toConstuct,tileHandler,"/home/sck/metaOptimization/data/vtk++TT0TT180完全约束有filter/wfc_classical_end.stp")