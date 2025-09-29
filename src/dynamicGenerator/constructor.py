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

def export_cell_structures(mesh:Mesh, rho:np.ndarray ,tileHandle:TileHandler, output_filename):
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
    cell_type_ids:np.ndarray=np.argmax(rho,axis=-1,keepdims=False).tolist()

    # 创建STEP写入器
    step_writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP203")
    # 遍历所有单元
    # for i, (cell_indices, type_id) in enumerate(zip(cells, cell_type_ids)):
    for i, (cell_indices, type_id) in enumerate(tqdm.tqdm(zip(cells, cell_type_ids), total=len(cells), desc="constructing",mininterval=1.0,)):
        # 获取单元类型名称
        type_name = tileHandle.typeList[type_id]
        
        # 获取单元顶点坐标
        cell_points:np.ndarray = points[cell_indices]
        
        # 获取构造方法并创建几何结构
        constructor = tileHandle.typeMethod[type_name].build
        shape = constructor(cell_points.tolist())
        
        # 将结构添加到STEP文件（作为独立实体）
        step_writer.Transfer(shape, STEPControl_AsIs)
    
    # 写入文件
    status = step_writer.Write(output_filename)
    
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP文件写入失败，错误代码: {status}")
    else:
        print(f"成功导出 {len(cells)} 个结构到 {output_filename}")

if __name__ == "__main__":
    from src.dynamicGenerator.TileImplement.Cube import BCC,FCC
    tileHandler = TileHandler(typeList=['BCC','FCC'], direction=(('back',"front"),("left","right"),("top","bottom")))
    tileHandler.register(['BCC','FCC'],[BCC,FCC])
    from jax_fem.generate_mesh import get_meshio_cell_type,box_mesh_gmsh
    import meshio
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 10, 10, 10
    if not os.path.exists("data/msh/box.msh"):
        meshio_mesh = box_mesh_gmsh(Nx=Nx,Ny=Ny,Nz=Nz,domain_x=Lx,domain_y=Ly,domain_z=Lz,data_dir="data",ele_type=ele_type)
    else:
        meshio_mesh = meshio.read("data/msh/box.msh")
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    rho=np.random.randn(10,10,10,2).reshape(-1,2)
    export_cell_structures(mesh,rho,tileHandler,"test.stp")