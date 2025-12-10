import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.WFC.TileHandler_JAX import TileHandler
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
                             mask=None, deflection=0.1):
    """导出每个单元为STL格式（使用TopoDS_Compound + BRep_Builder实现）"""
    points = mesh.points
    cells = mesh.cells
    mask = np.full(rho.shape[0], True, dtype=bool) if mask is None else mask
    # mask = np.sum(rho, axis=-1) > threshold
    cell_type_ids = np.argmax(rho, axis=-1, keepdims=False)
    cell_type_ids = np.where(mask, cell_type_ids, -1)
    if "void" in tileHandle.typeList:
        void_type_id = tileHandle.typeList.index("void")
        cell_type_ids = np.where(cell_type_ids == void_type_id, -1, cell_type_ids)

    # 按要求创建复合形状（核心修改部分）
    total_compound = TopoDS_Compound()
    total_builder = BRep_Builder()
    total_builder.MakeCompound(total_compound)
    n=0
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
            n+=1
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
    print(f"成功导出 {n} 个单元到STL文件: {output_filename}")


def create_directory_if_not_exists(directory_path):
    try:
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"创建目录失败: {e}")


if __name__ == "__main__":
    from src.dynamicGenerator.TileImplement.CubeSTP import STPtile
    from src.dynamicGenerator.TileImplement.voidCube import Voidtile

    # ZCYS = STPtile("data/stp/ZCYS.stp", (-5,-5,-5,5,5,5,0.,0.,0.,10,10,10))
    # ZCYSx0 = STPtile("data/stp/ZCYSx0.stp", (-5,-5,-5,5,5,5,0.,0.,0.,10,10,10))
    # ZCYSx180 = STPtile("data/stp/ZCYSx180.stp", (-5,-5,-5,5,5,5,0.,0.,0.,10,10,10))
    # tileHandler = TileHandler(
    #     typeList=['ZCYS','ZCYSx0','ZCYSx180'],
    #     direction=(('back',"front"),("left","right"),("top","bottom")),
    #     direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5}
    # )
    # tileHandler.register(['ZCYS','ZCYSx0','ZCYSx180'], [ZCYS, ZCYSx0, ZCYSx180])
    # (x_min, y_min, z_min, x_max, y_max, z_max,
    #  center_x, center_y, center_z,
    #  size_x, size_y, size_z)
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
    # TTz0 = STPtile('data/stp/TTz0.STEP',(-0.01, 0.,-0.01,
    #                                       0.01,0.02,0.01,
    #                                       0.,0.0,0.01,
    #                                       0.02,0.02,0.02))
    # TTz180 = STPtile('data/stp/TTz180.STEP',(-0.01,0.,-0.01,
    #                                       0.01,0.02,0.01,
    #                                       0.,0.,0.01,
    #                                       0.02,0.02,0.02))
    # TTy0=STPtile("data/stp/TTy0.STEP",(-0.01,0.,-0.01,
    #                                   0.01,0.02,0.01,
    #                                   0.,0.01,0.,
    #                                   0.02,0.02,0.02))
    # TTy180=STPtile("data/stp/TTy180.STEP",(-0.01,0.,-0.01,
    #                                   0.01,0.02,0.01,
    #                                   0.,0.01,0.,
    #                                   0.02,0.02,0.02))


    a=STPtile("data/stp/a.STEP",(-0.45,-0.45,-0.45,
                                      0.45,0.45,0.45,
                                      0.,0.,0.,
                                      0.9,0.9,0.9))
    c=STPtile("data/stp/c.STEP",(-0.45,-0.45,-0.45,
                                    0.45,0.45,0.45,
                                    0.,0.,0.,
                                    0.9,0.9,0.9))
    e=STPtile("data/stp/e.STEP",(-0.45,-0.45,-0.45,
                                    0.45,0.45,0.45,
                                    0.,0.,0.,
                                    0.9,0.9,0.9))
    void = Voidtile()
    



    # void = Voidtile()
    # vv = STPtile("data/stp/ZCYS.stp", (-5,-5,-5,5,5,5,0.,0.,0.,10,10,10))
    # tileHandler = TileHandler(typeList=['pp','TTx0','TTx180','vv'], direction=(('back',"front"),("left","right"),("top","bottom")), direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    # tileHandler = TileHandler(typeList=['pp', 'TTx0', 'TTx180','vv'], 
    #                       direction=(('back',"front"),("left","right"),("top","bottom")),
    #                       direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    # tileHandler.selfConnectable(typeName=['pp','TTx0', 'TTx180','vv'],value=1)
    # tileHandler.setConnectiability(fromTypeName='pp',toTypeName=[ 'TTx0','TTx180'],direction="isotropy",value=1,dual=True)
    # tileHandler.setConnectiability(fromTypeName='TTx180',toTypeName=[ 'TTx0',],direction="isotropy",value=1,dual=True)
    # tileHandler.setConnectiability(fromTypeName='pp',toTypeName=[ 'TTx0',],direction="right",value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='pp',toTypeName=[ 'TTx180',],direction="left",value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='TTx180',toTypeName=[ 'TTx180',],direction=["left","right"],value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='TTx0',toTypeName=[ 'TTx0',],direction=["left","right"],value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='vv',toTypeName=[ 'pp','TTx0','TTx180'],direction="isotropy",value=1,dual=True)
    # tileHandler.register(['pp','TTx0','TTx180','vv'],[pp,TTx0,TTx180,vv])
    # tileHandler.constantlize_compatibility()

    # void = Voidtile()
    # tileHandler = TileHandler(typeList=['pp','TTx0','TTx180'], direction=(('back',"front"),("left","right"),("top","bottom")), direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    # tileHandler = TileHandler(typeList=['pp', 'TTx0', 'TTx180'], 
    #                       direction=(('back',"front"),("left","right"),("top","bottom")),
    #                       direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    # tileHandler.selfConnectable(typeName=['pp','TTx0', 'TTx180'],value=1)
    # tileHandler.setConnectiability(fromTypeName='pp',toTypeName=[ 'TTx0','TTx180'],direction="isotropy",value=1,dual=True)
    # tileHandler.setConnectiability(fromTypeName='TTx180',toTypeName=[ 'TTx0',],direction="isotropy",value=1,dual=True)
    # tileHandler.setConnectiability(fromTypeName='pp',toTypeName=[ 'TTx0',],direction="right",value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='pp',toTypeName=[ 'TTx180',],direction="left",value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='TTx180',toTypeName=[ 'TTx180',],direction=["left","right"],value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='TTx0',toTypeName=[ 'TTx0',],direction=["left","right"],value=0,dual=True)
    # tileHandler.register(['pp','TTx0','TTx180'],[pp,TTx0,TTx180])

    # tileHandler = TileHandler(typeList=['pp', 'TTy0', 'TTy180'], 
    #                           direction=(('y+',"y-"),("x-","x+"),("z+","z-")),
    #                           direction_map={"z+":0,"x+":1,"z-":2,"x-":3,"y+":4,"y-":5})
    # tileHandler.selfConnectable(typeName=['pp','TTy0', 'TTy180',],value=1)
    # tileHandler.setConnectiability(fromTypeName='pp',toTypeName=[ 'TTy0','TTy180'],direction="isotropy",value=1,dual=True)
    # tileHandler.setConnectiability(fromTypeName='TTy180',toTypeName=[ 'TTy0',],direction="isotropy",value=1,dual=True)
    # tileHandler.setConnectiability(fromTypeName='pp',toTypeName=[ 'TTy0',],direction="y+",value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='pp',toTypeName=[ 'TTy180',],direction="y-",value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='TTy180',toTypeName=[ 'TTy180',],direction=["y+","y-"],value=0,dual=True)
    # tileHandler.setConnectiability(fromTypeName='TTy0',toTypeName=[ 'TTy0',],direction=["y+","y-"],value=0,dual=True)
    # # tileHandler.setConnectiability(fromTypeName='void',toTypeName=[ '++weak','TTz0','TTz180'],direction="isotropy",value=1,dual=True)
    # tileHandler.register(['pp','TTy0','TTy180'],[pp,TTy0,TTy180])


    tileHandler = TileHandler(typeList=['a','c','e','void'],direction=(('y+',"y-"),("x-","x+"),))
    tileHandler.selfConnectable(typeName="e",direction='isotropy',value=1)

    tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c',],direction='y-',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','a'],direction='x-',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName=['e','c','a'],direction='x+',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName='c',direction='y+',value=1,dual=True)



    tileHandler.setConnectiability(fromTypeName='a',toTypeName=['a','e'],direction='y+',value=-1,dual=True)
    tileHandler.setConnectiability(fromTypeName='a',toTypeName=['a'],direction='y-',value=-1,dual=True)



    tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','a'],direction='y+',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','a','c'],direction='x-',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName=['e','a','c'],direction='x+',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName='a',direction='y-',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName=['c'],direction='y+',value=-1,dual=True)
    tileHandler.setConnectiability(fromTypeName='c',toTypeName=['c','e'],direction='y-',value=-1,dual=True)


    tileHandler.setConnectiability(fromTypeName='e',toTypeName='a',direction='y+',value=1,dual=True)
    tileHandler.setConnectiability(fromTypeName='e',toTypeName='c',direction='y-',value=1,dual=True)


    tileHandler.selfConnectable(typeName="void",direction='isotropy',value=1)
    tileHandler.setConnectiability(fromTypeName='void',toTypeName=['a','c','e'],direction=['x+','x-','y+','y-'],value=1,dual=True)

    tileHandler.register(['a','c','e','void'],[a,c,e,void])

    tileHandler.constantlize_compatibility()


    # pp=STPtile("data/stp/++.stp",(-0.01,0.,-0.01,
    #                               0.01,0.02,0.01,
    #                               0.,0.01,0.,
    #                               0.02,0.02,0.02))
    # tileHandler = TileHandler(typeList=['pp'], direction=(('back',"front"),("left","right"),("top","bottom")), direction_map={"top":0,"right":1,"bottom":2,"left":3,"back":4,"front":5})
    # tileHandler.register(['pp'],[pp])
    # 加载网格和结果（与原代码一致）
    from jax_fem.generate_mesh import get_meshio_cell_type, box_mesh_gmsh
    import meshio
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    # Lx, Ly, Lz = 40., 5., 20.
    # Nx, Ny, Nz = 40, 5, 20
    # Lx, Ly, Lz = 20., 5., 10.
    # Nx, Ny, Nz = 20, 5, 10
    Lx, Ly, Lz = 60., 30., 1.
    Nx, Ny, Nz = 60, 30, 1
    create_directory_if_not_exists("data/msh")
    mshname = f"L{Lx}{Ly}{Lz}N{Nx}{Ny}{Nz}.msh"
    if not os.path.exists(f"data/msh/{mshname}"):
        meshio_mesh = box_mesh_gmsh(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz, 
                                   data_dir="data", ele_type=ele_type, name=mshname)
    else:
        meshio_mesh = meshio.read(f"data/msh/{mshname}")
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    prefix="2D/"
    pathname= 'vtkfacevoidnofilter1.2p3333SimpWFCsigma2D后处理过了'
    print(f"{pathname}")
    # rho_oped = np.load(f"/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/{prefix+pathname}/npy/100.npy").reshape(-1, tileHandler.typeNum+1)
    # rho_oped = rho_oped[...,:3].reshape(-1,tileHandler.typeNum)
    # wfcEnd = np.load(f"/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/{prefix+pathname}/npy/wfc_classical_end.npy").reshape(-1, tileHandler.typeNum)
    from jax_fem.utils import extract_theta_from_vtu
    vtkinfo = extract_theta_from_vtu(f'/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/{prefix+pathname}/sol_136.vtu')
    rho_oped=[]
    for i in range(tileHandler.typeNum):
        rho_oped.append(vtkinfo[f'theta{i}'][...,None])

    rho_oped = np.concatenate(rho_oped,axis=-1)
    rho_oped = rho_oped.reshape(-1,tileHandler.typeNum)
    from src.WFC.adjacencyCSR import build_hex8_adjacency_with_meshio
    adj_csr = build_hex8_adjacency_with_meshio(mesh=meshio_mesh)

    from src.WFC.mixWFC import waveFunctionCollapse as classicalWFC
    wfcEnd,_,_=classicalWFC(init_probs=rho_oped,adj_csr=adj_csr,tileHandler=tileHandler)

    print(f"wfc mean:{wfcEnd.mean()}")
    mask=np.max(rho_oped,axis=-1,keepdims=False)>0.5
    # 导出为STL（加速效果：比STP快10-20倍）
    export_cell_structures_stl(
        mesh=mesh,
        rho=wfcEnd,
        tileHandle=tileHandler,
        output_filename=f"/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/{prefix+pathname}/{pathname}.stl",
        mask=mask,
        deflection=1.0  # 可调整：0.1（高精度慢）→ 1.0（低精度快）
    )
