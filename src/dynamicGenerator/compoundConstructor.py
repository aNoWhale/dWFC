"""
this file is used to translate TO result to structure, that is why called constructor
Optimized version that maintains consistent speed by using file merging
"""

import os
import sys
import tempfile
import numpy as np
import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import shutil
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.WFC.TileHandler import TileHandler
from jax_fem.generate_mesh import Mesh

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_Reader
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRep import BRep_Builder

def build_single_shape(args):
    """构建单个形状"""
    try:
        cell_points, constructor, type_id = args
        shape = constructor(cell_points.tolist())
        shape.Checked(True)
        return shape
    except Exception as e:
        print(f"构建形状时出错 (类型 {type_id}): {e}")
        return None

def export_cell_structures_consistent_speed(mesh: Mesh, rho: np.ndarray, tileHandle: TileHandler, 
                                         output_filename, sum_threshold=0.4, max_workers=None, 
                                         batch_size=100, merge_batch_size=10):
    """
    保持恒定速度的版本：通过控制每个STEP文件的形状数量来避免性能下降
    
    Args:
        batch_size: 每批处理的单元数量
        merge_batch_size: 每个临时文件包含的批次数量
    """
    points = mesh.points
    cells = mesh.cells
    rho_sum = np.sum(rho, axis=-1)
    mask = rho_sum > sum_threshold
    cell_type_ids = np.argmax(rho, axis=-1)
    cell_type_ids = np.where(mask, cell_type_ids, -1)

    # 过滤有效单元
    valid_indices = [(i, tid) for i, tid in enumerate(cell_type_ids) if tid != -1]
    print(f"有效单元数量: {len(valid_indices)}")
    
    if not valid_indices:
        print("没有有效的单元需要处理")
        return

    # 预缓存构造函数
    constructors = {}
    for type_id, type_name in enumerate(tileHandle.typeList):
        if type_name in tileHandle.typeMethod:
            constructors[type_id] = tileHandle.typeMethod[type_name].build

    # 准备任务
    tasks = []
    for i, type_id in valid_indices:
        if type_id not in constructors:
            continue
        cell_points = points[cells[i]]
        tasks.append((cell_points, constructors[type_id], type_id))

    print(f"准备处理 {len(tasks)} 个形状")
    
    start_time = time.time()
    
    if max_workers is None:
        max_workers = min(len(tasks), os.cpu_count() or 4)
    
    # 分批次处理，每个临时文件包含有限数量的形状
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    temp_files = []
    shapes_in_current_file = 0
    current_file_shapes = []
    
    try:
        with tqdm.tqdm(total=len(tasks), desc="处理进度", unit="shape") as pbar:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(tasks))
                batch_tasks = tasks[start_idx:end_idx]
                
                # 处理当前批次
                batch_shapes = process_batch_consistent(
                    batch_tasks, max_workers, f"批次 {batch_idx+1}/{total_batches}"
                )
                
                # 将当前批次的形状添加到当前文件
                current_file_shapes.extend(batch_shapes)
                shapes_in_current_file += len(batch_shapes)
                
                # 如果当前文件形状数量达到限制，写入临时文件
                if shapes_in_current_file >= batch_size * merge_batch_size or batch_idx == total_batches - 1:
                    if current_file_shapes:
                        temp_file = write_batch_to_temp_file(
                            current_file_shapes, 
                            f"temp_{len(temp_files)}", 
                            f"临时文件 {len(temp_files)+1}"
                        )
                        if temp_file:
                            temp_files.append(temp_file)
                        # 重置当前文件
                        current_file_shapes = []
                        shapes_in_current_file = 0
                
                # 更新进度
                pbar.update(len(batch_tasks))
                
                # 显示进度信息
                elapsed = time.time() - start_time
                processed = pbar.n
                if processed > 0:
                    avg_time = elapsed / processed
                    remaining = avg_time * (len(tasks) - processed)
                    pbar.set_postfix({
                        '临时文件': len(temp_files),
                        '已用时': f"{elapsed:.1f}s",
                        '剩余时间': f"{remaining:.1f}s",
                        '速度': f"{processed/elapsed:.1f} shape/s"
                    })
                
                # 及时清理内存
                del batch_shapes
                gc.collect()
        
        # 合并临时文件
        print(f"\n合并 {len(temp_files)} 个临时文件...")
        merge_step_files_efficient(temp_files, output_filename)
        
        total_time = time.time() - start_time
        print(f"✅ 处理完成!")
        print(f"   总用时: {total_time:.1f}秒")
        print(f"   平均速度: {len(tasks)/total_time:.1f} 形状/秒")
        print(f"   输出文件: {output_filename}")
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        raise
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def process_batch_consistent(tasks, max_workers, batch_desc):
    """处理单个批次的任务"""
    if not tasks:
        return []
    
    shapes = []
    
    # 批次进度条
    with tqdm.tqdm(total=len(tasks), desc=batch_desc, unit="shape", leave=False) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(build_single_shape, task): i for i, task in enumerate(tasks)}
            
            for future in as_completed(futures):
                try:
                    shape = future.result(timeout=60)
                    if shape is not None:
                        shapes.append(shape)
                except Exception as e:
                    print(f"任务处理出错: {e}")
                finally:
                    pbar.update(1)
    
    return shapes

def write_batch_to_temp_file(shapes, file_id, desc):
    """将一批形状写入临时STEP文件"""
    if not shapes:
        return None
        
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.stp', prefix=f'{file_id}_')
    temp_filename = temp_file.name
    temp_file.close()
    
    try:
        step_writer = STEPControl_Writer()
        Interface_Static.SetCVal("write.step.schema", "AP203")
        Interface_Static.SetIVal("write.step.verbose", 0)
        Interface_Static.SetIVal("write.step.product.mode", 1)
        
        success_count = 0
        with tqdm.tqdm(total=len(shapes), desc=desc, unit="shape", leave=False) as pbar:
            for shape in shapes:
                try:
                    if not shape.IsNull():
                        step_writer.Transfer(shape, STEPControl_AsIs)
                        success_count += 1
                except Exception as e:
                    print(f"传输形状时出错: {e}")
                finally:
                    pbar.update(1)
        
        if success_count > 0:
            status = step_writer.Write(temp_filename)
            if status == IFSelect_RetDone:
                return temp_filename
            else:
                print(f"临时文件写入失败: {status}")
        else:
            print(f"没有成功传输的形状")
            
    except Exception as e:
        print(f"写入临时文件时出错: {e}")
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return None

def merge_step_files_efficient(temp_files, output_filename):
    """高效合并多个STEP文件"""
    if not temp_files:
        return
    
    if len(temp_files) == 1:
        shutil.copy2(temp_files[0], output_filename)
        return
    
    try:
        # 使用更高效的合并策略：逐个文件添加，但控制每个写入器的形状数量
        final_writer = STEPControl_Writer()
        Interface_Static.SetCVal("write.step.schema", "AP203")
        Interface_Static.SetIVal("write.step.verbose", 0)
        Interface_Static.SetIVal("write.step.product.mode", 1)
        
        total_loaded = 0
        
        with tqdm.tqdm(total=len(temp_files), desc="合并文件", unit="file") as pbar:
            for temp_file in temp_files:
                if not os.path.exists(temp_file):
                    pbar.update(1)
                    continue
                    
                try:
                    reader = STEPControl_Reader()
                    status = reader.ReadFile(temp_file)
                    
                    if status == IFSelect_RetDone:
                        reader.TransferRoots()
                        nb_shapes = reader.NbShapes()
                        
                        for i in range(1, nb_shapes + 1):
                            shape = reader.Shape(i)
                            if not shape.IsNull():
                                final_writer.Transfer(shape, STEPControl_AsIs)
                                total_loaded += 1
                        
                except Exception as e:
                    print(f"读取临时文件出错: {e}")
                finally:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                
                pbar.update(1)
                pbar.set_postfix({'已加载形状': total_loaded})
        
        if total_loaded > 0:
            status = final_writer.Write(output_filename)
            if status != IFSelect_RetDone:
                raise RuntimeError(f"最终文件写入失败: {status}")
        else:
            raise RuntimeError("没有成功合并任何形状")
            
    except Exception as e:
        print(f"合并文件时出错: {e}")
        raise

def create_directory_if_not_exists(directory_path):
    """如果目录不存在则创建目录"""
    try:
        path = Path(directory_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"创建目录出错: {e}")

if __name__ == "__main__":
    from src.dynamicGenerator.TileImplement.CubeSTP import STPtile
    from jax_fem.generate_mesh import get_meshio_cell_type, box_mesh_gmsh
    import meshio
    
    # 初始化tile处理器
    pp = STPtile("data/stp/++.stp", (-0.01, 0., -0.01, 0.01, 0.02, 0.01, 0., 0.01, 0., 0.02, 0.02, 0.02))
    TTx0 = STPtile("data/stp/TTx0.stp", (-0.01, 0., -0.01, 0.01, 0.02, 0.01, 0., 0.01, 0., 0.02, 0.02, 0.02))
    TTx180 = STPtile("data/stp/TTx180.stp", (-0.01, 0., -0.01, 0.01, 0.02, 0.01, 0., 0.01, 0., 0.02, 0.02, 0.02))
    
    tileHandler = TileHandler(
        typeList=['pp', 'TTx0', 'TTx180'], 
        direction=(('back', "front"), ("left", "right"), ("top", "bottom")), 
        direction_map={"top": 0, "right": 1, "bottom": 2, "left": 3, "back": 4, "front": 5}
    )
    tileHandler.register(['pp', 'TTx0', 'TTx180'], [pp, TTx0, TTx180])
    
    # 生成或加载网格
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
    
    # 加载数据
    toConstruct = np.load("/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/vtk形状更好的++TT0TT180/npy/wfc_classical_end.npy").reshape(-1, tileHandler.typeNum)
    
    # 使用保持恒定速度的版本
    export_cell_structures_consistent_speed(
        mesh, toConstruct, tileHandler,
        "/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/vtk形状更好的++TT0TT180/wfc_classical_end_consistent.stp",
        max_workers=min(8, os.cpu_count()),
        batch_size=50,  # 较小的批次大小
        merge_batch_size=5  # 每个临时文件包含5个批次（约250个形状）
    )