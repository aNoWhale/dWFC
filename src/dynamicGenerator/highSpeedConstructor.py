"""
this file is used to translate TO result to structure, that is why called constructor
Optimized version with detailed progress tracking
"""

import os
import sys
import tempfile
import numpy as np
import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import gc
import shutil
import time
from datetime import datetime, timedelta

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

class ProgressTracker:
    """进度跟踪器，用于估计剩余时间"""
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = time.time()
        self.task_times = []
        
    def update(self, count=1):
        """更新完成的任务数"""
        self.completed_tasks += count
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if self.completed_tasks > 0:
            avg_time_per_task = elapsed / self.completed_tasks
            remaining_tasks = self.total_tasks - self.completed_tasks
            estimated_remaining = avg_time_per_task * remaining_tasks
            
            # 更新最近任务的时间记录（用于更准确的估计）
            if len(self.task_times) < 10:  # 只保留最近10个任务的时间
                self.task_times.append(elapsed if not self.task_times else elapsed - (self.start_time + sum(self.task_times)))
            else:
                self.task_times.pop(0)
                self.task_times.append(elapsed - (self.start_time + sum(self.task_times[:-1])))
            
            # 使用最近任务的加权平均时间
            if self.task_times:
                recent_avg = sum(self.task_times) / len(self.task_times)
                weighted_remaining = recent_avg * remaining_tasks
                # 结合总体平均和近期平均
                estimated_remaining = 0.7 * weighted_remaining + 0.3 * estimated_remaining
            
            return elapsed, estimated_remaining
        return elapsed, float('inf')
    
    def get_progress_info(self):
        """获取进度信息"""
        elapsed, remaining = self.update(0)  # 不增加计数，只计算
        progress_pct = (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 0
        
        info = {
            'completed': self.completed_tasks,
            'total': self.total_tasks,
            'percentage': progress_pct,
            'elapsed': timedelta(seconds=int(elapsed)),
            'remaining': timedelta(seconds=int(remaining)) if remaining != float('inf') else "估算中...",
            'eta': datetime.now() + timedelta(seconds=int(remaining)) if remaining != float('inf') else "未知"
        }
        return info

def build_single_shape(args):
    """构建单个形状"""
    try:
        cell_points, constructor, type_id, task_id = args
        start_time = time.time()
        shape = constructor(cell_points.tolist())
        shape.Checked(True)
        processing_time = time.time() - start_time
        return shape, task_id, processing_time
    except Exception as e:
        print(f"构建形状时出错 (类型 {type_id}, 任务 {task_id}): {e}")
        return None, task_id, 0

def export_cell_structures_optimized_with_progress(mesh: Mesh, rho: np.ndarray, tileHandle: TileHandler, 
                                                 output_filename, sum_threshold=0.4, max_workers=None, 
                                                 batch_size=100, use_threading=True):
    """
    带详细进度跟踪的优化版本
    
    Args:
        use_threading: 使用线程池而非进程池，避免序列化问题
        batch_size: 每批处理的单元数量
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
    task_id = 0
    for i, type_id in valid_indices:
        if type_id not in constructors:
            continue
        cell_points = points[cells[i]]
        tasks.append((cell_points, constructors[type_id], type_id, task_id))
        task_id += 1

    print(f"准备处理 {len(tasks)} 个形状")
    
    # 初始化进度跟踪器
    progress_tracker = ProgressTracker(len(tasks))

    # 分批次处理以避免内存问题
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    temp_files = []
    total_shapes_processed = 0

    try:
        # 创建主进度条
        with tqdm.tqdm(total=len(tasks), desc="总进度", unit="shape", 
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as main_pbar:
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(tasks))
                batch_tasks = tasks[start_idx:end_idx]
                
                batch_desc = f"批次 {batch_idx + 1}/{total_batches}"
                print(f"\n处理{batch_desc}, 数量: {len(batch_tasks)}")
                
                # 处理当前批次
                batch_shapes = process_batch_with_progress(batch_tasks, use_threading, max_workers, batch_desc)
                
                if batch_shapes:
                    # 将当前批次写入临时STEP文件
                    temp_file = write_batch_to_step_with_progress(batch_shapes, f"batch_{batch_idx}", batch_desc)
                    if temp_file:
                        temp_files.append(temp_file)
                        total_shapes_processed += len(batch_shapes)
                
                # 更新主进度条
                main_pbar.update(len(batch_tasks))
                
                # 显示详细进度信息
                progress_info = progress_tracker.get_progress_info()
                main_pbar.set_postfix({
                    '完成度': f"{progress_info['percentage']:.1f}%",
                    '已用时': str(progress_info['elapsed']),
                    '剩余时间': str(progress_info['remaining']),
                    '预计完成': str(progress_info['eta'])[11:19] if isinstance(progress_info['eta'], datetime) else "未知"
                })
                
                # 清理内存
                del batch_shapes
                gc.collect()
            
            # 合并临时文件，保持几何体独立性
            print(f"\n开始合并 {len(temp_files)} 个临时文件...")
            merge_step_files_with_progress(temp_files, output_filename, total_shapes_processed)
            
            # 最终统计
            progress_info = progress_tracker.get_progress_info()
            print(f"\n✅ 处理完成!")
            print(f"   总处理单元: {progress_info['completed']}/{progress_info['total']}")
            print(f"   总用时: {progress_info['elapsed']}")
            print(f"   输出文件: {output_filename}")
        
    except Exception as e:
        print(f"\n❌ 处理过程中出错: {e}")
        raise
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def process_batch_with_progress(tasks, use_threading=True, max_workers=None, batch_desc=""):
    """处理单个批次的任务，带进度条"""
    if not tasks:
        return []
    
    if max_workers is None:
        max_workers = min(len(tasks), os.cpu_count() or 4)
    
    shapes = []
    completed_tasks = 0
    
    # 批次进度条
    with tqdm.tqdm(total=len(tasks), desc=f"{batch_desc} - 构建形状", 
                  unit="shape", leave=False, 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        if use_threading:
            # 使用线程池
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_task = {executor.submit(build_single_shape, task): task for task in tasks}
                
                # 处理完成的任务
                for future in as_completed(future_to_task):
                    try:
                        result, task_id, processing_time = future.result(timeout=300)  # 5分钟超时
                        if result is not None:
                            shapes.append(result)
                        completed_tasks += 1
                        pbar.update(1)
                        pbar.set_postfix({'最近耗时': f'{processing_time:.2f}s'})
                    except Exception as e:
                        print(f"\n任务处理出错: {e}")
                        completed_tasks += 1
                        pbar.update(1)
        else:
            # 使用进程池
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(build_single_shape, task): task for task in tasks}
                
                for future in as_completed(future_to_task):
                    try:
                        result, task_id, processing_time = future.result(timeout=300)
                        if result is not None:
                            shapes.append(result)
                        completed_tasks += 1
                        pbar.update(1)
                        pbar.set_postfix({'最近耗时': f'{processing_time:.2f}s'})
                    except Exception as e:
                        print(f"\n任务处理出错: {e}")
                        completed_tasks += 1
                        pbar.update(1)
    
    print(f"{batch_desc} - 完成: {len(shapes)}/{len(tasks)} 个形状")
    return shapes

def write_batch_to_step_with_progress(shapes, batch_name, batch_desc):
    """将批次中的形状作为独立实体写入STEP文件，带进度条"""
    if not shapes:
        return None
        
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.stp', prefix=f'batch_{batch_name}_')
    temp_filename = temp_file.name
    temp_file.close()
    
    try:
        step_writer = STEPControl_Writer()
        Interface_Static.SetCVal("write.step.schema", "AP203")
        Interface_Static.SetIVal("write.step.verbose", 0)
        Interface_Static.SetIVal("write.step.product.mode", 1)
        
        # 添加写入进度条
        success_count = 0
        with tqdm.tqdm(total=len(shapes), desc=f"{batch_desc} - 写入文件", 
                      unit="shape", leave=False) as pbar:
            
            for i, shape in enumerate(shapes):
                try:
                    if not shape.IsNull():
                        shape.Checked(True)
                        transfer_status = step_writer.Transfer(shape, STEPControl_AsIs)
                        if transfer_status:
                            success_count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"传输形状 {i} 时出错: {e}")
                    pbar.update(1)
                    continue
        
        if success_count > 0:
            status = step_writer.Write(temp_filename)
            if status == IFSelect_RetDone:
                print(f"{batch_desc} - 成功写入 {success_count} 个形状到临时文件")
                return temp_filename
            else:
                print(f"{batch_desc} - STEP文件写入失败，状态: {status}")
        else:
            print(f"{batch_desc} - 没有成功传输的形状")
            
    except Exception as e:
        print(f"{batch_desc} - 写入时出错: {e}")
    
    # 清理临时文件
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return None

def merge_step_files_with_progress(temp_files, output_filename, total_shapes):
    """合并多个STEP文件，带进度条"""
    if not temp_files:
        print("没有临时文件可合并")
        return
    
    if len(temp_files) == 1:
        # 直接复制单个文件
        shutil.copy2(temp_files[0], output_filename)
        print(f"直接复制临时文件到 {output_filename}")
        return
    
    try:
        # 创建最终的STEP写入器
        final_writer = STEPControl_Writer()
        Interface_Static.SetCVal("write.step.schema", "AP203")
        Interface_Static.SetIVal("write.step.verbose", 0)
        Interface_Static.SetIVal("write.step.product.mode", 1)
        
        total_loaded = 0
        
        # 合并进度条
        with tqdm.tqdm(total=len(temp_files), desc="合并临时文件", unit="file") as pbar:
            for temp_file in temp_files:
                if not os.path.exists(temp_file):
                    pbar.update(1)
                    continue
                    
                try:
                    # 读取临时文件中的形状
                    reader = STEPControl_Reader()
                    status = reader.ReadFile(temp_file)
                    
                    if status == IFSelect_RetDone:
                        reader.TransferRoots()
                        nb_shapes = reader.NbShapes()
                        
                        shapes_loaded = 0
                        for i in range(1, nb_shapes + 1):
                            shape = reader.Shape(i)
                            if not shape.IsNull():
                                shape.Checked(True)
                                final_writer.Transfer(shape, STEPControl_AsIs)
                                shapes_loaded += 1
                                total_loaded += 1
                        
                        pbar.set_postfix({'本文件形状': shapes_loaded, '累计形状': total_loaded})
                    
                except Exception as e:
                    print(f"读取临时文件 {temp_file} 时出错: {e}")
                finally:
                    # 删除临时文件
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                
                pbar.update(1)
        
        if total_loaded > 0:
            print(f"开始写入最终文件，共 {total_loaded} 个形状...")
            status = final_writer.Write(output_filename)
            if status == IFSelect_RetDone:
                print(f"✅ 成功合并 {len(temp_files)} 个临时文件，共 {total_loaded}/{total_shapes} 个形状")
            else:
                raise RuntimeError(f"最终STEP文件写入失败: {status}")
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
            print(f"目录 '{directory_path}' 创建成功")
        else:
            print(f"目录 '{directory_path}' 已存在")
    except Exception as e:
        print(f"创建目录 '{directory_path}' 时出错: {e}")

# 性能测试函数
def benchmark_export(mesh, rho, tileHandle, output_filename, methods=['optimized'], batch_sizes=[50, 100, 200]):
    """性能基准测试"""
    results = {}
    
    for method in methods:
        for batch_size in batch_sizes:
            print(f"\n{'='*50}")
            print(f"测试方法: {method}, 批次大小: {batch_size}")
            print(f"{'='*50}")
            
            test_filename = output_filename.replace('.stp', f'_{method}_batch{batch_size}.stp')
            
            start_time = time.time()
            try:
                if method == 'optimized':
                    export_cell_structures_optimized_with_progress(
                        mesh, rho, tileHandle, test_filename, 
                        batch_size=batch_size, max_workers=os.cpu_count()
                    )
                # 可以添加其他方法的测试
                
                end_time = time.time()
                duration = end_time - start_time
                results[f"{method}_batch{batch_size}"] = duration
                print(f"✅ 完成测试，耗时: {duration:.2f}秒")
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                results[f"{method}_batch{batch_size}"] = None
    
    # 输出性能比较
    print(f"\n{'='*50}")
    print("性能测试结果:")
    print(f"{'='*50}")
    for config, duration in results.items():
        if duration is not None:
            print(f"{config}: {duration:.2f}秒")
        else:
            print(f"{config}: 失败")
    
    return results

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
    
    # 使用带进度条的优化版本
    print("开始导出结构...")
    start_time = time.time()
    
    export_cell_structures_optimized_with_progress(
        mesh, toConstruct, tileHandler,
        "/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/vtk形状更好的++TT0TT180/wfc_classical_end_with_progress.stp",
        max_workers=min(8, os.cpu_count()),  # 根据CPU核心数调整
        batch_size=200,  # 每批处理200个单元
        use_threading=True  # 使用线程池避免序列化问题
    )
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\n🎉 导出完成! 总用时: {total_duration:.2f}秒 ({timedelta(seconds=int(total_duration))})")
    
    # 可选：运行性能基准测试
    # print("\n运行性能基准测试...")
    # benchmark_results = benchmark_export(
    #     mesh, toConstruct, tileHandler,
    #     "/mnt/c/Users/Administrator/Desktop/metaDesign/一些好结果/vtk形状更好的++TT0TT180/benchmark"
    # )