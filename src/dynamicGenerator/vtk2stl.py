import vtk


def vtu_threshold_to_stl(vtu_path, stl_path, scalar_name, lower_thresh, upper_thresh):
    """
    从VTU文件阈值筛选后导出STL（兼容VTK 7.x/8.x/9.x所有版本）
    :param vtu_path: 输入VTU文件路径
    :param stl_path: 输出STL文件路径
    :param scalar_name: 要筛选的标量数组名（如"Pressure"）
    :param lower_thresh: 阈值下限
    :param upper_thresh: 阈值上限
    """
    # 1. 读取VTU文件
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()

    # 2. 阈值筛选（核心：适配所有VTK版本的阈值逻辑）
    threshold = vtk.vtkThreshold()
    threshold.SetInputConnection(reader.GetOutputPort())
    # 指定筛选的数组（Cell Data，若为Point Data则改为FIELD_ASSOCIATION_POINTS）
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, scalar_name)
    
    # 【终极兼容】设置阈值逻辑：筛选"介于上下限之间"的单元
    # 第一步：设置阈值类型（THRESHOLD_BETWEEN = 介于/THRESHOLD_LOWER = 低于上限/THRESHOLD_UPPER = 高于下限）
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    # 第二步：设置上下限
    threshold.SetLowerThreshold(lower_thresh)
    threshold.SetUpperThreshold(upper_thresh)
    threshold.Update()  # 执行筛选

    # 3. 提取表面网格（STL仅支持表面，移除内部体网格）
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputConnection(threshold.GetOutputPort())
    geometry_filter.Update()

    # 4. 三角化网格（STL必需：强制转为三角面）
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(geometry_filter.GetOutput())
    triangle_filter.Update()

    # 5. 导出STL（二进制格式，体积更小）
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(stl_path)
    stl_writer.SetInputConnection(triangle_filter.GetOutputPort())
    stl_writer.SetFileTypeToBinary()
    stl_writer.Write()




# -------------------------- 调用示例 --------------------------
if __name__ == "__main__":
    # 替换为你的参数：VTU路径、STL输出路径、筛选的数组名、阈值范围
    path=f"data/vtk"
    filename="sol_083.vtu"
    vtu_threshold_to_stl(
        vtu_path=f"{path}/{filename}",
        stl_path=f"{path}/output_threshold.stl",
        scalar_name="all",  # 替换为VTU中实际的标量名（如Temperature、Displacement）
        lower_thresh=0,      # 阈值下限
        upper_thresh=2       # 阈值上限
    )
