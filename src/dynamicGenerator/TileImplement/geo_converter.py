#!/usr/bin/env python3
"""
Gmsh .geo 文件处理工具
用于查看、转换和处理.geo文件
"""

import gmsh
import sys
import os

def view_geo_file(geo_file):
    """
    使用Gmsh GUI查看.geo文件
    """
    print(f"打开 {geo_file} ...")
    
    gmsh.initialize()
    gmsh.open(geo_file)
    
    # 启动GUI
    gmsh.fltk.run()
    
    gmsh.finalize()


def convert_geo_to_mesh(geo_file, output_format="msh", mesh_dim=3, mesh_size=0.1):
    """
    将.geo文件转换为网格文件
    
    参数:
    - geo_file: 输入的.geo文件
    - output_format: 输出格式 (msh, stl, vtk, ply, obj等)
    - mesh_dim: 网格维度 (1, 2, 3)
    - mesh_size: 网格尺寸
    """
    
    if not os.path.exists(geo_file):
        print(f"错误：文件 {geo_file} 不存在")
        return False
    
    print(f"转换 {geo_file} 为网格...")
    
    # 初始化Gmsh
    gmsh.initialize()
    
    try:
        # 打开.geo文件
        gmsh.open(geo_file)
        
        # 设置网格参数
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 1)
        
        # 生成网格
        print(f"生成{mesh_dim}D网格...")
        gmsh.model.mesh.generate(mesh_dim)
        
        # 获取网格信息
        nodes = gmsh.model.mesh.getNodes()
        elements = gmsh.model.mesh.getElements()
        print(f"节点数: {len(nodes[0])}")
        print(f"单元数: {sum(len(e) for e in elements[1]) if len(elements) > 1 else 0}")
        
        # 输出文件名
        base_name = os.path.splitext(geo_file)[0]
        output_file = f"{base_name}.{output_format}"
        
        # 保存网格
        gmsh.write(output_file)
        print(f"✓ 网格已保存到: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"错误: {e}")
        return False
        
    finally:
        gmsh.finalize()


def batch_convert_geo(geo_file, formats=["msh", "stl", "vtk"], mesh_size=0.1):
    """
    将.geo文件转换为多种格式
    """
    print(f"批量转换 {geo_file}")
    
    gmsh.initialize()
    
    try:
        gmsh.open(geo_file)
        
        # 设置网格参数
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2)
        
        # 生成3D网格
        print("生成3D网格...")
        gmsh.model.mesh.generate(3)
        
        # 保存为多种格式
        base_name = os.path.splitext(geo_file)[0]
        for fmt in formats:
            output_file = f"{base_name}.{fmt}"
            gmsh.write(output_file)
            print(f"✓ 已保存: {output_file}")
            
    except Exception as e:
        print(f"错误: {e}")
        
    finally:
        gmsh.finalize()


def geo_to_mesh_cli():
    """
    命令行接口
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Gmsh .geo文件转换工具')
    parser.add_argument('geo_file', help='输入的.geo文件')
    parser.add_argument('-f', '--format', default='msh', 
                       choices=['msh', 'stl', 'vtk', 'ply', 'obj', 'off'],
                       help='输出格式 (默认: msh)')
    parser.add_argument('-d', '--dimension', type=int, default=3,
                       choices=[1, 2, 3],
                       help='网格维度 (默认: 3)')
    parser.add_argument('-s', '--size', type=float, default=0.1,
                       help='网格尺寸 (默认: 0.1)')
    parser.add_argument('-v', '--view', action='store_true',
                       help='使用GUI查看')
    parser.add_argument('-b', '--batch', action='store_true',
                       help='批量转换为多种格式')
    
    args = parser.parse_args()
    
    if args.view:
        view_geo_file(args.geo_file)
    elif args.batch:
        batch_convert_geo(args.geo_file, mesh_size=args.size)
    else:
        convert_geo_to_mesh(args.geo_file, 
                           output_format=args.format,
                           mesh_dim=args.dimension,
                           mesh_size=args.size)


# 快速转换脚本
def quick_convert(geo_file):
    """
    快速转换.geo文件为常用格式
    """
    print(f"\n快速转换 {geo_file}")
    print("-" * 40)
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    # 打开文件
    gmsh.open(geo_file)
    
    # 快速网格生成
    print("使用快速网格生成...")
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 0)  # 跳过优化以加快速度
    
    # 自动设置网格尺寸
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.0)
    
    # 生成网格
    try:
        print("生成1D网格...")
        gmsh.model.mesh.generate(1)
        
        print("生成2D网格...")
        gmsh.model.mesh.generate(2)
        
        print("生成3D网格...")
        gmsh.model.mesh.generate(3)
        
        # 保存
        base = os.path.splitext(geo_file)[0]
        
        # MSH格式（Gmsh原生）
        msh_file = f"{base}.msh"
        gmsh.write(msh_file)
        print(f"✓ 保存MSH: {msh_file}")
        
        # STL格式（3D打印）
        stl_file = f"{base}.stl"
        gmsh.write(stl_file)
        print(f"✓ 保存STL: {stl_file}")
        
        # 获取统计信息
        nodes = gmsh.model.mesh.getNodes()
        print(f"\n网格统计:")
        print(f"  节点数: {len(nodes[0])}")
        
    except Exception as e:
        print(f"网格生成失败: {e}")
        print("尝试只保存几何体...")
        
        # 保存BREP格式（几何体）
        brep_file = f"{base}.brep"
        gmsh.write(brep_file)
        print(f"✓ 保存几何体: {brep_file}")
    
    gmsh.finalize()


if __name__ == "__main__":
    print("Gmsh .geo 文件工具")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # 命令行模式
        geo_to_mesh_cli()
    else:
        # 交互模式
        print("\n选择操作:")
        print("1. 查看.geo文件（GUI）")
        print("2. 转换为网格文件")
        print("3. 批量转换")
        print("4. 快速转换")
        
        choice = input("\n请选择 (1-4): ")
        
        if choice == '1':
            geo_file = input("输入.geo文件路径: ")
            view_geo_file(geo_file)
            
        elif choice == '2':
            geo_file = input("输入.geo文件路径: ")
            format_type = input("输出格式 (msh/stl/vtk) [默认msh]: ") or "msh"
            mesh_size = float(input("网格尺寸 [默认0.1]: ") or 0.1)
            convert_geo_to_mesh(geo_file, format_type, mesh_size=mesh_size)
            
        elif choice == '3':
            geo_file = input("输入.geo文件路径: ")
            batch_convert_geo(geo_file)
            
        elif choice == '4':
            geo_file = input("输入.geo文件路径: ")
            quick_convert(geo_file)