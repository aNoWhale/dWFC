import numpy as np
import json
import argparse
import os

def build_compliance_matrix(params):
    """根据材料参数构建6x6柔度矩阵（Voigt符号）"""
    S = np.zeros((6, 6))
    
    # 正轴方向弹性参数
    S[0, 0] = 1 / params['E11']    # S11
    S[1, 1] = 1 / params['E22']    # S22
    S[2, 2] = 1 / params['E33']    # S33
    
    # 泊松比相关项（强制矩阵对称性）
    S[0, 1] = S[1, 0] = -params['V12'] / params['E11']  # S12 = S21
    S[0, 2] = S[2, 0] = -params['V13'] / params['E11']  # S13 = S31
    S[1, 2] = S[2, 1] = -params['V23'] / params['E22']  # S23 = S32
    
    # 剪切模量相关项
    S[3, 3] = 1 / params['G23']    # S44（对应23平面剪切）
    S[4, 4] = 1 / params['G13']    # S55（对应13平面剪切）
    S[5, 5] = 1 / params['G12']    # S66（对应12平面剪切）
    
    return S

def rotation_matrix(axis, theta):
    """生成3x3旋转矩阵"""
    cosθ = np.cos(theta)
    sinθ = np.sin(theta)
    
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, cosθ, -sinθ],
            [0, sinθ, cosθ]
        ])
    elif axis == 'y':
        return np.array([
            [cosθ, 0, sinθ],
            [0, 1, 0],
            [-sinθ, 0, cosθ]
        ])
    elif axis == 'z':
        return np.array([
            [cosθ, -sinθ, 0],
            [sinθ, cosθ, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("旋转轴必须为 'x', 'y' 或 'z'")

def voigt_rotation_matrix(R):
    """将3x3旋转矩阵转换为6x6 Voigt符号旋转矩阵"""
    R11, R12, R13 = R[0]
    R21, R22, R23 = R[1]
    R31, R32, R33 = R[2]
    
    Rv = np.zeros((6, 6))
    # 正应力/正应变分量 (11, 22, 33)
    Rv[0] = [R11**2, R12**2, R13**2, R12*R13, R11*R13, R11*R12]
    Rv[1] = [R21**2, R22**2, R23**2, R22*R23, R21*R23, R21*R22]
    Rv[2] = [R31**2, R32**2, R33**2, R32*R33, R31*R33, R31*R32]
    # 剪切分量 (23, 13, 12)
    Rv[3] = [2*R21*R31, 2*R22*R32, 2*R23*R33, R22*R33+R23*R32, R21*R33+R23*R31, R21*R32+R22*R31]
    Rv[4] = [2*R11*R31, 2*R12*R32, 2*R13*R33, R12*R33+R13*R32, R11*R33+R13*R31, R11*R32+R12*R31]
    Rv[5] = [2*R11*R21, 2*R12*R22, 2*R13*R23, R12*R23+R13*R22, R11*R23+R13*R21, R11*R22+R12*R21]
    return Rv

def rotate_material_properties(params, axis, theta_deg):
    """旋转材料属性"""
    # 1. 构建原始柔度矩阵
    S_original = build_compliance_matrix(params)
    
    # 2. 计算旋转矩阵（转换为弧度）
    theta_rad = np.radians(theta_deg)
    R = rotation_matrix(axis, theta_rad)
    Rv = voigt_rotation_matrix(R)
    
    # 3. 旋转柔度矩阵：S' = Rv * S_original * Rv^T
    S_rotated = Rv @ S_original @ Rv.T
    
    # 4. 从旋转后的柔度矩阵提取材料参数
    params_rot = {}
    # 弹性模量
    params_rot['E11'] = 1 / S_rotated[0, 0]
    params_rot['E22'] = 1 / S_rotated[1, 1]
    params_rot['E33'] = 1 / S_rotated[2, 2]
    # 泊松比
    params_rot['V12'] = -S_rotated[0, 1] * params_rot['E11']
    params_rot['V13'] = -S_rotated[0, 2] * params_rot['E11']
    params_rot['V21'] = -S_rotated[1, 0] * params_rot['E22']
    params_rot['V23'] = -S_rotated[1, 2] * params_rot['E22']
    params_rot['V31'] = -S_rotated[2, 0] * params_rot['E33']
    params_rot['V32'] = -S_rotated[2, 1] * params_rot['E33']
    # 剪切模量
    params_rot['G12'] = 1 / S_rotated[5, 5]
    params_rot['G13'] = 1 / S_rotated[4, 4]
    params_rot['G23'] = 1 / S_rotated[3, 3]
    
    return params_rot

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='旋转各向异性材料属性参数')
    parser.add_argument('input_file', help='包含材料参数的JSON文件路径')
    parser.add_argument('--axis', choices=['x', 'y', 'z'], default='z', 
                      help='旋转轴（默认：z）')
    parser.add_argument('--angle', type=float, default=45, 
                      help='旋转角度（度，默认：45）')
    args = parser.parse_args()

    # 验证输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误：输入文件 '{args.input_file}' 不存在！")
        return

    # 读取输入文件（JSON格式）
    try:
        with open(args.input_file, 'r') as f:
            material_params = json.load(f)
        # 验证必要参数是否存在
        required_keys = {'E11', 'E22', 'E33', 'V12', 'V13', 'V21', 'V23', 'V31', 'V32', 'G12', 'G13', 'G23'}
        if not required_keys.issubset(material_params.keys()):
            missing = required_keys - material_params.keys()
            print(f"错误：输入文件缺少必要参数：{missing}")
            return
    except json.JSONDecodeError:
        print("错误：输入文件不是有效的JSON格式！")
        return
    except Exception as e:
        print(f"读取文件时出错：{str(e)}")
        return

    # 执行旋转计算
    try:
        rotated_params = rotate_material_properties(
            material_params, 
            args.axis, 
            args.angle
        )
    except Exception as e:
        print(f"计算时出错：{str(e)}")
        return

    # 确定输出文件路径（与脚本同目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
    input_filename = os.path.basename(args.input_file)
    input_name, input_ext = os.path.splitext(input_filename)
    output_filename = f"{input_name}_rot_{args.axis}{args.angle}deg{input_ext}"
    output_path = os.path.join(script_dir, output_filename)

    # 保存输出文件
    try:
        with open(output_path, 'w') as f:
            # 保留6位小数，确保JSON可读性
            json.dump(rotated_params, f, indent=4, sort_keys=True, ensure_ascii=False,
                      default=lambda x: round(x, 6) if isinstance(x, float) else x)
        print(f"旋转完成！结果已保存至：\n{output_path}")
    except Exception as e:
        print(f"保存文件时出错：{str(e)}")
        return

if __name__ == "__main__":
    """
    python rotate_material.py /path/to/your/material_params.json --axis z --angle 45
    """
    main()