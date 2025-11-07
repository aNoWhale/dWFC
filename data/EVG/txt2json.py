import sys
import os
import json
#  Property        Value                Unit        
#    E11     1702.3564146022377     Stress units    
#    V12    0.022572436378880425       ratio        
#    V13    0.022576361758055316       ratio        
#    E22     1702.6633785365532     Stress units    
#    V21    0.02257651025812507        ratio        
#    V23    0.02261441890609239        ratio        
#    E33     1702.505053691791      Stress units    
#    V31    0.022578331723600454       ratio        
#    V32    0.02261231590299088        ratio        
#    G12     7.410334696244505      Stress units    
#    G13     7.409309513770837      Stress units    
#    G23     7.414125470330955      Stress units    
#   CTE X           N/A                 N/A         
#   CTE Y           N/A                 N/A         
#   CTE Z           N/A                 N/A         
# Total mass=1.39996333684424e-15 Mass units 
# Homogenised density=1.74995428839888e-10 Density units 
# processing duration 470.000066995621 Seconds

def txt_to_json(txt_path):
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 提取输入文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(txt_path))[0]
    
    # 输出JSON文件路径（脚本目录下，同名）
    json_path = os.path.join(script_dir, f"{file_name}.json")
    
    # 存储提取的数据
    result = {}
    
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                # 去除首尾空白并分割行（处理多空格分隔）
                parts = line.strip().split()
                if not parts:
                    continue  # 跳过空行
                
                prop = parts[0]
                # 只保留E、V、G开头的属性
                if prop.startswith(('E', 'V', 'G')) and len(parts) >= 2:
                    try:
                        # 尝试将值转换为浮点数
                        value = float(parts[1])
                        result[prop] = value
                    except ValueError:
                        # 若转换失败则保留原始字符串（应对可能的非数字值）
                        result[prop] = parts[1]
        
        # 写入JSON文件
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        print(f"转换成功！JSON文件已保存至：{json_path}")
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {txt_path}")
    except Exception as e:
        print(f"转换失败：{str(e)}")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("使用方法：python 脚本名字.py txt文件路径")
        sys.exit(1)
    
    txt_file_path = sys.argv[1]
    txt_to_json(txt_file_path)

