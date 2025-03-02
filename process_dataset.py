# 用于处理数据集
import pandas as pd
import os

def process_csv(file_path, field_a, field_b):
    """
    处理CSV文件，提取指定的两个字段并创建新的CSV文件
    
    参数:
        file_path: 原CSV文件路径
        field_a: 要提取的第一个字段名
        field_b: 要提取的第二个字段名
    
    返回:
        新CSV文件的路径
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查字段是否存在
        if field_a not in df.columns or field_b not in df.columns:
            raise ValueError(f"字段 '{field_a}' 或 '{field_b}' 在CSV文件中不存在")
        
        # 创建新的DataFrame，只包含指定的两个字段，并对field_b进行转换
        new_df = pd.DataFrame({
            'x': df[field_a],
            'y': df[field_b] * -27.211386245988  # 将field_b的值乘以-27.211386245988
        })
        
        # 生成新文件名
        file_name = os.path.basename(file_path)
        new_file_name = f"processed_{file_name}"
        dir_name = os.path.dirname(file_path)
        new_file_path = os.path.join(dir_name, new_file_name)
        
        # 保存新的CSV文件
        new_df.to_csv(new_file_path, index=False)
        
        print(f"处理完成！新文件已保存为: {new_file_path}")
        return new_file_path
    
    except Exception as e:
        print(f"处理CSV文件时出错: {e}")
        return None

if __name__ == "__main__":
    # 获取用户输入
    csv_path = "dataset/raw_dataset/qm9.csv"
    field_a = "smiles"
    field_b = "lumo"
    
    
    # 处理CSV文件
    process_csv(csv_path, field_a, field_b)