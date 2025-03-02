import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_model import predict, smiles_to_graph

# 测试预测函数
def test_prediction():
    # 测试一些SMILES字符串
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因
        "C1=CC=C(C=C1)C(=O)O"  # 苯甲酸
    ]
    
    # 使用绝对路径定位模型文件
    model_path = "train_results/models/best_model.pt"
    print(f"使用模型路径: {model_path}")
    print("测试分子预测结果:")
    for smiles in test_smiles:
        result = predict(smiles, model_path=model_path)
        if result is not None:
            print(f"SMILES: {smiles}")
            print(f"预测值: {result:.4f}")
        else:
            print(f"SMILES: {smiles} - 无法解析")
    
    return True

def evaluate_qm9_predictions():
    """
    从QM9数据集读取SMILES和LUMO值，进行预测并评估模型性能
    """
    # 使用绝对路径定位模型文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = "train_results/models/best_model.pt"
    
    print(f"使用模型路径: {model_path}")
    print("正在评估QM9数据集上的模型性能...")
    
    # 读取QM9数据集 - 同样使用绝对路径
    qm9_path = "dataset/raw_dataset/qm9.csv"
    try:
        df = pd.read_csv(qm9_path)
        # 仅使用前10,000个数据点
        df = df.head(10000)
        smiles_list = df['smiles'].tolist()
        lumo_values = -df['lumo'].values * 27.2114  # 转换为eV
    except Exception as e:
        print(f"读取数据集失败: {e}")
        return False
    
    # 进行预测
    predictions = []
    valid_indices = []
    
    print(f"开始预测{len(smiles_list)}个分子...")
    for i, smiles in enumerate(smiles_list):
        if i % 100 == 0:
            print(f"已处理 {i}/{len(smiles_list)} 个分子")
        
        try:
            pred = predict(smiles, model_path=model_path)
            if pred is not None:
                predictions.append(pred)
                valid_indices.append(i)
        except Exception as e:
            print(f"预测分子 {smiles} 时出错: {e}")
    
    # 获取对应的真实值
    true_values = lumo_values[valid_indices]
    
    # 计算评估指标
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    rse = np.sqrt(np.sum((np.array(predictions) - true_values)**2) / np.sum(true_values**2))
    
    print(f"评估结果:")
    print(f"样本数量: {len(predictions)}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"RSE: {rse:.6f}")
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    
    # 预测值与真实值的散点图
    plt.subplot(2, 1, 1)
    plt.scatter(true_values, predictions, alpha=0.5)
    
    # 添加y=x线
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('真实值 (-LUMO)')
    plt.ylabel('预测值')
    plt.title('QM9数据集上的预测结果')
    
    # 绘制残差图
    plt.subplot(2, 1, 2)
    residuals = np.array(predictions) - true_values
    plt.scatter(true_values, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('真实值 (-LUMO)')
    plt.ylabel('残差 (预测值 - 真实值)')
    plt.title('残差分布')
    
    plt.tight_layout()
    plt.savefig('qm9_prediction_results.png')
    plt.show()
    
    return True

if __name__ == "__main__":
    test_prediction()
    evaluate_qm9_predictions()
