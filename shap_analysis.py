import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdmolops
from train_model import MolecularGraph, smiles_to_graph
import shap
from tqdm import tqdm
import random

def load_model(model_path):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path)
    model = MolecularGraph(num_features=1, dim=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def create_background_dataset(smiles_list, num_samples=150):
    """从SMILES列表中随机选择样本创建背景数据集"""
    selected_smiles = random.sample(smiles_list, min(num_samples, len(smiles_list)))
    graphs = []
    for smiles in selected_smiles:
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graph.smiles = smiles  # 保存SMILES字符串
            graphs.append(graph)
    return graphs

def compute_shap_values(model, background_graphs, target_graphs):
    """计算SHAP值"""
    def model_predict(graphs):
        predictions = []
        with torch.no_grad():
            for graph in graphs:
                out = model(graph)
                predictions.append(out.item())
        return np.array(predictions)
    
    # 创建SHAP解释器
    explainer = shap.KernelExplainer(model_predict, background_graphs)
    
    # 计算目标样本的SHAP值
    shap_values = explainer.shap_values(target_graphs)
    return shap_values

def visualize_atom_importance(shap_values, target_graphs, output_path):
    """可视化原子重要性"""
    # 提取原子SHAP值
    atom_shap_values = []
    for i, graph in enumerate(target_graphs):
        mol = Chem.MolFromSmiles(graph.smiles)
        if mol is not None:
            for j in range(len(mol.GetAtoms())):
                atom_shap_values.append({
                    'molecule_idx': i,
                    'atom_idx': j,
                    'shap_value': shap_values[i][j],
                    'atomic_num': mol.GetAtomWithIdx(j).GetAtomicNum()
                })
    
    # 创建热图数据
    df = pd.DataFrame(atom_shap_values)
    pivot_table = df.pivot_table(
        values='shap_value',
        index='molecule_idx',
        columns='atom_idx',
        aggfunc='mean'
    )
    
    # 绘制热图
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_table, cmap='RdBu_r', center=0)
    plt.title('原子SHAP值热图')
    plt.xlabel('原子索引')
    plt.ylabel('分子索引')
    plt.savefig(output_path + '_atom_importance.png')
    plt.close()

def visualize_bond_importance(shap_values, target_graphs, output_path):
    """可视化化学键重要性"""
    # 提取化学键SHAP值
    bond_shap_values = []
    for i, graph in enumerate(target_graphs):
        mol = Chem.MolFromSmiles(graph.smiles)
        if mol is not None:
            for j, bond in enumerate(mol.GetBonds()):
                bond_shap_values.append({
                    'molecule_idx': i,
                    'bond_idx': j,
                    'shap_value': shap_values[i][len(mol.GetAtoms()) + j],
                    'bond_type': bond.GetBondType()
                })
    
    # 创建热图数据
    df = pd.DataFrame(bond_shap_values)
    pivot_table = df.pivot_table(
        values='shap_value',
        index='molecule_idx',
        columns='bond_idx',
        aggfunc='mean'
    )
    
    # 绘制热图
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_table, cmap='RdBu_r', center=0)
    plt.title('化学键SHAP值热图')
    plt.xlabel('化学键索引')
    plt.ylabel('分子索引')
    plt.savefig(output_path + '_bond_importance.png')
    plt.close()

def main():
    # 加载模型
    model_path = "train_results/models/best_model.pt"
    model = load_model(model_path)
    
    # 读取数据集
    df = pd.read_csv('dataset/processed_dataset/processed_qm9.csv')
    smiles_list = df['x'].tolist()
    
    # 创建背景数据集和目标数据集
    print("创建背景数据集和目标数据集...")
    background_graphs = create_background_dataset(smiles_list, num_samples=150)
    target_graphs = create_background_dataset(smiles_list, num_samples=150)
    
    # 计算SHAP值
    print("计算SHAP值...")
    shap_values = compute_shap_values(model, background_graphs, target_graphs)
    
    # 可视化结果
    print("生成可视化结果...")
    output_path = "train_results/images/shap_analysis"
    visualize_atom_importance(shap_values, target_graphs, output_path)
    visualize_bond_importance(shap_values, target_graphs, output_path)
    
    print("分析完成！结果已保存到 train_results/images/shap_analysis_*.png")

if __name__ == "__main__":
    main() 