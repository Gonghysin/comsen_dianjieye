import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import MinMaxScaler
import os

# 默认数据集路径
DEFAULT_AFFINITY_FILE = os.path.join('data', 'smiles_affinity.csv')
DEFAULT_IONIZATION_FILE = os.path.join('data', 'smiles_ionization.csv')

def normalize_features(features):
    """归一化原子特征"""
    if features is None:
        return None
    
    scaler = MinMaxScaler()
    shape = features.shape
    flattened = features.reshape(-1, shape[-1])
    normalized = scaler.fit_transform(flattened)
    return normalized.reshape(shape)

def smiles_to_graph(smiles):
    """将SMILES转换为分子图"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    # 获取原子特征
    num_atoms = mol.GetNumAtoms()
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetIsAromatic(),
        ]
        atom_features.append(features)
    
    # 获取邻接矩阵
    adj_matrix = np.zeros((num_atoms, num_atoms))
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        adj_matrix[start, end] = bond_type
        adj_matrix[end, start] = bond_type
    
    atom_features = np.array(atom_features)
    
    # 归一化特征
    atom_features = normalize_features(atom_features)
    
    return atom_features, adj_matrix

def pad_array(array, target_size):
    """将数组填充到目标大小"""
    if array is None:
        return np.zeros((target_size, 5))  # 5是特征维度
    
    current_size = array.shape[0]
    if current_size >= target_size:
        return array[:target_size]
    
    padding = np.zeros((target_size - current_size, array.shape[1]))
    return np.vstack([array, padding])

def pad_adjacency(adj, target_size):
    """将邻接矩阵填充到目标大小"""
    if adj is None:
        return np.zeros((target_size, target_size))
    
    current_size = adj.shape[0]
    if current_size >= target_size:
        return adj[:target_size, :target_size]
    
    new_adj = np.zeros((target_size, target_size))
    new_adj[:current_size, :current_size] = adj
    return new_adj

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, properties, max_atoms=100):
        self.data = []
        self.properties = torch.tensor(properties, dtype=torch.float32)
        self.max_atoms = max_atoms
        
        print(f"\n处理分子数据...")
        print(f"总分子数: {len(smiles_list)}")
        
        for i, smiles in enumerate(smiles_list):
            atom_features, adj_matrix = smiles_to_graph(smiles)
            if atom_features is not None:
                # 填充到固定大小
                padded_features = pad_array(atom_features, max_atoms)
                padded_adj = pad_adjacency(adj_matrix, max_atoms)
                
                self.data.append({
                    'atom_features': torch.tensor(padded_features, dtype=torch.float32),
                    'adj_matrix': torch.tensor(padded_adj, dtype=torch.float32)
                })
            
            if i % 100 == 0:
                print(f"已处理 {i} 个分子")
        
        print(f"成功处理的分子数: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'atom_features': self.data[idx]['atom_features'],
            'adj_matrix': self.data[idx]['adj_matrix'],
            'properties': self.properties[idx]
        }

def normalize_properties(properties):
    """标准化属性值"""
    mean = np.mean(properties, axis=0)
    std = np.std(properties, axis=0)
    return (properties - mean) / std, mean, std

def denormalize_properties(normalized_properties, mean, std):
    """反标准化属性值"""
    return normalized_properties * std + mean

def load_and_preprocess_data(affinity_file, ionization_file, max_samples=1000):
    """加载并预处理数据
    Args:
        affinity_file: 亲和力数据文件
        ionization_file: 电离能数据文件
        max_samples: 最大样本数量，默认1000
    """
    print("开始加载数据...")
    
    # 读取亲和力数据
    affinity_df = pd.read_csv(affinity_file)
    ionization_df = pd.read_csv(ionization_file)
    
    # 随机选择指定数量的样本
    if max_samples and max_samples < len(affinity_df):
        indices = np.random.choice(len(affinity_df), max_samples, replace=False)
        affinity_df = affinity_df.iloc[indices]
        ionization_df = ionization_df.iloc[indices]
        print(f"随机选择 {max_samples} 个样本进行训练")
    
    # 获取SMILES和属性
    smiles_list = affinity_df.iloc[:, 0].tolist()
    affinity_values = affinity_df.iloc[:, 1].values
    ionization_values = ionization_df.iloc[:, 1].values
    
    # 数据清洗和归一化
    # 移除异常值（超过3个标准差的值）
    affinity_mean = np.mean(affinity_values)
    affinity_std = np.std(affinity_values)
    ionization_mean = np.mean(ionization_values)
    ionization_std = np.std(ionization_values)
    
    valid_indices = []
    for i in range(len(smiles_list)):
        if (abs(affinity_values[i] - affinity_mean) < 3 * affinity_std and
            abs(ionization_values[i] - ionization_mean) < 3 * ionization_std):
            valid_indices.append(i)
    
    smiles_list = [smiles_list[i] for i in valid_indices]
    affinity_values = affinity_values[valid_indices]
    ionization_values = ionization_values[valid_indices]
    
    # 重新计算均值和标准差
    affinity_mean = np.mean(affinity_values)
    affinity_std = np.std(affinity_values)
    ionization_mean = np.mean(ionization_values)
    ionization_std = np.std(ionization_values)
    
    # 归一化到[-1, 1]范围
    affinity_norm = (affinity_values - affinity_mean) / (affinity_std + 1e-8)
    ionization_norm = (ionization_values - ionization_mean) / (ionization_std + 1e-8)
    
    # 裁剪到[-3, 3]范围
    affinity_norm = np.clip(affinity_norm, -3, 3)
    ionization_norm = np.clip(ionization_norm, -3, 3)
    
    # 缩放到[-1, 1]范围
    affinity_norm = affinity_norm / 3
    ionization_norm = ionization_norm / 3
    
    properties = np.stack([affinity_norm, ionization_norm], axis=1)
    
    print(f"加载了 {len(smiles_list)} 条数据")
    print("\n电离度数据前5行：")
    print(ionization_df.head())
    print("\n属性统计：")
    print(f"均值: [{affinity_mean:.4f} {ionization_mean:.4f}]")
    print(f"标准差: [{affinity_std:.4f} {ionization_std:.4f}]")
    
    return smiles_list, properties, [affinity_mean, ionization_mean], [affinity_std, ionization_std] 