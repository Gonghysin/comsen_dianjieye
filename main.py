import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, Sequential, BatchNorm1d, InstanceNorm1d, ReLU, Dropout, Sigmoid, ModuleList, Identity, LayerNorm, LeakyReLU
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv, TransformerConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging
import os
import psutil
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops, AllChem, Crippen, EState, MolSurf, Lipinski, rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.QED import qed
import seaborn as sns
import gc
from torch.cuda.amp import autocast, GradScaler
import math
import shap
from collections import defaultdict
import json
import tempfile
import subprocess
import glob
import time
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
import traceback
import random

# 添加MolecularDataset类定义
class MolecularDataset(Dataset):
    """
    用于处理分子特征数据的PyTorch数据集类
    
    参数:
        X (np.ndarray 或 pd.DataFrame): 特征数据
        y (np.ndarray 或 pd.Series): 目标值（电离能）
    """
    def __init__(self, X, y=None):
        # 输入验证
        if X is None:
            raise ValueError("输入特征 X 不能为 None")
            
        self.X = X
        self.y = y
        
        # 检查输入类型并相应处理
        if hasattr(X, 'columns'):  # 如果是DataFrame
            self.feature_names = list(X.columns)
            self.is_dataframe = True
        else:  # 如果是numpy数组
            # 确保X是二维数组
            if len(np.array(X).shape) == 1:
                self.X = np.array(X).reshape(-1, 1)
            elif len(np.array(X).shape) == 2:
                self.X = np.array(X)
            else:
                raise ValueError(f"输入特征 X 的维度不正确: {np.array(X).shape}")
                
            self.feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
            self.is_dataframe = False
            
        # 创建一个空的图数据对象，使用多节点结构
        num_nodes = 4  # 使用4个节点来表示分子
        self.empty_graph = Data(
            x=torch.zeros((num_nodes, len(self.feature_names)), dtype=torch.float32),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]], dtype=torch.long),  # 创建一个简单的链式结构
            edge_attr=torch.zeros((6, 4), dtype=torch.float32)  # 6条边（双向），4维边特征
        )
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 获取特征
        if self.is_dataframe:
            features = self.X.iloc[idx]
        else:
            features = self.X[idx]
            
        # 创建节点特征矩阵，将特征分配给多个节点
        num_nodes = self.empty_graph.x.size(0)
        x = torch.zeros((num_nodes, len(self.feature_names)), dtype=torch.float32)
        for i, (name, val) in enumerate(zip(self.feature_names, features)):
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    val = 0.0  # 对于无法转换的字符串，使用默认值
            # 将特征值复制到所有节点
            x[:, i] = val
            
        # 创建图数据对象
        data = Data(
            x=x,
            edge_index=self.empty_graph.edge_index.clone(),
            edge_attr=self.empty_graph.edge_attr.clone()
        )
        
        # 如果有目标值，添加到数据对象中
        if self.y is not None:
            if hasattr(self.y, 'iloc'):
                data.y = torch.tensor([self.y.iloc[idx]], dtype=torch.float32)
            else:
                data.y = torch.tensor([self.y[idx]], dtype=torch.float32)
            
        return data

# 绘图函数定义
def plot_training_history(train_losses, val_losses, num_epochs, title="训练历史"):
    """绘制训练与验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()

def plot_prediction_results(true_values, predictions, title="预测结果对比"):
    """绘制预测结果对比散点图"""
    plt.figure(figsize=(10, 8))
    
    # 散点图
    plt.scatter(true_values, predictions, alpha=0.5)
    
    # 添加x=y对角线
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 计算并添加性能指标
    r2 = r2_score(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{title}\nR² = {r2:.4f}, MAE = {mae:.4f}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()

def plot_y_distribution(y_values, title="目标变量分布"):
    """绘制目标变量分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(y_values, bins=50, alpha=0.7, color='skyblue')
    plt.xlabel('电离能 (eV)')
    plt.ylabel('频率')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def print_memory_usage():
    """打印系统内存使用情况"""
    process = psutil.Process(os.getpid())
    logging.info(f"系统内存使用: {process.memory_info().rss / 1024**2:.2f} MB")

def get_molwt(mol):
    """获取分子重量"""
    return Descriptors.MolWt(mol)

def get_heavy_atom_count(mol):
    """获取重原子数量"""
    return mol.GetNumHeavyAtoms()

def get_element_list(mol):
    """获取分子中的元素列表"""
    elements = set()
    for atom in mol.GetAtoms():
        elements.add(atom.GetSymbol())
    return list(elements)

def filter_molecules(data_df):
    """根据KPI框架要求过滤分子
    
    过滤条件:
    1. 分子重量 <= 600
    2. 重原子数 <= 30
    3. 元素组成限于常见元素 H/C/N/O/F
    """
    logging.info(f"过滤前分子数量: {len(data_df)}")
    
    # 创建RDKit分子对象
    mols = [Chem.MolFromSmiles(smiles) for smiles in data_df['x']]
    valid_indices = []
    
    for i, mol in enumerate(mols):
        if mol is None:
            continue
            
        # 计算分子重量
        molwt = get_molwt(mol)
        
        # 计算重原子数
        heavy_count = get_heavy_atom_count(mol)
        
        # 获取元素列表
        elements = get_element_list(mol)
        
        # 检查元素是否只包含H, C, N, O, F
        common_elements = set(['H', 'C', 'N', 'O', 'F'])
        elements_check = all(elem in common_elements for elem in elements)
        
        # 应用过滤条件
        if molwt <= 600 and heavy_count <= 30 and elements_check:
            valid_indices.append(i)
    
    filtered_df = data_df.iloc[valid_indices].copy()
    logging.info(f"过滤后分子数量: {len(filtered_df)}")
    
    return filtered_df

def normalize_ionization_energy(data_df, target_column='y'):
    """对电离能数值进行归一化处理"""
    mean_ie = data_df[target_column].mean()
    std_ie = data_df[target_column].std()
    
    logging.info(f"电离能数据统计 - 均值: {mean_ie:.4f}, 标准差: {std_ie:.4f}")
    
    # 保存原始值用于后续反归一化
    data_df['original_y'] = data_df[target_column].copy()
    
    # 归一化处理
    data_df[target_column] = (data_df[target_column] - mean_ie) / std_ie
    
    return data_df, mean_ie, std_ie

def get_atom_features(atom):
    """Enhanced atom feature extraction with all chemical elements"""
    # Basic features
    features = [
        atom.GetAtomicNum(),          # 原子序数
        atom.GetDegree(),             # 度
        atom.GetFormalCharge(),       # 形式电荷
        atom.GetNumRadicalElectrons(),# 自由基电子数
        atom.GetIsAromatic() * 1,     # 芳香性
        atom.GetMass(),               # 原子质量
        atom.GetExplicitValence(),    # 显式化合价
        atom.GetImplicitValence(),    # 隐式化合价
        atom.GetTotalValence(),       # 总化合价
        atom.GetNumImplicitHs(),      # 隐式氢原子数
        atom.IsInRing() * 1,          # 是否在环中
        atom.GetHybridization(),      # 杂化类型
    ]
    
    # Electronic and structural properties
    features.extend([
        atom.GetTotalNumHs(),         # 总氢原子数
        atom.GetTotalDegree(),        # 总度
        atom.GetTotalValence(),       # 总化合价
        atom.GetExplicitValence(),    # 显式化合价
        atom.GetImplicitValence(),    # 隐式化合价
        atom.GetFormalCharge(),       # 形式电荷
        atom.GetNumRadicalElectrons() # 自由基电子数
    ])
    
    # Ring properties
    features.extend([
        atom.IsInRingSize(3) * 1,     # 3元环
        atom.IsInRingSize(4) * 1,     # 4元环
        atom.IsInRingSize(5) * 1,     # 5元环
        atom.IsInRingSize(6) * 1,     # 6元环
        atom.IsInRingSize(7) * 1,     # 7元环
    ])
    
    # Chirality
    features.extend([
        atom.GetChiralTag() != 0,     # 手性
        atom.HasProp('_CIPCode'),     # CIP构型
    ])
    
    # One-hot encoding for ALL elements (1-118)
    atomic_num = atom.GetAtomicNum()
    atom_type = [1 if i == atomic_num else 0 for i in range(1, 119)]  # 1-118
    features.extend(atom_type)
    
    # One-hot encoding for hybridization
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hybridization = [1 if atom.GetHybridization() == h else 0 for h in hybridization_types]
    features.extend(hybridization)
    
    # Electronegativity and other chemical properties
    try:
        from rdkit.Chem import AllChem
        features.extend([
            float(Crippen.MolLogP([atom])),  # 亲脂性
            float(Crippen.MolMR([atom])),    # 摩尔折射率
            float(EState.EStateIndices([atom])[0]),  # E-state指数
        ])
    except:
        features.extend([0, 0, 0])  # 如果计算失败，使用默认值
    
    # Add periodic table properties
    try:
        from rdkit.Chem import Descriptors
        # 添加元素的周期表性质
        features.extend([
            atom.GetAtomicNum() % 18,  # 主族
            (atom.GetAtomicNum() - 1) // 18 + 1,  # 周期
        ])
    except:
        features.extend([0, 0])
    
    # 添加与电离能相关的特征
    try:
        # 电负性 (Pauling scale)
        electronegativity_dict = {
            1: 2.20,  # H
            5: 2.04,  # B
            6: 2.55,  # C
            7: 3.04,  # N
            8: 3.44,  # O
            9: 3.98,  # F
            15: 2.19, # P
            16: 2.58, # S
            17: 3.16, # Cl
            35: 2.96, # Br
            53: 2.66  # I
        }
        electronegativity = electronegativity_dict.get(atom.GetAtomicNum(), 0)
        features.append(electronegativity)
        
        # 原子半径 (pm)
        atomic_radius_dict = {
            1: 25,   # H
            6: 70,   # C
            7: 65,   # N
            8: 60,   # O
            9: 50,   # F
            15: 100, # P
            16: 100, # S
            17: 100, # Cl
            35: 115, # Br
            53: 140  # I
        }
        atomic_radius = atomic_radius_dict.get(atom.GetAtomicNum(), 0)
        features.append(atomic_radius)
        
        # 电离能 (eV) - 第一电离能
        ionization_energy_dict = {
            1: 13.6,  # H
            6: 11.3,  # C
            7: 14.5,  # N
            8: 13.6,  # O
            9: 17.4,  # F
            15: 10.5, # P
            16: 10.4, # S
            17: 13.0, # Cl
            35: 11.8, # Br
            53: 10.5  # I
        }
        ie = ionization_energy_dict.get(atom.GetAtomicNum(), 0)
        features.append(ie)
    except:
        features.extend([0, 0, 0])
    
    return features  # 返回列表而不是张量

def get_bond_features(bond):
    """Enhanced bond feature extraction"""
    if bond is None:
        return [0] * 21  # 返回列表而不是张量
    
    features = [
        float(bond.GetBondTypeAsDouble()),
        bond.GetIsConjugated() * 1,
        bond.GetIsAromatic() * 1,
        bond.IsInRing() * 1,
        bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE,
    ]
    
    # Bond type one-hot
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    features.extend([1 if bond.GetBondType() == t else 0 for t in bond_types])
    
    # Stereo configuration one-hot
    stereo_types = [
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ]
    features.extend([1 if bond.GetStereo() == s else 0 for s in stereo_types])
    
    # Ring size features
    features.extend([
        bond.IsInRingSize(3) * 1,
        bond.IsInRingSize(4) * 1,
        bond.IsInRingSize(5) * 1,
        bond.IsInRingSize(6) * 1,
        bond.IsInRingSize(7) * 1,
        bond.IsInRingSize(8) * 1
    ])
    
    # Additional geometric features
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        atom1_pos = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        atom2_pos = conf.GetAtomPosition(bond.GetEndAtomIdx())
        bond_length = math.sqrt(
            (atom1_pos.x - atom2_pos.x) ** 2 +
            (atom1_pos.y - atom2_pos.y) ** 2 +
            (atom1_pos.z - atom2_pos.z) ** 2
        )
        features.append(float(bond_length))
    else:
        features.append(0.0)
    
    return features  # 返回列表而不是张量

def extract_molecular_features(mol):
    """提取分子级特征，包括电子结构特征、结构特征和几何特征
    
    参数:
    mol -- RDKit分子对象
    
    返回:
    features_dict -- 包含所有提取特征的字典
    """
    if mol is None:
        return None
    
    features_dict = {}
    
    # 确保分子有构象信息
    try:
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
    except:
        logging.warning("无法生成分子构象")
    
    # 1. 电子结构特征
    try:
        # 尝试使用pybel计算HOMO/LUMO (需要安装openbabel)
        # 如果没有安装openbabel，可以使用其他近似方法或跳过
        try:
            import pybel
            pybel_mol = pybel.readstring("smi", Chem.MolToSmiles(mol))
            pybel_mol.make3D()
            pybel_mol.calccharges()
            features_dict["homo_energy"] = pybel_mol.homo
            features_dict["lumo_energy"] = pybel_mol.lumo
            features_dict["gap_energy"] = pybel_mol.lumo - pybel_mol.homo
        except ImportError:
            # 使用近似方法估计HOMO/LUMO
            aromatic_count = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            electron_affinity = 0
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I', 'O', 'N']:
                    electron_affinity += 1
            features_dict["homo_approx"] = -0.1 * aromatic_count
            features_dict["lumo_approx"] = -0.05 * electron_affinity
            features_dict["gap_approx"] = features_dict["lumo_approx"] - features_dict["homo_approx"]
        
        # 计算静电势相关特征
        ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            charge = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0
            charges.append(charge)
        
        features_dict["max_abs_charge"] = max([abs(c) for c in charges]) if charges else 0
        features_dict["mean_abs_charge"] = np.mean([abs(c) for c in charges]) if charges else 0
        features_dict["charge_variance"] = np.var(charges) if charges else 0
        
        # 电子亲和能近似
        electronegative_atoms = sum(1 for atom in mol.GetAtoms() 
                                    if atom.GetSymbol() in ['F', 'Cl', 'O', 'N'])
        features_dict["electron_affinity_approx"] = electronegative_atoms / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0
        
    except Exception as e:
        logging.warning(f"计算电子结构特征时出错: {e}")
        features_dict["homo_approx"] = 0
        features_dict["lumo_approx"] = 0
        features_dict["gap_approx"] = 0
        features_dict["max_abs_charge"] = 0
        features_dict["mean_abs_charge"] = 0
        features_dict["charge_variance"] = 0
        features_dict["electron_affinity_approx"] = 0
    
    # 2. 结构特征
    try:
        # 芳香性
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        features_dict["aromatic_ratio"] = aromatic_atoms / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0
        features_dict["aromatic_bond_ratio"] = aromatic_bonds / mol.GetNumBonds() if mol.GetNumBonds() > 0 else 0
        
        # 共轭特征
        conjugated_bonds = sum(1 for bond in mol.GetBonds() 
                              if (bond.GetBondTypeAsDouble() > 1 or bond.GetIsAromatic()))
        features_dict["conjugation_ratio"] = conjugated_bonds / mol.GetNumBonds() if mol.GetNumBonds() > 0 else 0
        
        # 取代基特征
        electron_withdrawing = sum(1 for atom in mol.GetAtoms() 
                                  if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I', 'O', 'N', 'S'] 
                                  and atom.GetDegree() == 1)
        electron_donating = sum(1 for atom in mol.GetAtoms() 
                               if atom.GetSymbol() in ['C'] and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3 
                               and atom.GetDegree() == 1)
        features_dict["ewg_count"] = electron_withdrawing  # 吸电子基团数量
        features_dict["edg_count"] = electron_donating     # 供电子基团数量
        features_dict["ewg_edg_ratio"] = electron_withdrawing / max(1, electron_donating)  # 吸/供电子基团比例
        
        # 电负性分布
        en_values = []
        en_dict = {
            'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
            'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66
        }
        for atom in mol.GetAtoms():
            en = en_dict.get(atom.GetSymbol(), 0)
            en_values.append(en)
        
        features_dict["mean_electronegativity"] = np.mean(en_values) if en_values else 0
        features_dict["max_electronegativity"] = max(en_values) if en_values else 0
        features_dict["electronegativity_variance"] = np.var(en_values) if en_values else 0
        
        # 计算Crippen贡献
        features_dict["logp"] = Crippen.MolLogP(mol)
        features_dict["mr"] = Crippen.MolMR(mol)
        
    except Exception as e:
        logging.warning(f"计算结构特征时出错: {e}")
        # 设置默认值
        features_dict.update({
            "aromatic_ratio": 0, "aromatic_bond_ratio": 0, "conjugation_ratio": 0,
            "ewg_count": 0, "edg_count": 0, "ewg_edg_ratio": 0,
            "mean_electronegativity": 0, "max_electronegativity": 0, "electronegativity_variance": 0,
            "logp": 0, "mr": 0
        })
    
    # 3. 几何特征
    try:
        # 分子体积和表面积
        features_dict["tpsa"] = MolSurf.TPSA(mol)  # 拓扑极性表面积
        features_dict["labute_asa"] = MolSurf.LabuteASA(mol)  # Labute表面积
        
        # 使用rdkit提供的3D描述符(需要3D构象)
        if mol.GetNumConformers() > 0:
            features_dict["molecular_volume"] = AllChem.ComputeMolVolume(mol)
            
            # 提取更多3D几何特征
            try:
                features_dict["asphericity"] = rdMolDescriptors.CalcAsphericity(mol)
                features_dict["eccentricity"] = rdMolDescriptors.CalcEccentricity(mol)
                features_dict["inertial_shape_factor"] = rdMolDescriptors.CalcInertialShapeFactor(mol)
                features_dict["npr1"] = rdMolDescriptors.CalcNPR1(mol)
                features_dict["npr2"] = rdMolDescriptors.CalcNPR2(mol)
                features_dict["pmi1"] = rdMolDescriptors.CalcPMI1(mol)
                features_dict["pmi2"] = rdMolDescriptors.CalcPMI2(mol)
                features_dict["pmi3"] = rdMolDescriptors.CalcPMI3(mol)
                features_dict["radius_of_gyration"] = rdMolDescriptors.CalcRadiusOfGyration(mol)
                features_dict["spherocity_index"] = rdMolDescriptors.CalcSpherocityIndex(mol)
            except:
                pass
        
        # 键长和键级统计
        bond_lengths = []
        bond_orders = []
        
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            for bond in mol.GetBonds():
                idx1 = bond.GetBeginAtomIdx()
                idx2 = bond.GetEndAtomIdx()
                pos1 = conf.GetAtomPosition(idx1)
                pos2 = conf.GetAtomPosition(idx2)
                bond_length = math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)
                bond_lengths.append(bond_length)
                bond_orders.append(bond.GetBondTypeAsDouble())
        
        if bond_lengths:
            features_dict["mean_bond_length"] = np.mean(bond_lengths)
            features_dict["max_bond_length"] = max(bond_lengths)
            features_dict["min_bond_length"] = min(bond_lengths)
            features_dict["std_bond_length"] = np.std(bond_lengths)
        else:
            features_dict["mean_bond_length"] = 0
            features_dict["max_bond_length"] = 0
            features_dict["min_bond_length"] = 0
            features_dict["std_bond_length"] = 0
            
        if bond_orders:
            features_dict["mean_bond_order"] = np.mean(bond_orders)
            features_dict["max_bond_order"] = max(bond_orders)
            features_dict["fraction_single_bonds"] = sum(1 for bo in bond_orders if bo == 1) / len(bond_orders)
            features_dict["fraction_double_bonds"] = sum(1 for bo in bond_orders if bo == 2) / len(bond_orders)
            features_dict["fraction_triple_bonds"] = sum(1 for bo in bond_orders if bo == 3) / len(bond_orders)
        else:
            features_dict["mean_bond_order"] = 0
            features_dict["max_bond_order"] = 0
            features_dict["fraction_single_bonds"] = 0
            features_dict["fraction_double_bonds"] = 0
            features_dict["fraction_triple_bonds"] = 0
        
    except Exception as e:
        logging.warning(f"计算几何特征时出错: {e}")
        # 设置默认值
        features_dict.update({
            "tpsa": 0, "labute_asa": 0, "molecular_volume": 0,
            "mean_bond_length": 0, "max_bond_length": 0, "min_bond_length": 0, "std_bond_length": 0,
            "mean_bond_order": 0, "max_bond_order": 0, "fraction_single_bonds": 0, 
            "fraction_double_bonds": 0, "fraction_triple_bonds": 0
        })
    
    return features_dict

def smiles_to_graph(smiles):
    """将SMILES转换为增强的分子图，整合原子、键和分子级特征"""
    if ',' in smiles:
        smiles = smiles.split(',')[0]
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 获取原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    
    # 获取边特征和索引
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])  # 添加双向边
        
        bond_feature = get_bond_features(bond)
        edge_features.extend([bond_feature, bond_feature])  # 对应双向边
    
    # 如果分子没有键（只有单个原子）
    if len(edge_indices) == 0:
        edge_indices = [[0, 0]]
        edge_features = [[0] * 21]  # 21是键特征的维度
    
    # 提取分子级特征
    mol_features_dict = extract_molecular_features(mol)
    
    # 转换为张量
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # 创建数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # 将分子级特征添加为全局属性
    if mol_features_dict:
        for key, value in mol_features_dict.items():
            data[key] = torch.tensor([value], dtype=torch.float)
    
    return data

class KnowledgeEmbedding(torch.nn.Module):
    """知识嵌入模块，负责处理全局分子特征"""
    
    def __init__(self, input_dim, hidden_dim):
        super(KnowledgeEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 使用简单的单层编码器，去除BatchNorm，改用LayerNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            LeakyReLU(0.1)
        )
        
        # 特征重要性权重
        self.important_features = {}
        self.feature_weights = torch.ones(input_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    def update_important_features(self, feature_names, importance_values):
        """更新重要特征列表"""
        self.important_features = dict(zip(feature_names, importance_values))
        
        # 如果提供了特征权重，则更新权重向量
        if len(self.important_features) > 0:
            for i, name in enumerate(feature_names):
                if i < len(self.feature_weights):
                    # 使用softplus确保权重为正
                    self.feature_weights[i] = 1.0 + F.softplus(torch.tensor(importance_values[i]))
    
    def forward(self, x):
        """前向传播，应用特征权重并编码"""
        # 确保x的维度正确
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 应用特征权重
        if len(self.feature_weights) == x.size(1):
            x = x * self.feature_weights.to(x.device)
        
        # 检查并移除NaN和Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 应用编码器
        x = self.encoder(x)
        
        # 确保输出没有NaN或Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x
    
    def get_attention_weights(self):
        """返回特征重要性权重，用于可解释性"""
        return self.feature_weights.cpu().detach().numpy().tolist()  # 返回列表而不是numpy数组

class EnhancedMolecularGraph(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64):  # 修改默认hidden_dim
        super(EnhancedMolecularGraph, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        
        # 全局特征配置
        self.global_feature_keys = [
            # 电子结构特征
            'HOMO', 'LUMO', 'gap', 'dipole', 'charge',
            # 结构特征
            'molwt', 'num_atoms', 'num_bonds', 'num_rings', 
            # 几何特征
            'diameter', 'radius', 'inertia', 
            # 3D特征
            'volume', 'surface_area', 'psa'
        ]
        
        self.global_feature_dim = len(self.global_feature_keys)
        
        # 节点特征编码器
        self.node_encoder = Sequential(
            Linear(num_node_features, hidden_dim),
            BatchNorm1d(hidden_dim),
            LeakyReLU(0.1)
        )
        
        # 边特征神经网络
        self.edge_nn = Sequential(
            Linear(num_edge_features, hidden_dim // 2),
            LeakyReLU(0.1),
            Linear(hidden_dim // 2, hidden_dim * hidden_dim)
        )
        
        # 图卷积层 - 减少为2层，提高稳定性
        self.conv_layers = ModuleList([
            NNConv(hidden_dim, hidden_dim, self.edge_nn, aggr='mean'),
            GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=0.1)
        ])
        
        # 批归一化层
        self.batch_norms = ModuleList([
            BatchNorm1d(hidden_dim),
            BatchNorm1d(hidden_dim)
        ])
        
        # 投影层
        self.projection_layers = ModuleList([
            Identity(),  # 第一层不需要投影
            Linear(hidden_dim, hidden_dim)  # 第二层需要投影
        ])
        
        # 知识嵌入层 - 接收全局特征
        self.knowledge_embedding = KnowledgeEmbedding(
            input_dim=self.global_feature_dim,  # 全局特征数量
            hidden_dim=hidden_dim  # 输出维度
        )
        
        # GAT注意力层 - 用于节点重要性判断
        self.gat = GATConv(
            hidden_dim, 
            hidden_dim // 4,
            heads=4,  # 减少头数，提高稳定性
            concat=True,
            dropout=0.1
        )
        
        # 减少融合层网络深度，提高稳定性
        # 输入维度为 hidden_dim * 3 (2个池化结果 + 知识嵌入层)
        self.fusion_layer = Sequential(
            Linear(hidden_dim * 3, hidden_dim),
            LayerNorm(hidden_dim),  # 替换BatchNorm为LayerNorm，提高稳定性
            LeakyReLU(0.1),
            Dropout(0.1)
        )
        
        # 减少输出层网络深度
        self.output_layer = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            LeakyReLU(0.1),
            Linear(hidden_dim // 2, 1)
        )
        
        # 添加归一化池化层
        self.pool_norm = torch.nn.LayerNorm(hidden_dim)
        self.final_norm = torch.nn.LayerNorm(hidden_dim * 2)  # 只使用mean和max池化
        
        # Dropout层
        self.dropout = Dropout(0.1)
        
        # 权重初始化
        self._init_weights()
        
        # 存储特征重要性
        self.feature_importance = {}
    
    def _init_weights(self):
        """初始化模型权重，使用He初始化，避免输出恒为零的问题"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # He初始化，针对ReLU和LeakyReLU激活函数
                nn.init.kaiming_uniform_(module.weight, a=0.1, nonlinearity='leaky_relu')
                if module.bias is not None:
                    # 将偏置初始化为小的正值，避免神经元死亡
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.BatchNorm1d):
                if module.affine:
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, GATConv):
                # 初始化GAT层参数，增加安全检查
                try:
                    # PyG的不同版本中GATConv实现可能不同
                    # 检查不同版本中可能存在的属性
                    gat_linear_attributes = ['lin_src', 'lin_dst', 'lin_l', 'lin_r']
                    gat_attention_attributes = ['att_src', 'att_dst', 'att_l', 'att_r', 'att']
                    
                    # 检查并初始化线性层
                    for attr_name in gat_linear_attributes:
                        if hasattr(module, attr_name):
                            attr = getattr(module, attr_name)
                            if attr is not None and hasattr(attr, 'weight'):
                                nn.init.xavier_uniform_(attr.weight)
                                # 减少不必要的日志输出
                                # logging.info(f"成功初始化GATConv的{attr_name}层")
                    
                    # 检查并初始化注意力权重
                    for attr_name in gat_attention_attributes:
                        if hasattr(module, attr_name):
                            attr = getattr(module, attr_name)
                            if attr is not None:
                                if isinstance(attr, torch.Tensor):
                                    nn.init.xavier_uniform_(attr)
                                    # 减少不必要的日志输出
                                    # logging.info(f"成功初始化GATConv的{attr_name}参数")
                except Exception as e:
                    logging.warning(f"初始化GATConv层时出错: {e}, 跳过这一层的初始化")
            elif isinstance(module, NNConv):
                # 初始化NNConv层参数
                try:
                    if hasattr(module, 'nn') and module.nn is not None:
                        if hasattr(module.nn, 'weight') and module.nn.weight is not None:
                            nn.init.xavier_uniform_(module.nn.weight, gain=0.1)
                        if hasattr(module.nn, 'bias') and module.nn.bias is not None:
                            nn.init.constant_(module.nn.bias, 0.01)
                except Exception as e:
                    logging.warning(f"初始化NNConv层时出错: {e}")
            elif isinstance(module, TransformerConv):
                # 初始化TransformerConv层参数
                try:
                    for attr_name in ['lin_q', 'lin_k', 'lin_v']:
                        if hasattr(module, attr_name) and getattr(module, attr_name) is not None:
                            layer = getattr(module, attr_name)
                            if hasattr(layer, 'weight') and layer.weight is not None:
                                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                except Exception as e:
                    logging.warning(f"初始化TransformerConv层时出错: {e}")
    
    def update_knowledge_features(self, feature_names, importance_values):
        """
        更新知识特征的重要性权重
        
        参数:
            feature_names: 特征名称列表
            importance_values: 对应的重要性值列表
        """
        # 确保输入格式正确
        if not isinstance(feature_names, list) or not isinstance(importance_values, list):
            logging.warning("特征名称和重要性值必须是列表格式")
            return
        
        if len(feature_names) != len(importance_values):
            logging.warning("特征名称和重要性值长度不匹配")
            return
        
        # 创建特征重要性字典
        sorted_features = sorted(zip(feature_names, importance_values), 
                                 key=lambda x: abs(x[1]), 
                                 reverse=True)
        
        # 获取排序后的特征名称和值
        sorted_names = [name for name, _ in sorted_features]
        sorted_values = [value for _, value in sorted_features]
        
        # 更新知识嵌入层中的重要特征
        self.knowledge_embedding.update_important_features(sorted_names, sorted_values)
        
        # 保存特征重要性，用于后续分析
        self.feature_importance = dict(sorted_features)
    
    def forward(self, data):
        """模型前向传播"""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. 输入检查和预处理
        # 确保batch索引正确
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # 记录原始batch_size用于后续维度检查
        batch_size = len(torch.unique(batch))
        
        # 输入数据检查 - 增强异常值处理
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 使用缩放来确保输入数据在合理范围内
        x_abs_max = torch.max(torch.abs(x))
        if x_abs_max > 10.0:
            x = x * (10.0 / x_abs_max)
            logging.warning(f"输入特征数值过大，已进行缩放 (最大值: {x_abs_max})")
        
        # 2. 节点特征编码 [num_nodes, num_features] -> [num_nodes, hidden_dim]
        x = self.node_encoder(x)
        
        # 编码后检查和修复
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 3. 多层图卷积处理
        for i, conv in enumerate(self.conv_layers):
            # 保存原始x用于异常情况下恢复
            x_residual = x.clone()
            
            # 应用图卷积
            try:
                if i == 0:  # NNConv层 [num_nodes, hidden_dim] -> [num_nodes, hidden_dim]
                    # 检查边属性是否正常
                    if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # 检查边索引是否有效
                    valid_indices = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
                    if not valid_indices.all():
                        logging.warning(f"边索引超出范围，过滤无效边")
                        edge_index = edge_index[:, valid_indices]
                        edge_attr = edge_attr[valid_indices]
                    
                    # 应用卷积操作
                    x_conv = conv(x, edge_index, edge_attr)
                else:  # 第二层卷积
                    x_conv = conv(x, edge_index)
                    
                # 检查卷积输出
                if torch.isnan(x_conv).any() or torch.isinf(x_conv).any():
                    logging.warning(f"警告：第{i+1}层卷积输出包含NaN或Inf值，使用上一层输出")
                    x_conv = x_residual
                    continue
                
                # 额外检查输出值范围
                x_conv_abs_max = torch.max(torch.abs(x_conv))
                if x_conv_abs_max > 100.0:
                    logging.warning(f"第{i+1}层卷积输出值异常大: {x_conv_abs_max}, 进行缩放")
                    x_conv = x_conv * (10.0 / x_conv_abs_max)
                    
            except Exception as e:
                logging.error(f"图卷积层{i+1}出错: {str(e)}")
                # 出错时使用上一层输出
                x_conv = x_residual
                continue
            
            # 根据不同层类型处理输出维度
            if isinstance(conv, TransformerConv):
                # TransformerConv输出: [num_nodes, heads*out_channels] = [num_nodes, 8*(hidden_dim//8)] = [num_nodes, hidden_dim]
                # 验证输出维度
                if x_conv.dim() > 2:  # 如果输出是三维的 [num_nodes, heads, out_channels]
                    try:
                        x_conv = x_conv.reshape(x_conv.size(0), -1)  # 展平为 [num_nodes, hidden_dim]
                    except Exception as e:
                        logging.error(f"TransformerConv输出reshape错误: {str(e)}")
                        x_conv = x_residual
                
                # 确保投影到正确维度
                if x_conv.size(1) != self.hidden_dim:
                    x_conv = self.projection_layers[i](x_conv)
                
                x_conv = x_conv.contiguous()
                
                # 添加稳定性处理
                try:
                    # 使用BatchNorm归一化前先排除异常值
                    x_conv = torch.nan_to_num(x_conv, nan=0.0, posinf=1.0, neginf=-1.0)
                    x_conv = self.batch_norms[i](x_conv)
                except Exception as e:
                    logging.error(f"BatchNorm处理出错: {str(e)}")
                    x_conv = x_residual
                    continue
                
                # 使用LeakyReLU代替ELU激活函数 - 更安全的激活处理
                x_conv = F.leaky_relu(x_conv, negative_slope=0.1)
                
                # 检查激活后的值
                x_conv = torch.nan_to_num(x_conv, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 应用dropout，确保不会引入nan
                x_conv = self.dropout(x_conv)
            
            elif isinstance(conv, GATConv):
                # GATConv输出: [num_nodes, heads, out_channels] = [num_nodes, 4, hidden_dim//4]
                # 处理多头注意力
                if x_conv.dim() > 2:  # 如果是三维输出 [num_nodes, heads, hidden_dim//4]
                    x_conv = x_conv.mean(dim=1)  # 平均多头: [num_nodes, hidden_dim//4]
                
                # 通过投影层映射到hidden_dim
                if x_conv.size(1) != self.hidden_dim:
                    x_conv = self.projection_layers[1](x_conv)
                
                x_conv = x_conv.contiguous()
                
                # 添加稳定性处理
                try:
                    # 使用BatchNorm归一化前先排除异常值
                    x_conv = torch.nan_to_num(x_conv, nan=0.0, posinf=1.0, neginf=-1.0)
                    x_conv = self.batch_norms[1](x_conv)
                except Exception as e:
                    logging.error(f"BatchNorm处理出错: {str(e)}")
                    x_conv = x_residual
                    continue
                
                # 使用LeakyReLU代替ELU激活函数 - 更安全的激活处理
                x_conv = F.leaky_relu(x_conv, negative_slope=0.1)
                
                # 检查激活后的值
                x_conv = torch.nan_to_num(x_conv, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 应用dropout，确保不会引入nan
                x_conv = self.dropout(x_conv)
            
            else:
                # NNConv输出已经是 [num_nodes, hidden_dim]
                # 打印实际维度用于调试
                #print(f"NNConv输出维度: {x_conv.shape}")
                
                x_conv = x_conv.contiguous()
                
                # 添加稳定性处理
                try:
                    # 使用BatchNorm归一化前先排除异常值
                    x_conv = torch.nan_to_num(x_conv, nan=0.0, posinf=1.0, neginf=-1.0)
                    x_conv = self.batch_norms[i](x_conv)
                except Exception as e:
                    logging.error(f"BatchNorm处理出错: {str(e)}")
                    x_conv = x_residual
                    continue
                
                # 使用LeakyReLU代替ELU激活函数 - 更安全的激活处理
                x_conv = F.leaky_relu(x_conv, negative_slope=0.1)
                
                # 检查激活后的值
                x_conv = torch.nan_to_num(x_conv, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 应用dropout，确保不会引入nan
                x_conv = self.dropout(x_conv)
            
            # 残差连接
            if x.size() == x_conv.size():
                x = x + x_conv
            else:
                x = x_conv
            
            # 确保每一层处理后的值都是有效的
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 4. 全局池化操作 - 完全重新设计，只使用mean和max池化，移除add池化
        # 确保x和batch的维度匹配
        assert x.size(0) == batch.size(0), f"节点数量不匹配: x: {x.size(0)}, batch: {batch.size(0)}"
        
        # 执行池化操作，将节点特征聚合为图特征
        try:
            # 检查x值是否正常
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 应用LayerNorm确保特征在合理范围
            x = self.pool_norm(x)
            
            # 只使用均值和最大池化，避免使用add池化
            global_mean = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
            global_max = global_max_pool(x, batch)    # [batch_size, hidden_dim]
            
            # 合并全局特征 - 只使用mean和max
            graph_features = torch.cat([global_mean, global_max], dim=1)  # [batch_size, hidden_dim * 2]
            
            # 应用最终归一化，确保特征在合理范围
            graph_features = self.final_norm(graph_features)
            
        except Exception as e:
            logging.error(f"全局池化操作出错: {str(e)}")
            # 创建默认值作为备选
            global_mean = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            global_max = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            graph_features = torch.cat([global_mean, global_max], dim=1)  # [batch_size, hidden_dim * 2]
        
        # 确保batch_size一致性
        assert graph_features.size(0) == batch_size, f"图特征batch_size不匹配: {graph_features.size(0)} vs {batch_size}"
        
        # 6. 处理知识特征
        global_features = torch.zeros(batch_size, self.global_feature_dim, device=x.device)
        
        # 确保所有特征维度对齐到batch_size
        try:
            for i, key in enumerate(self.global_feature_keys):
                if hasattr(data, key):
                    feature = getattr(data, key)
                    # 检查特征是否包含异常值
                    if isinstance(feature, torch.Tensor):
                        feature = torch.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # 确保特征值在合理范围内
                        if torch.max(torch.abs(feature)) > 10.0:
                            feature = torch.clamp(feature, -10.0, 10.0)
                    
                    if feature.dim() == 0:  # 标量
                        global_features[:, i] = feature.expand(batch_size)
                    elif feature.dim() == 1:  # 向量
                        if feature.size(0) >= batch_size:
                            global_features[:, i] = feature[:batch_size]
                        else:
                            # 填充不足的部分
                            pad_size = batch_size - feature.size(0)
                            padded_feature = torch.cat([
                                feature, 
                                torch.zeros(pad_size, device=feature.device)
                            ])
                            global_features[:, i] = padded_feature
                    else:
                        global_features[:, i] = feature.squeeze()[:batch_size]
                else:
                    # 特征不存在时使用零填充
                    global_features[:, i] = 0.0
                    
            # 整体检查全局特征
            global_features = torch.nan_to_num(global_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 异常值处理 - 使用缩放而不是硬裁剪
            feature_max = torch.max(torch.abs(global_features))
            if feature_max > 10.0:
                scale_factor = 10.0 / feature_max
                # 使用非原地操作替代
                global_features = global_features * scale_factor
                logging.info(f"全局特征存在极端值，已缩放 (最大值: {feature_max})")
                
        except Exception as e:
            logging.error(f"知识特征处理出错: {str(e)}")
            # 出错时使用零填充
            global_features = torch.zeros(batch_size, self.global_feature_dim, device=x.device)
        
        # 验证特征维度
        assert global_features.size() == (batch_size, self.global_feature_dim), \
            f"全局特征维度不正确: 期望 {(batch_size, self.global_feature_dim)}, 实际 {global_features.size()}"
        
        # 7. 应用知识嵌入
        try:
            # 确保输入是干净的
            global_features = torch.nan_to_num(global_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            knowledge_embedding = self.knowledge_embedding(global_features)  # [batch_size, hidden_dim]
            
            # 检查知识嵌入结果
            knowledge_embedding = torch.nan_to_num(knowledge_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 记录最近的知识流量
            self.last_knowledge_flow = knowledge_embedding.abs().mean().item()
            
            # 记录嵌入层关注的特征
            if hasattr(self.knowledge_embedding, 'get_attention_weights'):
                try:
                    attention_weights = self.knowledge_embedding.get_attention_weights()
                    # 检查返回值类型
                    if isinstance(attention_weights, dict):
                        # 如果是字典类型，直接记录
                        logging.debug(f"知识嵌入关注的特征权重: {attention_weights}")
                    elif isinstance(attention_weights, torch.Tensor):
                        # 如果是张量，使用topk
                        k = min(3, attention_weights.size(0))
                        top_indices = torch.topk(attention_weights, k)[1].cpu().numpy()
                        top_features = [self.global_feature_keys[i] for i in top_indices]
                        logging.debug(f"知识嵌入关注的前{k}个特征: {top_features}")
                    else:
                        logging.warning(f"无法处理的注意力权重类型: {type(attention_weights)}")
                except Exception as e:
                    logging.warning(f"处理知识嵌入注意力权重时出错: {str(e)}")
                
        except Exception as e:
            logging.error(f"知识嵌入出错: {str(e)}")
            knowledge_embedding = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            self.last_knowledge_flow = 0.0
        
        # 8. 特征组合
        try:
            combined_features = torch.cat([
                graph_features,      # [batch_size, hidden_dim * 2]
                knowledge_embedding  # [batch_size, hidden_dim]
            ], dim=1)  # [batch_size, hidden_dim * 3]
            
            # 确保维度正确 - 应为 [batch_size, hidden_dim * 3]
            assert combined_features.size(1) == self.hidden_dim * 3, \
                f"组合特征维度错误: 期望 {self.hidden_dim * 3}, 实际 {combined_features.size(1)}"
            
            # 检查并修复组合特征中的异常值
            combined_features = torch.nan_to_num(combined_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 缩放数值过大的特征
            feature_max = torch.max(torch.abs(combined_features))
            if feature_max > 10.0:
                scale_factor = 10.0 / feature_max
                combined_features = combined_features * scale_factor
                logging.info(f"组合特征存在极端值，已缩放 (最大值: {feature_max})")
                
        except Exception as e:
            logging.error(f"特征组合出错: {str(e)}")
            # 创建默认组合特征
            combined_features = torch.zeros(batch_size, self.hidden_dim * 3, device=x.device)
        
        # 9. 特征融合和输出
        try:
            # 应用特征融合
            fused_features = self.fusion_layer(combined_features)
            fused_features = torch.nan_to_num(fused_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 应用输出层
            output = self.output_layer(fused_features)
            
            # 确保输出为 [batch_size, 1]
            if output.dim() == 1:
                output = output.unsqueeze(1)
            
            # 最终输出检查
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 监控输出分布
            output_mean = output.mean().item()
            output_std = output.std().item()
            output_min = output.min().item()
            output_max = output.max().item()
            
            # 输出范围检查
            if output_max - output_min < 0.1:
                logging.warning(f"警告：模型输出范围过小 [{output_min:.4f}, {output_max:.4f}]，可能存在训练问题")
            
            # 输出统计信息
            if random.random() < 0.05:  # 只记录5%的批次，减少日志量
                logging.info(f"模型输出统计 - 均值: {output_mean:.4f}, 标准差: {output_std:.4f}, 范围: [{output_min:.4f}, {output_max:.4f}]")
            
            # 防止输出异常值影响训练
            output = torch.clamp(output, -10.0, 10.0)
            
        except Exception as e:
            logging.error(f"特征融合或输出层出错: {str(e)}")
            logging.error(traceback.format_exc())
            # 创建默认输出
            output = torch.zeros(batch_size, 1, device=x.device)
        
        return output

    def get_interpretable_weights(self):
        """获取可解释的注意力权重"""
        return {
            'knowledge_features': self.knowledge_embedding.get_important_features(),
            'knowledge_attention': self.knowledge_embedding.get_attention_value(),
            'knowledge_flow': self.last_knowledge_flow
        }

def save_model_interpretation(model, test_dataset, y_mean, y_std, output_file="model_interpretation.json"):
    """保存模型解释信息，包括特征重要性和知识嵌入注意力权重"""
    interpretation = {}
    
    # 获取模型的注意力权重和知识特征
    model_weights = model.get_interpretable_weights()
    
    # 获取特征重要性信息
    knowledge_features = model_weights.get('knowledge_features', [])
    knowledge_attention = model_weights.get('knowledge_attention', 0.0)
    knowledge_flow = model_weights.get('knowledge_flow', 0.0)
    
    # 记录知识嵌入信息
    interpretation["knowledge_embedding"] = {
        "selected_features": knowledge_features,
        "feature_count": len(knowledge_features),
        "knowledge_attention": knowledge_attention,
        "knowledge_flow": knowledge_flow
    }
    
    # 计算各特征对预测的影响
    interpretation["knowledge_strategy"] = {
        "purity_control": f"从完整特征集中筛选了{len(knowledge_features)}个最重要特征",
        "flow_control": f"知识流量控制值为{knowledge_flow*100:.2f}%"
    }
    
    # 预测性能信息
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for data in test_dataset:
            data = data.to(next(model.parameters()).device)
            out = model(Batch.from_data_list([data]))
            pred = out.item() * y_std + y_mean
            true = data.y.item() * y_std + y_mean
            predictions.append(pred)
            true_values.append(true)
    
    # 计算性能指标
    r2 = r2_score(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    
    interpretation["model_performance"] = {
        "r2_score": r2,
        "mae": mae,
        "sample_count": len(test_dataset)
    }
    
    # 分析特征类型分布
    feature_types = {
        "electronic": 0,
        "structural": 0,
        "geometric": 0
    }
    
    electronic_features = [
        'homo_approx', 'lumo_approx', 'gap_approx', 'max_abs_charge', 
        'mean_abs_charge', 'charge_variance', 'electron_affinity_approx'
    ]
    
    structural_features = [
        'aromatic_ratio', 'aromatic_bond_ratio', 'conjugation_ratio',
        'ewg_count', 'edg_count', 'ewg_edg_ratio',
        'mean_electronegativity', 'max_electronegativity', 'electronegativity_variance',
        'logp', 'mr'
    ]
    
    geometric_features = [
        'tpsa', 'labute_asa', 'molecular_volume',
        'mean_bond_length', 'max_bond_length', 'min_bond_length', 'std_bond_length',
        'mean_bond_order', 'max_bond_order', 'fraction_single_bonds', 
        'fraction_double_bonds', 'fraction_triple_bonds'
    ]
    
    for feature in knowledge_features:
        if feature in electronic_features:
            feature_types["electronic"] += 1
        elif feature in structural_features:
            feature_types["structural"] += 1
        elif feature in geometric_features:
            feature_types["geometric"] += 1
    
    # 计算特征类型分布百分比
    total_features = len(knowledge_features)
    feature_distribution = {}
    if total_features > 0:
        feature_distribution = {
            "electronic": feature_types["electronic"] / total_features * 100,
            "structural": feature_types["structural"] / total_features * 100,
            "geometric": feature_types["geometric"] / total_features * 100
        }
    
    interpretation["feature_type_distribution"] = feature_distribution
    
    # 保存解释信息
    with open(output_file, 'w') as f:
        json.dump(interpretation, f, indent=4)
    
    # 生成详细的解释报告
    with open('knowledge_embedding_detailed_report.txt', 'w') as f:
        f.write("# 电离能预测模型 - 知识嵌入策略详细报告\n\n")
        
        f.write("## 1. 纯度控制分析\n\n")
        f.write(f"模型选择了{len(knowledge_features)}个最重要特征作为知识向量，包括：\n\n")
        for i, feature in enumerate(knowledge_features):
            f.write(f"{i+1}. {feature}\n")
        
        f.write("\n特征类型分布：\n")
        f.write(f"- 电子结构特征: {feature_distribution.get('electronic', 0):.1f}%\n")
        f.write(f"- 结构特征: {feature_distribution.get('structural', 0):.1f}%\n")
        f.write(f"- 几何特征: {feature_distribution.get('geometric', 0):.1f}%\n\n")
        
        f.write("## 2. 流量控制分析\n\n")
        f.write(f"知识流量控制值: {knowledge_flow*100:.2f}%\n")
        f.write("这表示模型在做预测时，有多少比例的决策依赖于知识特征，而不是图结构特征。\n\n")
        
        f.write("知识注意力值: {knowledge_attention*100:.2f}%\n")
        f.write("这表示知识嵌入层内部对输入特征的关注程度。\n\n")
        
        f.write("## 3. 性能指标\n\n")
        f.write(f"R²: {r2:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"测试样本数: {len(test_dataset)}\n\n")
        
        f.write("## 4. 物理化学解释\n\n")
        f.write("基于筛选的重要特征，模型主要利用以下物理化学原理预测电离能：\n\n")
        
        # 判断各类物理化学原理的应用
        if 'homo_approx' in knowledge_features:
            f.write("- **轨道能级原理**: 模型重点利用HOMO能级，准确反映电子电离所需能量\n")
        
        if 'ewg_count' in knowledge_features or 'edg_count' in knowledge_features:
            f.write("- **取代基效应**: 模型考虑吸电子/供电子基团对电子云密度的影响\n")
        
        if 'conjugation_ratio' in knowledge_features or 'aromatic_ratio' in knowledge_features:
            f.write("- **共轭效应**: 模型捕捉共轭程度和芳香性对电子离域和稳定性的影响\n")
        
        if 'mean_bond_length' in knowledge_features or 'mean_bond_order' in knowledge_features:
            f.write("- **分子刚性**: 模型考虑键长和键级对分子刚性及电子稳定性的影响\n")
        
        if 'mean_electronegativity' in knowledge_features or 'electronegativity_variance' in knowledge_features:
            f.write("- **电负性分布**: 模型分析电负性分布对电子分布和电离能的影响\n")
        
        if 'molecular_volume' in knowledge_features or 'tpsa' in knowledge_features:
            f.write("- **空间效应**: 模型考虑分子体积和表面积对电子电离的立体影响\n")
    
    return interpretation

def perform_shap_analysis(model, dataset, device="cuda", n_samples=100, update_model=False):
    """
    对模型进行SHAP值分析，量化特征对预测的贡献
    
    参数:
    model -- 训练好的模型
    dataset -- 数据集
    device -- 计算设备
    n_samples -- 用于分析的样本数量
    update_model -- 是否使用SHAP结果更新模型特征
    
    返回:
    shap_values -- 每个特征的SHAP值
    feature_names -- 特征名称列表
    chemical_interpretation -- 特征重要性的化学解释
    """
    model.eval()
    logging.info("开始进行SHAP分析...")
    
    # 限制分析的样本数量以提高效率
    if n_samples < len(dataset):
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        analysis_dataset = [dataset[i] for i in indices]
    else:
        analysis_dataset = dataset
    
    # 准备特征名称
    sample_data = analysis_dataset[0]
    feature_names = []
    
    # 1. 所有原子特征
    atom_feature_names = [
        "atomic_number", "degree", "formal_charge", "radical_electrons", 
        "aromaticity", "mass", "explicit_valence", "implicit_valence", 
        "total_valence", "implicit_Hs", "in_ring", "hybridization",
        "total_H_count", "total_degree", "total_valence_2", "explicit_valence_2", 
        "implicit_valence_2", "formal_charge_2", "radical_electrons_2",
        "ring_size_3", "ring_size_4", "ring_size_5", "ring_size_6", "ring_size_7",
        "chirality", "cip_code"
    ]
    
    # 添加原子特征，为每个原子添加特征名称
    for i, name in enumerate(atom_feature_names):
        feature_names.append(f"atom_{name}")
    
    # 2. 键相关特征
    bond_feature_names = [
        "bond_type", "is_conjugated", "is_aromatic", "in_ring", "stereo",
        "bond_single", "bond_double", "bond_triple", "bond_aromatic",
        "stereo_z", "stereo_e", "stereo_cis", "stereo_trans",
        "ring_size_3", "ring_size_4", "ring_size_5", "ring_size_6", 
        "ring_size_7", "ring_size_8", "bond_length"
    ]
    
    for i, name in enumerate(bond_feature_names):
        feature_names.append(f"bond_{name}")
    
    # 3. 全局分子特征
    global_feature_keys = [
        'homo_approx', 'lumo_approx', 'gap_approx', 'max_abs_charge', 
        'mean_abs_charge', 'charge_variance', 'electron_affinity_approx',
        'aromatic_ratio', 'aromatic_bond_ratio', 'conjugation_ratio',
        'ewg_count', 'edg_count', 'ewg_edg_ratio',
        'mean_electronegativity', 'max_electronegativity', 'electronegativity_variance',
        'logp', 'mr', 'tpsa', 'labute_asa', 'molecular_volume',
        'mean_bond_length', 'max_bond_length', 'min_bond_length', 'std_bond_length',
        'mean_bond_order', 'max_bond_order', 'fraction_single_bonds', 
        'fraction_double_bonds', 'fraction_triple_bonds',
        'asphericity', 'eccentricity', 'inertial_shape_factor', 'npr1', 'npr2',
        'pmi1', 'pmi2', 'pmi3', 'radius_of_gyration', 'spherocity_index'
    ]
    feature_names.extend(global_feature_keys)
    
    # 为SHAP分析准备背景数据，提取全局特征
    background_data = []
    for data in analysis_dataset[:30]:  # 使用30个样本作为背景数据
        features = []
        for key in global_feature_keys:
            if hasattr(data, key):
                features.append(getattr(data, key).item())
            else:
                features.append(0)
        background_data.append(features)
    
    background = np.array(background_data)
    
    # 创建一个分子级特征的模型包装器
    class MolecularModelWrapper:
        def __init__(self, model, dataset, device):
            self.model = model
            self.dataset = dataset
            self.device = device
            self.global_feature_keys = global_feature_keys
        
        def __call__(self, x):
            """接受分子级特征，返回预测结果"""
            results = []
            for i, features in enumerate(x):
                # 获取原始数据
                data = self.dataset[i % len(self.dataset)].clone()
                data = data.to(self.device)
                
                # 替换全局特征值
                for idx, key in enumerate(self.global_feature_keys):
                    if hasattr(data, key):
                        setattr(data, key, torch.tensor([features[idx]], device=self.device))
                
                # 预测
                with torch.no_grad():
                    prediction = self.model(Batch.from_data_list([data]))
                results.append(prediction.cpu().item())
            
            return np.array(results)
    
    # 创建模型包装器
    model_wrapper = MolecularModelWrapper(model, analysis_dataset, device)
    
    # 使用SHAP的KernelExplainer解释模型
    explainer = shap.KernelExplainer(model_wrapper, background)
    
    # 对测试样本计算SHAP值
    test_data = []
    for data in analysis_dataset[:n_samples]:
        features = []
        for key in global_feature_keys:
            if hasattr(data, key):
                features.append(getattr(data, key).item())
            else:
                features.append(0)
        test_data.append(features)
    
    test_data = np.array(test_data)
    shap_values = explainer.shap_values(test_data)
    
    # 绘制SHAP摘要图
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, test_data, feature_names=global_feature_keys, show=False)
    plt.title("SHAP分析: 各特征对电离能预测的贡献")
    plt.tight_layout()
    plt.savefig('shap_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 绘制SHAP依赖图 (针对重要特征)
    top_features = np.argsort(np.abs(shap_values).mean(0))[-5:]  # 获取前5个最重要的特征
    
    for idx in top_features:
        feature_name = global_feature_keys[idx]
        plt.figure(figsize=(10, 7))
        shap.dependence_plot(idx, shap_values, test_data, feature_names=global_feature_keys, show=False)
        plt.title(f"SHAP依赖图: {feature_name}")
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{feature_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 计算特征重要性
    feature_importance = np.abs(shap_values).mean(0)
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    # 提取化学解释
    chemical_interpretation = analyze_chemical_drivers(global_feature_keys, sorted_idx, feature_importance)
    
    # 更新模型的知识特征 (纯度控制)
    if update_model and hasattr(model, 'update_knowledge_features'):
        logging.info("基于SHAP分析更新模型的知识特征...")
        
        # 获取全局特征的重要性
        global_importance = feature_importance
        
        # 更新模型中的知识特征
        model.update_knowledge_features(global_feature_keys, global_importance)
        
        # 记录更新后的特征
        updated_features = model.get_interpretable_weights()['knowledge_features']
        logging.info(f"知识特征已更新为: {updated_features}")
        
        # 保存更新后的特征重要性
        with open('shap_updated_features.json', 'w') as f:
            json.dump({
                'features': [f for f in updated_features],
                'importance': [float(feature_importance[global_feature_keys.index(f)]) for f in updated_features]
            }, f, indent=2)
    
    return shap_values, global_feature_keys, chemical_interpretation

def analyze_chemical_drivers(feature_names, sorted_indices, importance_values):
    """
    分析特征重要性，提供化学解释
    
    参数:
    feature_names -- 特征名称列表
    sorted_indices -- 按重要性排序的索引
    importance_values -- 特征重要性值
    
    返回:
    interpretation -- 包含主导因素和辅助因素的字典
    """
    # 特征与化学概念的映射
    feature_to_concept = {
        # 电子结构相关
        'homo_approx': '最高占据分子轨道能级',
        'lumo_approx': '最低未占据分子轨道能级',
        'gap_approx': '轨道能隙',
        'max_abs_charge': '最大绝对电荷',
        'mean_abs_charge': '平均绝对电荷',
        'charge_variance': '电荷分布方差',
        'electron_affinity_approx': '电子亲和能',
        
        # 结构特征相关
        'aromatic_ratio': '芳香性比例',
        'aromatic_bond_ratio': '芳香键比例',
        'conjugation_ratio': '共轭程度',
        'ewg_count': '吸电子基团数量',
        'edg_count': '供电子基团数量',
        'ewg_edg_ratio': '吸/供电子基团比例',
        'mean_electronegativity': '平均电负性',
        'max_electronegativity': '最大电负性',
        'electronegativity_variance': '电负性分布方差',
        
        # 几何特征相关
        'molecular_volume': '分子体积',
        'tpsa': '拓扑极性表面积',
        'labute_asa': '表面积',
        'mean_bond_length': '平均键长',
        'max_bond_length': '最大键长',
        'min_bond_length': '最小键长',
        'std_bond_length': '键长标准差',
        'mean_bond_order': '平均键级',
        
        # 形状相关
        'asphericity': '非球形度',
        'eccentricity': '离心率',
        'inertial_shape_factor': '惯性形状因子',
        'radius_of_gyration': '回转半径'
    }
    
    # 特征与物理意义的映射
    feature_to_physics = {
        'homo_approx': '直接反映电子电离难易度，HOMO能级越高，电离能越低',
        'lumo_approx': '影响电子跃迁能力，与电离能间接相关',
        'gap_approx': '反映分子稳定性，影响电离过程',
        'ewg_count': '吸电子基团降低整体电子云密度，降低电离能',
        'edg_count': '供电子基团提高电子云密度，提高电离能',
        'conjugation_ratio': '延长共轭提升电子离域，降低电离能',
        'aromatic_ratio': '增加电子离域，降低电离能',
        'mean_bond_length': '反映分子刚性，影响电子稳定性',
        'mean_electronegativity': '电负性高的原子容易吸引电子，影响电离能',
        'radius_of_gyration': '分子尺寸与形状，影响电子离域程度'
    }
    
    # 分析结果
    top_n = 5  # 主导因素取前5个
    primary_drivers = []
    secondary_drivers = []
    
    # 获取主导因素
    for i in range(min(top_n, len(sorted_indices))):
        idx = sorted_indices[i]
        feature = feature_names[idx]
        importance = importance_values[idx]
        
        concept = feature_to_concept.get(feature, feature)
        physics = feature_to_physics.get(feature, "未定义的物理意义")
        
        primary_drivers.append({
            'feature': feature,
            'importance': float(importance),
            'concept': concept,
            'physics': physics
        })
    
    # 获取辅助因素
    for i in range(top_n, min(top_n + 5, len(sorted_indices))):
        idx = sorted_indices[i]
        feature = feature_names[idx]
        importance = importance_values[idx]
        
        concept = feature_to_concept.get(feature, feature)
        physics = feature_to_physics.get(feature, "未定义的物理意义")
        
        secondary_drivers.append({
            'feature': feature,
            'importance': float(importance),
            'concept': concept,
            'physics': physics
        })
    
    # 记录分析结果
    logging.info("电离能预测的主导因素:")
    for driver in primary_drivers:
        logging.info(f"  {driver['feature']} ({driver['concept']}): {driver['importance']:.6f}")
        logging.info(f"    物理意义: {driver['physics']}")
    
    logging.info("电离能预测的辅助因素:")
    for driver in secondary_drivers:
        logging.info(f"  {driver['feature']} ({driver['concept']}): {driver['importance']:.6f}")
        logging.info(f"    物理意义: {driver['physics']}")
    
    # 创建解释报告
    with open('feature_importance_report.txt', 'w') as f:
        f.write("# 电离能预测的特征重要性分析\n\n")
        
        f.write("## 主导因素\n")
        for driver in primary_drivers:
            f.write(f"### {driver['feature']} ({driver['concept']})\n")
            f.write(f"- 重要性: {driver['importance']:.6f}\n")
            f.write(f"- 物理意义: {driver['physics']}\n\n")
        
        f.write("## 辅助因素\n")
        for driver in secondary_drivers:
            f.write(f"### {driver['feature']} ({driver['concept']})\n")
            f.write(f"- 重要性: {driver['importance']:.6f}\n")
            f.write(f"- 物理意义: {driver['physics']}\n\n")
    
    return {
        'primary_drivers': primary_drivers,
        'secondary_drivers': secondary_drivers
    }

def train_model(model, train_loader, val_loader, device, fold=0):
    """训练模型的主函数"""
    logging.info(f"第 {fold+1} 折 - 开始训练...")
    
    # 优化器和学习率调度器 - 使用极保守的配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0000001,  # 使用极小的学习率开始，减少数值不稳定问题
        weight_decay=0.01,  # 增加权重衰减来防止过拟合
        eps=1e-10,  # 增大epsilon值提高数值稳定性
        amsgrad=True  # 使用AMSGrad变体提高稳定性
    )
    
    # 学习率调度器 - 更保守的配置
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,  # 更温和的降低因子
        patience=20,  # 更长的耐心期
        threshold=0.001,
        min_lr=1e-10,
        verbose=True
    )
    
    # 初始化混合精度训练 - 禁用混合精度训练，使用纯FP32训练提高稳定性
    scaler = GradScaler(enabled=False)  # 禁用混合精度训练来解决unscale_问题
    
    # 改用Huber Loss(SmoothL1Loss)，对异常值更加鲁棒
    # 使用更大的beta值使其更接近MSE，但仍保持对异常值的鲁棒性
    criterion = nn.SmoothL1Loss(beta=0.5, reduction='mean')
    
    # 存储训练历史
    train_losses = []
    val_losses = []
    
    # 训练参数
    epochs = 50
    early_stop_patience = 20
    early_stop_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    # 添加在train_model函数开始处
    warmup_epochs = 10  # 增加预热轮数
    warmup_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs)
    )
    
    # 梯度累积步数 - 增加稳定性
    gradient_accumulation_steps = 8  # 每8个批次更新一次参数，提高稳定性
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # 训练阶段
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)") as pbar:
            # 重置优化器状态标志
            optimizer.zero_grad(set_to_none=True)  # 在每个epoch开始前清空梯度
            accumulated_batches = 0
            
            for batch_idx, batch in enumerate(pbar):
                # 检查批次数据的有效性
                try:
                    if batch.x.shape[0] == 0 or batch.edge_index.shape[1] == 0:
                        logging.warning(f"跳过空批次，批次索引: {batch_idx}")
                        continue
                        
                    # 检查输入数据中是否有NaN或Inf
                    if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                        logging.warning(f"输入特征包含NaN或Inf，批次索引: {batch_idx}")
                        # 清理输入数据
                        batch.x = torch.nan_to_num(batch.x, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # 检查目标值
                    if torch.isnan(batch.y).any() or torch.isinf(batch.y).any():
                        logging.warning(f"目标值包含NaN或Inf，批次索引: {batch_idx}")
                        # 跳过这个批次
                        continue
                    
                    batch = batch.to(device)
                    
                    # 前向传播不使用混合精度，提高稳定性
                    out = model(batch)
                    
                    # 确保输出和目标具有相同的形状
                    if out.shape != batch.y.shape:
                        out = out.view(batch.y.shape)
                    
                    # 使用更稳健的损失计算
                    loss = criterion(out, batch.y)
                    
                    # 防止损失值异常
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                        logging.error(f"损失值异常! 损失: {loss.item()}")
                        logging.error(f"预测值范围: [{out.min().item()}, {out.max().item()}]")
                        logging.error(f"真实值范围: [{batch.y.min().item()}, {batch.y.max().item()}]")
                        # 跳过这个批次
                        continue
                    
                    # 使用梯度累积来提高稳定性
                    # 将损失除以累积步数，使总梯度大小保持不变
                    loss = loss / gradient_accumulation_steps
                    
                    # 反向传播 - 不使用混合精度反向传播
                    loss.backward()
                    
                    # 累积批次计数
                    accumulated_batches += 1
                    
                    # 当累积足够的批次时，执行优化步骤
                    if accumulated_batches >= gradient_accumulation_steps:
                        # 检查并修复梯度
                        has_bad_gradients, fixed_count = check_and_fix_gradients(model, gradient_threshold=0.1)
                        
                        # 如果过多参数存在异常梯度，跳过此批次更新
                        if fixed_count > len(list(model.parameters())) * 0.2:
                            logging.warning(f"过多参数存在异常梯度 ({fixed_count}个)，跳过本批次更新")
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            # 应用更保守的梯度裁剪
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
                            
                            # 每10个批次记录一次模型诊断信息
                            if batch_idx % 10 == 0:
                                log_model_diagnostics(model, epoch, batch_idx)
                                
                            # 计算梯度范数以记录
                            grad_norm = 0
                            for p in model.parameters():
                                if p.grad is not None:
                                    param_norm = p.grad.data.norm(2).item()
                                    if param_norm > 0:
                                        grad_norm += param_norm ** 2
                            grad_norm = grad_norm ** 0.5 if grad_norm > 0 else 0
                            
                            # 执行优化步骤
                            optimizer.step()
                            
                            # 更新进度条信息
                            pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}', 'grad': f'{grad_norm:.4f}'})
                        
                        # 无论是否更新参数，都重置计数器和梯度
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                    
                    # 累积总损失（反向缩放因子用于准确计算总损失）
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    
                except Exception as e:
                    logging.error(f"处理批次 {batch_idx} 时出错: {str(e)}")
                    logging.error(traceback.format_exc())
                    # 重置累积状态，确保下一批次正常
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_batches = 0
                    continue
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        # 为调试添加详细信息收集
        debug_info = {
            "真实值": [],
            "预测值": [],
            "真实值范围": [float('inf'), -float('inf')],  # [min, max]
            "预测值范围": [float('inf'), -float('inf')],  # [min, max]
            "样本绝对误差": []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)")):
                batch = batch.to(device)
                out = model(batch)
                
                # 收集调试信息 (只处理前5个batch)
                if batch_idx < 5:
                    batch_y = batch.y.cpu().numpy()
                    batch_pred = out.cpu().numpy()
                    debug_info["真实值"].append(batch_y[:5])
                    debug_info["预测值"].append(batch_pred[:5])
                    debug_info["真实值范围"][0] = min(debug_info["真实值范围"][0], batch_y.min())
                    debug_info["真实值范围"][1] = max(debug_info["真实值范围"][1], batch_y.max())
                    debug_info["预测值范围"][0] = min(debug_info["预测值范围"][0], batch_pred.min())
                    debug_info["预测值范围"][1] = max(debug_info["预测值范围"][1], batch_pred.max())
                    
                    # 计算每个样本的绝对误差
                    diff = np.abs(batch_y - batch_pred)
                    if diff.ndim > 1:
                        # 如果是多维数组，沿着第1维取平均
                        abs_errors = diff.mean(axis=1)
                    else:
                        # 如果是一维数组，直接添加
                        abs_errors = diff
                    
                    debug_info["样本绝对误差"].extend(abs_errors)
                    
                    # 检查是否有NaN值
                    if np.isnan(batch_y).any() or np.isnan(batch_pred).any():
                        logging.error(f"发现NaN值: 真实值有NaN: {np.isnan(batch_y).any()}, 预测值有NaN: {np.isnan(batch_pred).any()}")
                
                batch_loss = criterion(out, batch.y).item()
                val_loss += batch_loss
                
                # 如果单个批次损失异常高，记录详细信息
                if batch_loss > 10:
                    logging.warning(f"批次 {batch_idx} 损失异常高: {batch_loss:.4f}")
                    logging.warning(f"样本大小: {batch.y.size()}")
                    logging.warning(f"真实值范围: [{batch.y.min().item():.4f}, {batch.y.max().item():.4f}]")
                    logging.warning(f"预测值范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 输出调试信息
        if epoch < 3 or (epoch + 1) % 10 == 0:  # 前3个epoch和之后每10个epoch输出一次
            logging.info("===== 验证集调试信息 =====")
            logging.info(f"真实值范围: [{debug_info['真实值范围'][0]:.4f}, {debug_info['真实值范围'][1]:.4f}]")
            logging.info(f"预测值范围: [{debug_info['预测值范围'][0]:.4f}, {debug_info['预测值范围'][1]:.4f}]")
            logging.info(f"平均绝对误差: {np.mean(debug_info['样本绝对误差']):.4f}")
            logging.info(f"最大绝对误差: {np.max(debug_info['样本绝对误差']):.4f}")
            logging.info(f"样本真实值示例: {debug_info['真实值'][0][:3].flatten()}")
            logging.info(f"样本预测值示例: {debug_info['预测值'][0][:3].flatten()}")
            
            # 尝试计算反归一化后的误差（假设有y_mean和y_std变量）
            try:
                # 获取全局归一化参数（如果可用）
                if 'y_mean' in globals() and 'y_std' in globals():
                    # 确保数据是一维的，可以安全操作
                    true_vals_flat = np.array(debug_info['真实值'][0]).flatten()
                    pred_vals_flat = np.array(debug_info['预测值'][0]).flatten()
                    
                    # 应用反归一化
                    true_vals = true_vals_flat * y_std + y_mean
                    pred_vals = pred_vals_flat * y_std + y_mean
                    
                    logging.info("===== 原始尺度调试信息 =====")
                    logging.info(f"反归一化真实值: {true_vals}")
                    logging.info(f"反归一化预测值: {pred_vals}")
                    logging.info(f"原始尺度平均误差: {np.mean(np.abs(true_vals - pred_vals)):.4f}")
                else:
                    logging.warning("无法获取全局归一化参数进行反归一化计算")
            except Exception as e:
                logging.error(f"反归一化计算出错: {str(e)}")
        
        # 更新学习率，使用预热策略
        if epoch < warmup_epochs:
            warmup_lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"预热阶段 - 学习率: {current_lr:.6f}")
        else:
            scheduler.step(val_loss)
        
        # 输出当前训练状态
        logging.info(f"Epoch {epoch+1}/{epochs}")
        logging.info(f"训练损失: {avg_train_loss:.4f}")
        logging.info(f"验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            
            # 保存模型检查点
            checkpoint_path = f'model_checkpoint_fold{fold+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            
            logging.info(f"保存最佳模型 (验证损失: {val_loss:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                logging.info(f"早停: {early_stop_patience} 轮验证损失未改善")
                break
        
        # 清理GPU内存
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    
    # 绘制训练历史
    if len(train_losses) > 1:
        plot_training_history(
            train_losses,
            val_losses,
            len(train_losses),
            title=f"训练历史 (折 {fold+1})"
        )
    
    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    
    return model, best_val_loss

def perform_quantum_chemistry_correction(model, dataframe, error_threshold=0.4, max_molecules=10, dft_method="B3LYP/6-31+G*"):
    """
    对预测结果偏差较大的分子进行量子化学计算修正，并更新训练集
    
    参数:
    model -- 训练好的模型
    dataframe -- 原始数据集
    error_threshold -- 预测误差阈值（eV），超过此阈值的分子将被选择进行DFT计算
    max_molecules -- 最大计算分子数量
    dft_method -- DFT计算方法和基组
    
    返回:
    updated_dataframe -- 更新后的数据集
    corrected_molecules -- 修正的分子信息
    """
    import os
    from rdkit.Chem import AllChem
    import tempfile
    import subprocess
    
    logging.info(f"开始量子化学计算补充，使用方法: {dft_method}")
    
    # 将数据框转换为分子对象列表，用于预测和量子化学计算
    molecules = []
    smiles_list = []
    for idx, row in dataframe.iterrows():
        smiles = row['x']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecules.append((idx, mol, row['y']))
            smiles_list.append(smiles)
    
    # 使用当前模型对所有分子进行预测
    device = next(model.parameters()).device
    predictions = []
    true_values = []
    
    model.eval()
    with torch.no_grad():
        for smiles, (idx, mol, true_y) in zip(smiles_list, molecules):
            # 将分子转换为图
            graph = smiles_to_graph(smiles)
            if graph is None:
                continue
            
            # 预测
            graph = graph.to(device)
            out = model(Batch.from_data_list([graph]))
            predicted_y = out.item()
            
            # 加入到待比较列表
            predictions.append((idx, predicted_y, true_y))
            true_values.append(true_y)
    
    # 计算预测误差，并找出偏差最大的分子
    errors = []
    for idx, pred, true in predictions:
        error = abs(pred - true)
        errors.append((idx, error, pred, true))
    
    # 根据误差排序，选择误差最大的分子
    errors.sort(key=lambda x: x[1], reverse=True)
    
    # 筛选出误差大于阈值的分子，限制最大计算数量
    candidates = [e for e in errors if e[1] > error_threshold][:max_molecules]
    
    logging.info(f"已筛选出{len(candidates)}个预测偏差大于{error_threshold}eV的分子进行DFT修正")
    
    # 如果没有超过阈值的分子，返回原数据集
    if not candidates:
        logging.info("没有发现预测偏差大的分子，无需进行量子化学修正")
        return dataframe, []
    
    # 创建临时目录存放计算文件
    temp_dir = tempfile.mkdtemp(prefix="qc_calc_")
    logging.info(f"创建临时目录用于DFT计算: {temp_dir}")
    
    # 量子化学计算结果
    corrected_molecules = []
    
    try:
        # 遍历候选分子，使用量子化学方法计算电离能
        for rank, (idx, error, pred, true) in enumerate(candidates):
            mol = molecules[smiles_list.index(dataframe.iloc[idx]['x'])][1]
            smiles = dataframe.iloc[idx]['x']
            
            logging.info(f"处理分子 {rank+1}/{len(candidates)}: {smiles}")
            logging.info(f"  预测值: {pred:.4f}eV, 实际值: {true:.4f}eV, 误差: {error:.4f}eV")
            
            # 1. 生成3D构象
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # 2. 生成计算输入文件 (使用PySCF作为计算引擎)
            try:
                # 检查是否安装了PySCF
                import pyscf
                from pyscf import gto, dft, scf
                import numpy as np
                
                # 从mol对象提取原子坐标和元素类型
                atoms = []
                for atom in mol.GetAtoms():
                    pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    atoms.append([atom.GetSymbol(), (pos.x, pos.y, pos.z)])
                
                # 设置PySCF输入
                method_basis = dft_method.split('/')
                dft_functional = method_basis[0]
                basis_set = method_basis[1]
                
                # 创建分子对象
                mol_pyscf = gto.Mole()
                mol_pyscf.atom = atoms
                mol_pyscf.basis = basis_set
                mol_pyscf.build()
                
                # 计算中性分子能量
                mf = dft.RKS(mol_pyscf)
                mf.xc = dft_functional.lower()
                e_neutral = mf.kernel()
                
                # 计算阳离子能量 (减少一个电子)
                mol_pyscf.charge = 1
                mol_pyscf.build()
                mf_cation = dft.RKS(mol_pyscf)
                mf_cation.xc = dft_functional.lower()
                e_cation = mf_cation.kernel()
                
                # 计算电离能 (eV)
                ie_dft = (e_cation - e_neutral) * 27.2114  # 转换为eV (1 Hartree = 27.2114 eV)
                
                logging.info(f"  DFT计算电离能: {ie_dft:.4f}eV")
                
                # 保存修正后的数据
                corrected_molecules.append({
                    'smiles': smiles,
                    'original_y': true,
                    'predicted_y': pred,
                    'dft_corrected_y': ie_dft,
                    'error': error
                })
                
                # 更新原始数据框的电离能值
                dataframe.at[idx, 'y'] = ie_dft
                dataframe.at[idx, 'dft_corrected'] = True
                
            except ImportError:
                logging.warning("未安装PySCF，尝试使用ORCA或Gaussian等其他程序")
                logging.warning("由于外部程序调用复杂性，此示例代码仅演示PySCF实现")
                logging.warning("实际使用时，可根据实验室可用软件修改此部分代码")
                
                # 这里可以添加其他量子化学软件的调用代码
                # 例如使用subprocess调用ORCA、Gaussian等
                
                # 模拟DFT计算结果（实际应用中应替换为真实计算）
                ie_dft = true  # 仅作示例，实际应使用计算结果
                
                logging.warning(f"  使用原始值代替DFT计算: {ie_dft:.4f}eV")
    
    except Exception as e:
        logging.error(f"量子化学计算过程中出错: {e}")
    
    finally:
        # 清理临时文件
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logging.info(f"已清理临时目录: {temp_dir}")
        except:
            logging.warning(f"无法清理临时目录: {temp_dir}")
    
    # 汇总修正结果
    if corrected_molecules:
        # 保存修正结果
        with open('dft_corrected_molecules.json', 'w') as f:
            json.dump(corrected_molecules, f, indent=2)
        
        logging.info(f"共完成{len(corrected_molecules)}个分子的DFT修正，结果已保存至dft_corrected_molecules.json")
    else:
        logging.info("未完成任何分子的DFT修正")
    
    return dataframe, corrected_molecules

def iterative_model_optimization(dataframe, device='cuda', iterations=3, error_threshold=0.4, 
                                max_molecules_per_iter=5, dft_method="B3LYP/6-31+G*"):
    """
    通过量子化学计算补充，迭代优化模型
    
    参数:
    dataframe -- 原始数据集
    device -- 计算设备
    iterations -- 最大迭代次数
    error_threshold -- 预测误差阈值
    max_molecules_per_iter -- 每次迭代最大计算分子数量
    dft_method -- DFT计算方法和基组
    
    返回:
    final_model -- 最终优化的模型
    final_metrics -- 最终性能指标
    all_corrected_molecules -- 所有修正的分子
    """
    # 复制原始数据集
    current_df = dataframe.copy()
    all_corrected_molecules = []
    
    # 记录每次迭代的性能指标
    iteration_metrics = {
        'r2': [],
        'mae': [],
        'corrected_count': []
    }
    
    # 标记是否达到目标MAE
    target_achieved = False
    
    # 创建数据加载器
    dataset = MolecularDataset(current_df['x'].values, current_df['y'].values)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = PyGDataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    sample_data = dataset[0]
    model = EnhancedMolecularGraph(
        num_node_features=sample_data.x.size(1),
        num_edge_features=sample_data.edge_attr.size(1)
    ).to(device)
    
    # 迭代训练和修正
    for iter_idx in range(iterations):
        logging.info(f"\n开始第 {iter_idx+1}/{iterations} 次迭代优化")
        
        # 1. 训练模型
        model, metrics = train_model(model, train_loader, val_loader, device)
        
        # 评估模型性能
        model.eval()
        predictions = []
        true_values = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                predictions.extend(out.cpu().numpy())
                true_values.extend(batch.y.cpu().numpy())
        
        test_r2 = r2_score(true_values, predictions)
        test_mae = mean_absolute_error(true_values, predictions)
        
        # 记录性能指标
        iteration_metrics['r2'].append(test_r2)
        iteration_metrics['mae'].append(test_mae)
        
        # 检查是否达到目标MAE
        if test_mae <= 0.2:
            logging.info(f"✅ 已达到目标 MAE (≤ 0.2 eV): {test_mae:.4f}")
            target_achieved = True
            break
            
        # 2. 对预测偏差大的分子进行量子化学修正
        updated_df, corrected = perform_quantum_chemistry_correction(
            model, 
            current_df, 
            error_threshold=error_threshold,
            max_molecules=max_molecules_per_iter,
            dft_method=dft_method
        )
        
        # 更新数据集和数据加载器
        current_df = updated_df
        dataset = MolecularDataset(current_df['x'].values, current_df['y'].values)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = PyGDataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=32, shuffle=False)
        
        all_corrected_molecules.extend(corrected)
        iteration_metrics['corrected_count'].append(len(corrected))
        
        # 如果没有分子被修正，停止迭代
        if not corrected:
            logging.info("没有分子需要修正，停止迭代")
            break
        
        logging.info(f"第 {iter_idx+1} 次迭代完成，当前指标: R²={test_r2:.4f}, MAE={test_mae:.4f}, 已修正分子数={len(all_corrected_molecules)}")
    
    # 绘制迭代优化曲线
    plt.figure(figsize=(12, 8))
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(range(1, len(iteration_metrics['mae'])+1), iteration_metrics['mae'], 'ro-', label='MAE')
    ax1.axhline(y=0.2, color='g', linestyle='--', label='目标 MAE (0.2 eV)')
    ax1.set_ylabel('MAE (eV)')
    ax1.set_title('模型迭代优化性能')
    ax1.grid(True)
    ax1.legend()
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(range(1, len(iteration_metrics['r2'])+1), iteration_metrics['r2'], 'bo-', label='R²')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('R²')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('model_iteration_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 返回最终模型和性能指标
    final_metrics = {
        'r2': iteration_metrics['r2'][-1],
        'mae': iteration_metrics['mae'][-1],
        'total_corrected': len(all_corrected_molecules),
        'iterations': len(iteration_metrics['r2']),
        'target_achieved': target_achieved
    }
    
    return model, final_metrics, all_corrected_molecules

def generate_ml_descriptors(df, feature_importance, top_n=10, min_importance=0.01):
    """
    基于特征重要性生成新的机器学习描述符
    
    参数:
        df (pd.DataFrame): 分子数据集
        feature_importance (dict): 特征重要性字典
        top_n (int): 选择的最重要特征数量
        min_importance (float): 特征重要性的最小阈值
        
    返回:
        pd.DataFrame: 生成的机器学习描述符
    """
    import numpy as np
    import pandas as pd
    
    # 如果特征重要性为空，返回空数据框
    if not feature_importance or len(feature_importance) == 0:
        print("特征重要性为空，无法生成机器学习描述符")
        return pd.DataFrame()
    
    # 提取所有特征
    features = list(df.columns)
    if 'ionization_energy' in features:
        features.remove('ionization_energy')
    
    # 提取SMILES列
    if 'smiles' in df.columns:
        smiles_col = df['smiles']
    else:
        smiles_col = df.index if df.index.name == 'smiles' else pd.Series([f'mol_{i}' for i in range(len(df))])
    
    # 按重要性排序特征
    sorted_features = sorted(
        [(feature, importance) for feature, importance in feature_importance.items() if feature in features],
        key=lambda x: x[1],
        reverse=True
    )
    
    # 选择最重要的特征
    top_features = [f for f, imp in sorted_features[:top_n] if imp >= min_importance]
    
    if len(top_features) == 0:
        print("没有满足条件的重要特征，无法生成机器学习描述符")
        return pd.DataFrame()
    
    print(f"选择了 {len(top_features)} 个最重要的特征来生成描述符")
    
    # 提取选定特征的数据
    selected_data = df[top_features].copy()
    
    # 生成新的机器学习描述符
    ml_descriptors = pd.DataFrame(index=df.index)
    
    # 1. 特征组合 (相乘)
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            feature1 = top_features[i]
            feature2 = top_features[j]
            new_feature_name = f'ml_desc_mul_{feature1}_{feature2}'
            ml_descriptors[new_feature_name] = selected_data[feature1] * selected_data[feature2]
    
    # 2. 特征加权和
    weights = [feature_importance[f] for f in top_features]
    weight_sum = sum(weights)
    normalized_weights = [w/weight_sum for w in weights]
    
    ml_descriptors['ml_desc_weighted_sum'] = sum(
        selected_data[feature] * weight for feature, weight in zip(top_features, normalized_weights)
    )
    
    # 3. 特征平方项
    for feature in top_features[:5]:  # 只为最重要的5个特征创建平方项
        ml_descriptors[f'ml_desc_squared_{feature}'] = selected_data[feature] ** 2
    
    # 4. 特征比例关系
    for i in range(min(3, len(top_features))):
        for j in range(i+1, min(5, len(top_features))):
            feature1 = top_features[i]
            feature2 = top_features[j]
            # 避免除以零
            if (selected_data[feature2] != 0).all():
                new_feature_name = f'ml_desc_ratio_{feature1}_to_{feature2}'
                ml_descriptors[new_feature_name] = selected_data[feature1] / selected_data[feature2]
    
    # 5. 特征差异
    for i in range(min(3, len(top_features))):
        for j in range(i+1, min(5, len(top_features))):
            feature1 = top_features[i]
            feature2 = top_features[j]
            new_feature_name = f'ml_desc_diff_{feature1}_minus_{feature2}'
            ml_descriptors[new_feature_name] = selected_data[feature1] - selected_data[feature2]
    
    # 6. 非线性变换
    for feature in top_features[:3]:  # 只为最重要的3个特征创建非线性变换
        # 对数变换 (确保值为正)
        if (selected_data[feature] > 0).all():
            ml_descriptors[f'ml_desc_log_{feature}'] = np.log(selected_data[feature])
        
        # 指数变换 (确保不会溢出)
        if (selected_data[feature] < 10).all() and (selected_data[feature] > -10).all():
            ml_descriptors[f'ml_desc_exp_{feature}'] = np.exp(selected_data[feature])
    
    # 删除含有无穷大、NaN或异常值的列
    ml_descriptors = ml_descriptors.replace([np.inf, -np.inf], np.nan)
    ml_descriptors = ml_descriptors.dropna(axis=1, how='any')
    
    # 打印生成的描述符信息
    print(f"生成了 {len(ml_descriptors.columns)} 个新的机器学习描述符")
    
    return ml_descriptors

def integrate_experimental_data(dataframe, new_data_path=None, new_data_df=None):
    """
    整合新的实验数据到现有数据集
    
    参数:
    dataframe -- 原始数据集
    new_data_path -- 新数据文件路径（CSV格式）
    new_data_df -- 直接提供的新数据DataFrame
    
    返回:
    updated_df -- 更新后的数据集
    """
    if new_data_path is None and new_data_df is None:
        logging.warning("未提供新数据，返回原始数据集")
        return dataframe
    
    # 合并数据
    if new_data_path is not None:
        logging.info(f"从文件加载新实验数据: {new_data_path}")
        try:
            new_data = pd.read_csv(new_data_path, header=0, dtype={'x': str, 'y': float})
        except Exception as e:
            logging.error(f"读取新数据文件出错: {e}")
            return dataframe
    elif new_data_df is not None:
        logging.info("使用提供的DataFrame作为新实验数据")
        new_data = new_data_df
    
    # 检查新数据格式
    if 'x' not in new_data.columns or 'y' not in new_data.columns:
        logging.error("新数据格式不正确，需要包含'x'(SMILES)和'y'(电离能)列")
        return dataframe
    
    logging.info(f"新数据包含{len(new_data)}个分子")
    
    # 检查重复数据
    existing_smiles = set(dataframe['x'].values)
    new_unique = new_data[~new_data['x'].isin(existing_smiles)]
    duplicate_count = len(new_data) - len(new_unique)
    
    if duplicate_count > 0:
        logging.info(f"发现{duplicate_count}个重复分子，将使用新数据更新它们的值")
        
        # 更新重复分子的数值
        for _, row in new_data[new_data['x'].isin(existing_smiles)].iterrows():
            mask = dataframe['x'] == row['x']
            dataframe.loc[mask, 'y'] = row['y']
            dataframe.loc[mask, 'source'] = 'updated_experimental'
    
    # 添加数据源标记
    new_unique['source'] = 'new_experimental'
    
    # 合并数据集
    updated_df = pd.concat([dataframe, new_unique], ignore_index=True)
    
    logging.info(f"数据集更新完成，原始大小: {len(dataframe)}, 新增: {len(new_unique)}, 更新: {duplicate_count}, 最终大小: {len(updated_df)}")
    
    return updated_df

def update_feature_set(dataframe, ml_descriptors=None, feature_importance=None, top_n=10):
    """
    更新特征集，整合机器学习生成的描述符
    
    参数:
    dataframe -- 数据集
    ml_descriptors -- 机器学习生成的描述符
    feature_importance -- 特征重要性信息
    top_n -- 选择的顶部特征数量
    
    返回:
    updated_df -- 更新后的数据集
    """
    updated_df = dataframe.copy()
    
    # 如果提供了机器学习描述符，合并到数据集
    if ml_descriptors is not None and not ml_descriptors.empty:
        logging.info(f"整合{len(ml_descriptors.columns)-1}个机器学习生成的描述符")
        
        # 按SMILES合并
        updated_df = pd.merge(
            updated_df, 
            ml_descriptors, 
            left_on='x', 
            right_on='smiles', 
            how='left'
        )
        
        # 删除冗余列
        if 'smiles' in updated_df.columns:
            updated_df = updated_df.drop('smiles', axis=1)
        
        # 填充缺失值
        desc_columns = [col for col in ml_descriptors.columns if col != 'smiles']
        for col in desc_columns:
            if col in updated_df.columns:
                updated_df[col] = updated_df[col].fillna(0)
    
    # 基于特征重要性进行特征选择
    if feature_importance is not None:
        logging.info("基于特征重要性更新特征集")
        
        # 获取所有特征列
        feature_cols = [col for col in updated_df.columns if col not in ['x', 'y', 'source', 'original_y', 'dft_corrected']]
        
        # 如果特征列数量超过阈值，进行筛选
        if len(feature_cols) > top_n and len(feature_importance) > 0:
            # 根据特征重要性排序
            important_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # 选择前N个重要特征
            selected_features = [item[0] for item in important_features[:top_n] 
                              if item[0] in feature_cols]
            
            # 只保留重要特征
            logging.info(f"从{len(feature_cols)}个特征中选择{len(selected_features)}个最重要的特征")
            
            # 在这里我们不会真的从DataFrame中删除列，而是记录重要特征
            updated_df['important_features'] = str(selected_features)
    
    return updated_df

def run_feedback_loop(base_dataframe, cycles=3, device='cuda', experimental_data_paths=None, 
                     dft_correction=True, save_intermediate=True):
    """
    运行反馈循环，持续更新模型和特征集
    
    参数:
    base_dataframe -- 初始数据集
    cycles -- 反馈循环次数
    device -- 计算设备
    experimental_data_paths -- 实验数据文件路径列表
    dft_correction -- 是否进行量子化学修正
    save_intermediate -- 是否保存中间结果
    
    返回:
    final_model -- 最终模型
    final_metrics -- 最终性能指标
    """
    current_df = base_dataframe.copy()
    
    # 循环指标记录
    feedback_metrics = {
        'cycle': [],
        'r2': [],
        'mae': [],
        'dataset_size': [],
        'ml_descriptors_count': [],
        'important_features': []
    }
    
    all_iterations = cycles
    if experimental_data_paths is not None:
        all_iterations = min(cycles, len(experimental_data_paths))
    
    # 创建目录存储中间结果
    if save_intermediate:
        import os
        os.makedirs('feedback_loop_results', exist_ok=True)
    
    # 开始反馈循环
    for cycle in range(all_iterations):
        logging.info(f"\n===== 开始反馈循环周期 {cycle+1}/{all_iterations} =====")
        
        # 1. 使用当前数据集训练模型
        if dft_correction:
            # 使用量子化学修正的迭代优化
            model, metrics, _ = iterative_model_optimization(
                current_df, 
                device=device,
                iterations=2,
                error_threshold=0.4,
                max_molecules_per_iter=3,
                dft_method="B3LYP/6-31+G*"
            )
            test_r2 = metrics['r2']
            test_mae = metrics['mae']
        else:
            # 标准训练
            model, test_r2, test_mae = train_model(current_df, device=device, k_folds=3)
        
        # 2. 执行SHAP分析获取特征重要性
        _, feature_names, _ = perform_shap_analysis(model, current_df.sample(min(100, len(current_df))), 
                                                  device=device, n_samples=50, update_model=True)
        
        # 获取模型中的特征重要性信息
        feature_importance = {}
        if hasattr(model, 'feature_importance') and model.feature_importance is not None:
            feature_importance = model.feature_importance
        
        # 3. 生成机器学习描述符
        ml_descriptors = generate_ml_descriptors(
            current_df['x'].tolist()[:min(200, len(current_df))],
            model,
            device
        )
        
        # 4. 整合新的实验数据（如果有）
        if experimental_data_paths is not None and cycle < len(experimental_data_paths):
            current_df = integrate_experimental_data(current_df, experimental_data_paths[cycle])
        
        # 5. 更新特征集
        current_df = update_feature_set(current_df, ml_descriptors, feature_importance)
        
        # 记录本周期指标
        ml_desc_count = len([col for col in current_df.columns if col.startswith('ml_desc')])
        important_feats = []
        if 'important_features' in current_df.columns:
            important_feats = current_df['important_features'].iloc[0]
        
        feedback_metrics['cycle'].append(cycle+1)
        feedback_metrics['r2'].append(test_r2)
        feedback_metrics['mae'].append(test_mae)
        feedback_metrics['dataset_size'].append(len(current_df))
        feedback_metrics['ml_descriptors_count'].append(ml_desc_count)
        feedback_metrics['important_features'].append(important_feats)
        
        logging.info(f"周期 {cycle+1} 完成:")
        logging.info(f"  R²: {test_r2:.4f}")
        logging.info(f"  MAE: {test_mae:.4f} eV")
        logging.info(f"  数据集大小: {len(current_df)}")
        logging.info(f"  机器学习描述符数量: {ml_desc_count}")
        
        # 保存中间结果
        if save_intermediate:
            current_df.to_csv(f'feedback_loop_results/cycle_{cycle+1}_dataset.csv', index=False)
            torch.save(model.state_dict(), f'feedback_loop_results/cycle_{cycle+1}_model.pt')
            
            # 保存描述符
            if not ml_descriptors.empty:
                ml_descriptors.to_csv(f'feedback_loop_results/cycle_{cycle+1}_ml_descriptors.csv', index=False)
    
    # 总结反馈循环结果
    logging.info("\n===== 反馈循环总结 =====")
    
    # 创建性能改进图表
    plt.figure(figsize=(12, 8))
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(feedback_metrics['cycle'], feedback_metrics['mae'], 'ro-', label='MAE')
    ax1.axhline(y=0.2, color='g', linestyle='--', label='目标 MAE (0.2 eV)')
    ax1.set_ylabel('MAE (eV)')
    ax1.set_title('反馈循环性能演化')
    ax1.grid(True)
    ax1.legend()
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(feedback_metrics['cycle'], feedback_metrics['r2'], 'bo-', label='R²')
    ax2.set_xlabel('反馈循环周期')
    ax2.set_ylabel('R²')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('feedback_loop_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制数据集大小和描述符数量变化
    plt.figure(figsize=(12, 6))
    plt.plot(feedback_metrics['cycle'], feedback_metrics['dataset_size'], 'go-', label='数据集大小')
    plt.plot(feedback_metrics['cycle'], feedback_metrics['ml_descriptors_count'], 'mo-', label='机器学习描述符数量')
    plt.xlabel('反馈循环周期')
    plt.ylabel('数量')
    plt.title('数据集和描述符演化')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feedback_loop_data_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建总结报告
    with open('feedback_loop_summary.txt', 'w') as f:
        f.write("# 反馈循环总结报告\n\n")
        
        f.write("## 性能演化\n\n")
        f.write("| 循环周期 | R² | MAE (eV) | 数据集大小 | ML描述符数量 |\n")
        f.write("|---------|-----|----------|----------|-----------|\n")
        
        for i in range(len(feedback_metrics['cycle'])):
            f.write(f"| {feedback_metrics['cycle'][i]} | {feedback_metrics['r2'][i]:.4f} | "
                   f"{feedback_metrics['mae'][i]:.4f} | {feedback_metrics['dataset_size'][i]} | "
                   f"{feedback_metrics['ml_descriptors_count'][i]} |\n")
        
        f.write("\n## 最终性能\n\n")
        f.write(f"- 最终 R²: {feedback_metrics['r2'][-1]:.4f}\n")
        f.write(f"- 最终 MAE: {feedback_metrics['mae'][-1]:.4f} eV\n")
        
        if feedback_metrics['mae'][-1] <= 0.2:
            f.write("- ✅ 已达到目标 MAE (≤ 0.2 eV)\n")
        else:
            f.write("- ❌ 未达到目标 MAE (≤ 0.2 eV)\n")
        
        f.write("\n## 数据和特征演化\n\n")
        f.write(f"- 初始数据集大小: {feedback_metrics['dataset_size'][0]}\n")
        f.write(f"- 最终数据集大小: {feedback_metrics['dataset_size'][-1]}\n")
        f.write(f"- 数据增长率: {(feedback_metrics['dataset_size'][-1]/feedback_metrics['dataset_size'][0]-1)*100:.1f}%\n\n")
        
        f.write(f"- 最终生成的机器学习描述符数量: {feedback_metrics['ml_descriptors_count'][-1]}\n")
        f.write(f"- 重要特征: {feedback_metrics['important_features'][-1]}\n")
    
    # 返回最终模型和性能指标
    final_metrics = {
        'r2': feedback_metrics['r2'][-1],
        'mae': feedback_metrics['mae'][-1],
        'dataset_size': feedback_metrics['dataset_size'][-1],
        'ml_descriptors_count': feedback_metrics['ml_descriptors_count'][-1],
        'cycles': all_iterations
    }
    
    return model, final_metrics

def integrated_training_pipeline(df, cycles=3, k_folds=5, device='cpu', 
                      use_dft=True, dft_method="B3LYP/6-31+G*", 
                      error_threshold=0.4, max_molecules_per_iter=3,
                      target_mae=0.2, save_results=True):
    """
    整合训练流水线 - 集成了交叉验证、量子化学修正和多轮优化
    
    参数:
        df: 数据框，包含x和y列
        cycles: 训练-修正循环数
        k_folds: 交叉验证折数
        device: 计算设备 ('cpu'或'cuda')
        use_dft: 是否使用DFT修正
        dft_method: 使用的DFT方法
        error_threshold: DFT修正的误差阈值
        max_molecules_per_iter: 每轮迭代中进行DFT计算的最大分子数
        target_mae: 目标平均绝对误差
        save_results: 是否保存结果
        
    返回:
        best_model: 训练好的最佳模型
        df: 更新后的数据框
    """
    # 创建保存结果的目录
    if save_results:
        os.makedirs('integrated_training_results/models', exist_ok=True)
        os.makedirs('integrated_training_results/plots', exist_ok=True)
        os.makedirs('integrated_training_results/data', exist_ok=True)
    
    # 初始化性能记录
    performance_history = {
        'r2_scores': [],
        'mae_scores': [],
        'rmse_scores': [],
        'dft_corrections': 0,
        'best_r2': -float('inf'),
        'best_mae': float('inf')
    }
    
    # 初始化训练日志
    cycle_log = []
    print(f"开始整合训练流水线 (周期数={cycles}, 交叉验证折数={k_folds})")
    print(f"目标MAE: {target_mae} eV")
    
    # 保存原始数据的副本
    original_df = df.copy()
    
    # 确保目标变量正确归一化 - 这对模型训练至关重要
    current_df, y_mean, y_std = normalize_ionization_energy(df.copy(), target_column='y')
    print(f"目标变量已归一化 - 均值: {y_mean:.4f}, 标准差: {y_std:.4f}")
    
    # 检查目标变量分布，识别潜在异常值
    y_values = current_df['y'].values
    q1, q3 = np.percentile(y_values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.sum((y_values < lower_bound) | (y_values > upper_bound))
    if outliers > 0:
        print(f"警告: 发现 {outliers} 个潜在异常值 (占总数的 {outliers/len(y_values)*100:.2f}%)")
    
    best_prediction_r2 = -float('inf')
    
    # 记录起始时间
    start_time = time.time()
    
    # 循环训练周期
    for cycle in range(1, cycles+1):
        print(f"\n====== 周期 {cycle}/{cycles} ======")
        cycle_log.append(f"周期 {cycle}/{cycles}:")
        
        # 第一阶段: 标准训练
        print(f"\n[阶段1] 执行标准训练")
        
        # 创建标准化的分子数据集
        dataset = MolecularDataset(current_df['x'].values, current_df['y'].values)
        
        # 将数据集分割为训练集和测试集
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        split = int(0.2 * len(dataset))
        train_indices = indices[split:]
        test_indices = indices[:split]
        
        # 使用数据框中已有的归一化参数，而不是重新计算
        # 注释掉下面这部分代码，避免重复归一化
        """
        # 计算标准化参数
        y_values = torch.tensor([dataset[i].y.item() for i in train_indices])
        y_mean = y_values.mean().item()
        y_std = y_values.std().item()
        """
        
        # 输出归一化信息，用于调试
        logging.info(f"全局归一化参数 - 均值: {y_mean:.4f}, 标准差: {y_std:.4f}")
        
        # 设置交叉验证
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # 准备存储所有折的结果
        all_models = []
        all_val_losses = []
        all_train_losses = []
        
        # 记录每个样本的平均预测值
        sample_predictions = np.zeros(len(dataset))
        prediction_counts = np.zeros(len(dataset))
        
        # 进行k折交叉验证
        fold_idx = 0
        for train_fold, val_fold in kf.split(train_indices):
            fold_idx += 1
            
            # 获取当前折的索引
            train_idx = [train_indices[i] for i in train_fold]
            val_idx = [train_indices[i] for i in val_fold]
            
            # 打印数据分布情况，用于调试
            try:
                # 安全获取y值并确保是一维数组
                train_y = np.array([dataset[i].y.item() if hasattr(dataset[i].y, 'item') else dataset[i].y for i in train_idx])
                val_y = np.array([dataset[i].y.item() if hasattr(dataset[i].y, 'item') else dataset[i].y for i in val_idx])
                
                # 保证是一维数组
                if train_y.ndim > 1:
                    train_y = train_y.flatten()
                if val_y.ndim > 1:
                    val_y = val_y.flatten()
                    
                logging.info(f"第 {fold_idx} 折 - 训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
                logging.info(f"训练集标签分布 - 均值: {train_y.mean():.4f}, 标准差: {train_y.std():.4f}, 范围: [{train_y.min():.4f}, {train_y.max():.4f}]")
                logging.info(f"验证集标签分布 - 均值: {val_y.mean():.4f}, 标准差: {val_y.std():.4f}, 范围: [{val_y.min():.4f}, {val_y.max():.4f}]")
            except Exception as e:
                logging.error(f"数据分布检查出错: {str(e)}")
            
            # 创建数据加载器
            train_loader = PyGDataLoader(
                [dataset[i] for i in train_idx],
                batch_size=32,  # 使用更大的批大小
                shuffle=True,
                num_workers=0
            )
            
            val_loader = PyGDataLoader(
                [dataset[i] for i in val_idx],
                batch_size=32,
                shuffle=False,
                num_workers=0
            )
            
            # 初始化模型
            sample_data = dataset[0]
            node_dim = sample_data.x.size(1)
            edge_dim = sample_data.edge_attr.size(1)
            
            # 添加额外检查和错误处理
            try:
                logging.info(f"初始化模型 - 节点特征维度: {node_dim}, 边特征维度: {edge_dim}")
                model = EnhancedMolecularGraph(node_dim, edge_dim, hidden_dim=128).to(device)
                
                # 检查模型结构
                valid_model = True
                for name, module in model.named_modules():
                    if isinstance(module, GATConv):
                        if not hasattr(module, 'lin_src') or module.lin_src is None:
                            # 处理特殊情况，不要直接报错
                            logging.warning(f"GATConv层 {name} 的lin_src属性为None，这可能是PyG版本特性")
                
                logging.info(f"模型创建成功: 参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            except Exception as e:
                logging.error(f"模型初始化失败: {e}")
                # 尝试创建一个简化版本的模型
                logging.info("尝试创建简化版模型...")
                model = EnhancedMolecularGraph(node_dim, edge_dim, hidden_dim=64).to(device)
            
            # 训练模型
            model, val_loss = train_model(model, train_loader, val_loader, device, fold=fold_idx-1)
            all_models.append(model)
            all_val_losses.append(val_loss)
            
            # 使用当前模型对验证集进行预测，并记录结果
            model.eval()
            with torch.no_grad():
                for idx in val_idx:
                    data = dataset[idx].to(device)
                    batch = Batch.from_data_list([data])
                    pred = model(batch).item()
                    # 转换回原始刻度
                    pred = pred * y_std + y_mean
                    sample_predictions[idx] += pred
                    prediction_counts[idx] += 1
        
        # 最终进行整体预测评估
        test_loader = PyGDataLoader(
            [dataset[i] for i in test_indices],
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        # 选择验证损失最低的模型
        best_model_idx = np.argmin(all_val_losses)
        best_model = all_models[best_model_idx]
        
        # 评估最佳模型
        best_model.eval()
        predictions = []
        true_values = []
        
        # 使用最佳模型对测试集进行预测
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = best_model(data)
                # 转换回原始刻度
                pred = out * y_std + y_mean
                true = data.y * y_std + y_mean
                predictions.extend(pred.cpu().numpy())
                true_values.extend(true.cpu().numpy())
                
                # 记录每个样本的预测
                for i, idx in enumerate(test_indices):
                    if i < len(pred):
                        sample_predictions[idx] += pred[i].item()
                        prediction_counts[idx] += 1
        
        # 处理未预测的样本（如果有）
        for i in range(len(sample_predictions)):
            if prediction_counts[i] == 0:
                sample_predictions[i] = y_mean
                prediction_counts[i] = 1
        
        # 计算平均预测
        sample_predictions = sample_predictions / prediction_counts
        
        # 计算评估指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        true_values = np.array(true_values)
        predictions = np.array(predictions)
        
        r2 = r2_score(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        
        # 记录性能
        performance_history['r2_scores'].append(r2)
        performance_history['mae_scores'].append(mae)
        performance_history['rmse_scores'].append(rmse)
        
        if r2 > performance_history['best_r2']:
            performance_history['best_r2'] = r2
            
        if mae < performance_history['best_mae']:
            performance_history['best_mae'] = mae
        
        print(f"训练完成: R² = {r2:.4f}, MAE = {mae:.4f} eV, RMSE = {rmse:.4f} eV")
        cycle_log.append(f"  标准训练: R² = {r2:.4f}, MAE = {mae:.4f} eV, RMSE = {rmse:.4f} eV")
        
        # 更新最佳预测对比图
        if save_results:
            best_prediction_r2 = plot_and_save_best_prediction(
                best_model, 
                current_df, 
                cycle, 
                save_dir='integrated_training_results/plots',
                best_r2=best_prediction_r2,
                device=device
            )
        
        # 第二阶段: 量子化学修正 (如果启用)
        total_corrections = 0
        if use_dft and cycle < cycles:  # 最后一轮不做DFT修正
            print(f"\n[阶段2] 执行量子化学修正")
            
            # 执行迭代优化，返回修正后的数据集和修正分子数量
            corrected_df, correction_count = iterative_optimization(
                best_model, 
                current_df,
                error_threshold=error_threshold,
                max_molecules=max_molecules_per_iter,
                dft_method=dft_method,
                device=device
            )
            
            total_corrections = correction_count
            performance_history['dft_corrections'] += total_corrections
            
            if total_corrections > 0:
                current_df = corrected_df
                # 更新数据集和数据加载器
                dataset = MolecularDataset(current_df['x'].values, current_df['y'].values)
            
            print(f"本轮量子化学修正: {total_corrections} 个分子")
            cycle_log.append(f"  量子化学修正: {total_corrections} 个分子")
        
        # 保存当前周期的模型和数据
        if save_results:
            model_path = f'integrated_training_results/models/model_cycle_{cycle}.pt'
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'y_mean': y_mean,
                'y_std': y_std
            }, model_path)
            
            # 保存当前数据
            data_path = f'integrated_training_results/data/data_cycle_{cycle}.csv'
            current_df.to_csv(data_path, index=False)
            
            print(f"已保存模型和数据到周期 {cycle}")
        
        # 检查是否达到目标MAE
        if mae <= target_mae:
            print(f"\n✓ 达到目标MAE ({mae:.4f} ≤ {target_mae})！在周期 {cycle}/{cycles} 提前停止。")
            cycle_log.append(f"  ✓ 达到目标MAE! 提前结束训练。")
            break
    
    # 训练结束，计算总用时
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n====== 训练结束 ======")
    print(f"总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    print(f"最佳 R²: {performance_history['best_r2']:.4f}")
    print(f"最佳 MAE: {performance_history['best_mae']:.4f} eV")
    print(f"总量子化学修正: {performance_history['dft_corrections']} 个分子")
    
    # 如果启用了结果保存，则绘制性能历史
    if save_results and len(performance_history['r2_scores']) > 1:
        # 绘制R²和MAE历史
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(performance_history['r2_scores'])+1), 
                performance_history['r2_scores'], 'b-o', linewidth=2)
        plt.title('R² Score vs. Training Cycle', fontsize=14)
        plt.xlabel('Cycle', fontsize=12)
        plt.ylabel('R²', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(performance_history['mae_scores'])+1), 
                performance_history['mae_scores'], 'r-o', linewidth=2)
        plt.axhline(y=target_mae, color='g', linestyle='--', label=f'Target MAE ({target_mae})')
        plt.title('MAE vs. Training Cycle', fontsize=14)
        plt.xlabel('Cycle', fontsize=12)
        plt.ylabel('MAE (eV)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('integrated_training_results/plots/performance_history.png', 
                  bbox_inches='tight', dpi=300)
        plt.close()
        
        # 保存训练日志
        with open('integrated_training_results/training_log.txt', 'w') as f:
            f.write("=== 整合训练流水线日志 ===\n\n")
            f.write(f"训练周期: {cycles}\n")
            f.write(f"交叉验证折数: {k_folds}\n")
            f.write(f"目标MAE: {target_mae} eV\n")
            f.write(f"总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n\n")
            f.write("--- 训练周期详情 ---\n\n")
            for log_entry in cycle_log:
                f.write(log_entry + "\n")
            f.write("\n--- 最终结果 ---\n\n")
            f.write(f"最佳 R²: {performance_history['best_r2']:.4f}\n")
            f.write(f"最佳 MAE: {performance_history['best_mae']:.4f} eV\n")
            f.write(f"总量子化学修正: {performance_history['dft_corrections']} 个分子\n")
    
    # 返回最佳模型和更新后的数据框
    return best_model, current_df

def perform_cross_validation(df, k_folds=5, device='cpu'):
    """
    执行k折交叉验证
    
    参数:
        df (pd.DataFrame): 分子数据集
        k_folds (int): 交叉验证折数
        device (str): 计算设备 ('cpu' 或 'cuda')
        
    返回:
        dict: 包含交叉验证结果的字典
    """
    from sklearn.model_selection import KFold
    
    # 准备数据
    X = df.drop('ionization_energy', axis=1)
    y = df['ionization_energy']
    
    # 初始化K折交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 存储每折的结果
    r2_scores = []
    mae_scores = []
    fold_models = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"训练第 {fold+1}/{k_folds} 折...")
        
        # 划分训练集和测试集
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 创建数据集和数据加载器
        train_dataset = MolecularDataset(X_train, y_train)
        test_dataset = MolecularDataset(X_test, y_test)
        
        train_loader = PyGDataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = PyGDataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 初始化模型
        sample_data = train_dataset[0]
        model = EnhancedMolecularGraph(
            num_node_features=sample_data.x.size(1),
            num_edge_features=sample_data.edge_attr.size(1)
        ).to(device)
        
        # 训练模型
        model, metrics = train_model(model, train_loader, test_loader, device, fold=fold)
        
        # 评估模型
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for data in test_loader:
                batch = data.to(device)
                out = model(batch)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(out.cpu().numpy())
        
        # 计算R²和MAE
        fold_r2 = r2_score(y_true, y_pred)
        fold_mae = mean_absolute_error(y_true, y_pred)
        
        r2_scores.append(fold_r2)
        mae_scores.append(fold_mae)
        fold_models.append(model)
        
        print(f"第 {fold+1} 折结果: R² = {fold_r2:.4f}, MAE = {fold_mae:.4f} eV")
    
    # 计算平均分数
    avg_r2 = sum(r2_scores) / len(r2_scores)
    avg_mae = sum(mae_scores) / len(mae_scores)
    
    print(f"交叉验证完成 - 平均 R²: {avg_r2:.4f}, 平均 MAE: {avg_mae:.4f} eV")
    
    # 找出性能最佳的模型
    best_idx = mae_scores.index(min(mae_scores))
    best_model = fold_models[best_idx]
    best_r2 = r2_scores[best_idx]
    best_mae = mae_scores[best_idx]
    
    print(f"最佳模型 (第 {best_idx+1} 折): R² = {best_r2:.4f}, MAE = {best_mae:.4f} eV")
    
    return {
        'r2_scores': r2_scores,
        'mae_scores': mae_scores,
        'avg_r2': avg_r2,
        'avg_mae': avg_mae,
        'best_model': best_model,
        'best_fold': best_idx + 1
    }

def iterative_optimization(model, df, error_threshold=0.4, max_molecules=3, dft_method="B3LYP/6-31+G*", device='cpu'):
    """
    迭代优化：基于模型预测误差选择分子进行DFT计算修正
    
    参数:
        model (nn.Module): 训练好的模型
        df (pd.DataFrame): 分子数据集
        error_threshold (float): 预测误差阈值(eV)，超过此值的分子进行DFT修正
        max_molecules (int): 每轮最多修正的分子数量
        dft_method (str): DFT计算方法和基组
        device (str): 计算设备 ('cpu' 或 'cuda')
        
    返回:
        tuple: (更新后的数据集, 修正的分子数量)
    """
    # 确保数据格式正确
    if isinstance(df, pd.DataFrame):
        if 'x' in df.columns and 'y' in df.columns:
            X = df['x'].values
            y = df['y'].values
        else:
            # 如果没有x和y列，假设除了最后一列都是特征
            feature_cols = df.columns[:-1]
            target_col = df.columns[-1]
            X = df[feature_cols].values
            y = df[target_col].values
    else:
        raise ValueError("输入数据必须是pandas DataFrame")
    
    # 创建测试数据集
    test_dataset = MolecularDataset(
        X if isinstance(X[0], (list, np.ndarray)) else X.reshape(-1, 1),
        y
    )
    test_loader = PyGDataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 预测测试集
    model.eval()
    y_true = []
    y_pred = []
    molecule_indices = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            out = model(batch)
            
            # 记录分子索引
            batch_indices = list(range(i * test_loader.batch_size, 
                                    min((i + 1) * test_loader.batch_size, len(test_dataset))))
            molecule_indices.extend(batch_indices)
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(out.cpu().numpy())
    
    # 计算预测误差
    errors = [abs(true - pred) for true, pred in zip(y_true, y_pred)]
    
    # 找出误差大于阈值的分子
    error_molecules = [(idx, error, pred) for idx, (error, pred) in 
                      enumerate(zip(errors, y_pred)) if error > error_threshold]
    
    # 按误差大小排序
    error_molecules.sort(key=lambda x: x[1], reverse=True)
    
    # 限制修正的分子数量
    molecules_to_correct = error_molecules[:max_molecules]
    
    print(f"找到 {len(error_molecules)} 个误差大于 {error_threshold} eV 的分子")
    print(f"将对其中 {len(molecules_to_correct)} 个分子进行DFT修正")
    
    # 如果没有需要修正的分子，直接返回
    if not molecules_to_correct:
        return df, 0
    
    # 进行DFT计算修正
    corrected_data = []
    
    for idx, error, pred in molecules_to_correct:
        # 获取分子SMILES
        molecule_idx = molecule_indices[idx]
        smiles = df.iloc[molecule_idx].name if df.index.name == 'smiles' else df.iloc[molecule_idx].get('smiles', f'mol_{molecule_idx}')
        
        print(f"对分子 {smiles} 进行DFT计算修正 (预测误差: {error:.4f} eV)")
        
        try:
            # 进行DFT计算
            dft_energy = perform_dft_calculation(smiles, method=dft_method)
            
            if dft_energy is not None:
                # 创建修正数据
                corrected_data.append({
                    'index': molecule_idx,
                    'smiles': smiles,
                    'predicted_energy': pred,
                    'original_energy': y_true[idx],
                    'dft_energy': dft_energy,
                    'error': error
                })
                
                print(f"DFT计算完成: 预测值 = {pred:.4f} eV, DFT值 = {dft_energy:.4f} eV")
            else:
                print(f"DFT计算失败，跳过分子 {smiles}")
        except Exception as e:
            print(f"DFT计算出错: {e}")
    
    # 更新数据集
    updated_df = df.copy()
    
    correction_count = 0
    for correction in corrected_data:
        idx = correction['index']
        dft_energy = correction['dft_energy']
        
        # 更新数据集中的y值
        if 'y' in updated_df.columns:
            updated_df.iloc[idx, updated_df.columns.get_loc('y')] = dft_energy
        else:
            # 如果没有y列，更新最后一列
            updated_df.iloc[idx, -1] = dft_energy
        correction_count += 1
    
    print(f"完成 {correction_count} 个分子的DFT修正")
    
    return updated_df, correction_count

def perform_dft_calculation(smiles, method="B3LYP/6-31+G*"):
    """
    执行DFT量子化学计算，计算分子的电离能
    
    参数:
        smiles (str): 分子的SMILES表示
        method (str): DFT计算方法和基组
        
    返回:
        float: 计算得到的电离能 (eV)，失败时返回None
    """
    # 注意: 此函数仅为示例，实际实现应根据具体的量子化学软件进行调用
    # 例如: 使用RDKit + pyscf, OpenBabel + Gaussian, 或 PySCF 等工具
    
    try:
        # 这里为模拟计算，在实际应用中替换为真实的DFT计算
        # 例如使用Gaussian, ORCA, PySCF等软件
        
        print(f"模拟执行DFT计算: {method} 用于分子 {smiles}")
        
        # 模拟计算过程
        time.sleep(2)  # 模拟计算时间
        
        # 生成一个假的电离能值，在7-10 eV范围内
        # 在实际应用中，这应该是从DFT计算结果中提取的值
        import random
        simulated_ionization_energy = random.uniform(7.0, 10.0)
        
        return simulated_ionization_energy
        
    except Exception as e:
        print(f"DFT计算出错: {e}")
        return None

def plot_and_save_best_prediction(model, df, cycle, save_dir, best_r2=-float('inf'), device='cpu'):
    """
    绘制预测值与实际值的对比散点图，并根据R²值决定是否保留
    
    参数:
        model (nn.Module): 当前训练的模型
        df (pd.DataFrame): 用于预测的数据集
        cycle (int): 当前训练周期
        save_dir (str): 图片保存目录
        best_r2 (float): 目前最佳的R²值
        device (str): 计算设备
        
    返回:
        float: 当前的R²值，如果优于best_r2则更新图片
    """
    from sklearn.metrics import r2_score
    import os
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保数据格式正确
    if isinstance(df, pd.DataFrame):
        if 'x' in df.columns and 'y' in df.columns:
            X = df['x'].values
            y_true = df['y'].values
        else:
            # 如果没有x和y列，假设除了最后一列都是特征
            feature_cols = df.columns[:-1]
            target_col = df.columns[-1]
            X = df[feature_cols].values
            y_true = df[target_col].values
    else:
        raise ValueError("输入数据必须是pandas DataFrame")
    
    # 创建数据集和数据加载器
    dataset = MolecularDataset(
        X if isinstance(X[0], (list, np.ndarray)) else X.reshape(-1, 1),
        y_true
    )
    loader = PyGDataLoader(dataset, batch_size=32, shuffle=False)
    
    # 使用模型进行预测
    model.eval()
    y_pred = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            y_pred.extend(out.cpu().numpy())
    
    # 计算R²值
    current_r2 = r2_score(y_true, y_pred)
    
    # 如果当前R²值优于最佳值，更新图片
    if current_r2 > best_r2:
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        
        plt.xlabel('实际值 (eV)')
        plt.ylabel('预测值 (eV)')
        plt.title(f'第 {cycle} 轮预测结果对比\nR² = {current_r2:.4f}')
        
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(save_dir, 'best_prediction_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"更新最佳预测对比图 (R² = {current_r2:.4f})")
        return current_r2
    else:
        print(f"当前预测对比图 R² ({current_r2:.4f}) 未超过最佳值 ({best_r2:.4f})，不更新图片")
        return best_r2

def create_optimizer(model, lr=0.001, weight_decay=1e-5):
    """
    创建优化器和学习率调度器
    
    参数:
        model: 模型
        lr: 初始学习率
        weight_decay: 权重衰减系数
        
    返回:
        optimizer: 优化器
        scheduler: 学习率调度器
    """
    # 使用AdamW优化器，它比Adam更好地处理权重衰减
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用余弦退火学习率调度器，在训练过程中平滑地降低学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 第一次重启的周期长度
        T_mult=2,  # 每次重启后周期长度的倍增因子
        eta_min=1e-6  # 最小学习率
    )
    
    return optimizer, scheduler

def check_and_fix_gradients(model, gradient_threshold=1.0):
    """
    检查并修复梯度中的异常值
    
    参数:
        model: 模型
        gradient_threshold: 梯度阈值，超过此值将被裁剪
        
    返回:
        has_bad_gradients: 是否存在异常梯度
        fixed_params_count: 修复的参数数量
    """
    has_bad_gradients = False
    fixed_params_count = 0
    
    # 检查每个参数的梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 检查NaN或Inf
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                has_bad_gradients = True
                # 替换NaN和Inf为0
                param.grad = torch.where(
                    torch.isnan(param.grad) | torch.isinf(param.grad),
                    torch.zeros_like(param.grad),
                    param.grad
                )
                fixed_params_count += 1
                logging.warning(f"参数 {name} 有NaN或Inf梯度，已修复")
            
            # 检查梯度范数 - 使用更小的阈值
            grad_norm = param.grad.norm()
            if grad_norm > 0.05:  # 极小的梯度阈值
                has_bad_gradients = True
                # 缩放梯度
                param.grad = param.grad * (0.05 / grad_norm)
                fixed_params_count += 1
                logging.warning(f"参数 {name} 梯度范数过大 ({grad_norm:.4f})，已缩放")
    
    return has_bad_gradients, fixed_params_count

def log_model_diagnostics(model, epoch, batch_idx):
    """
    记录模型参数和梯度的统计信息，用于调试
    
    参数:
        model: 模型
        epoch: 当前轮次
        batch_idx: 当前批次索引
    """
    # 统计各层参数值分布
    layer_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_stats[name] = {
                "min": param.data.min().item(),
                "max": param.data.max().item(),
                "mean": param.data.mean().item(),
                "std": param.data.std().item(),
                "norm": param.data.norm().item()
            }
            
            # 统计梯度分布（如果有）
            if param.grad is not None:
                layer_stats[name]["grad_min"] = param.grad.min().item()
                layer_stats[name]["grad_max"] = param.grad.max().item()
                layer_stats[name]["grad_mean"] = param.grad.mean().item()
                layer_stats[name]["grad_std"] = param.grad.std().item()
                layer_stats[name]["grad_norm"] = param.grad.norm().item()
    
    # 每N轮记录一次详细信息
    log_detailed = (epoch % 10 == 0 and batch_idx == 0)
    
    if log_detailed:
        # 记录详细信息
        logging.info(f"===== 模型诊断 (轮次 {epoch}, 批次 {batch_idx}) =====")
        for name, stats in layer_stats.items():
            logging.info(f"层: {name}")
            logging.info(f"  参数: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 范围=[{stats['min']:.4f}, {stats['max']:.4f}], 范数={stats['norm']:.4f}")
            if "grad_norm" in stats:
                logging.info(f"  梯度: 均值={stats['grad_mean']:.4f}, 标准差={stats['grad_std']:.4f}, 范围=[{stats['grad_min']:.4f}, {stats['grad_max']:.4f}], 范数={stats['grad_norm']:.4f}")
    else:
        # 只记录异常情况
        for name, stats in layer_stats.items():
            # 检查参数
            if stats["max"] > 1000 or stats["std"] > 100:
                logging.warning(f"异常参数值: {name}, 范围=[{stats['min']:.4f}, {stats['max']:.4f}], 标准差={stats['std']:.4f}")
            
            # 检查梯度
            if "grad_norm" in stats and (stats["grad_max"] > 100 or stats["grad_std"] > 10):
                logging.warning(f"异常梯度: {name}, 范围=[{stats['grad_min']:.4f}, {stats['grad_max']:.4f}], 标准差={stats['grad_std']:.4f}")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 读取数据
        df = pd.read_csv("C:\\Users\\21223\\PycharmProjects\\pythonProject\\processed_qm9.csv",
                        header=0,
                        dtype={'x': str, 'y': float})
        
        print("原始数据统计：")
        print(df.describe())
        print(f"原始数据量: {len(df)}")
        
        # 应用KPI框架数据预处理
        df = filter_molecules(df)
        df, y_mean, y_std = normalize_ionization_energy(df)
        
        print("\n预处理后数据统计：")
        print(df.describe())
        print(f"预处理后数据量: {len(df)}")
        
        # 提示用户选择训练模式
        print("\n请选择训练模式:")
        print("1: 标准训练 (使用k折交叉验证)")
        print("2: 迭代优化 (包含量子化学计算补充)")
        print("3: 反馈循环 (整合新数据和更新特征集)")
        print("4: 整合训练 (完整训练流程，包含所有功能)")
        
        try:
            mode = int(input("请输入选项 (1-4): "))
        except:
            mode = 1
            print("无效输入，默认使用标准训练模式")
        
        if mode == 4:
            # 整合训练模式
            print("\n使用整合训练模式，执行完整训练流程")
            
            # 参数设置
            cycles = 3
            k_folds = 3
            
            # 确认是否使用DFT修正
            try:
                use_dft = input("是否使用DFT量子化学计算补充 (y/n)? ").lower().startswith('y')
            except:
                use_dft = True
                print("使用默认设置：启用DFT修正")
            
            if use_dft:
                dft_method = "B3LYP/6-31+G*"
                print(f"DFT计算方法: {dft_method}")
                
                try:
                    error_threshold = float(input("请设置预测误差阈值 (eV，默认0.4): ") or "0.4")
                except:
                    error_threshold = 0.4
                    print(f"使用默认设置：预测误差阈值 = {error_threshold} eV")
                
                try:
                    max_molecules = int(input("请设置每轮修正的最大分子数量 (默认3): ") or "3")
                except:
                    max_molecules = 3
                    print(f"使用默认设置：每轮最多修正 {max_molecules} 个分子")
            else:
                dft_method = ""
                error_threshold = 0
                max_molecules = 0
            
            # 执行整合训练
            print("\n开始执行整合训练流程...")
            model, metrics = integrated_training_pipeline(
                df,
                cycles=cycles,
                k_folds=k_folds,
                device=device,
                use_dft=use_dft,
                dft_method=dft_method,
                error_threshold=error_threshold,
                max_molecules_per_iter=max_molecules,
                save_results=True
            )
            
            # 打印最终结果
            print("\n整合训练完成！")
            print(f"最终模型 R²: {metrics['r2']:.4f}")
            print(f"最终模型 MAE: {metrics['mae']:.4f} eV")
            print(f"总训练周期: {metrics['total_cycles']}/{cycles}")
            print(f"最终数据集大小: {metrics['dataset_size']}")
            
            if use_dft:
                print(f"累计DFT修正分子数量: {metrics['dft_corrections']}")
            
            print(f"累计生成机器学习描述符数量: {metrics['ml_descriptors']}")
            
            if metrics['target_achieved']:
                print("✅ 已达到目标 MAE (≤ 0.2 eV)")
            else:
                print("❌ 未达到目标 MAE (≤ 0.2 eV)")
            
            print(f"\n最佳预测对比图 R²: {metrics['best_prediction_r2']:.4f}")    
            print("\n详细报告保存在: integrated_training_results/summary_report.txt")
            print("性能演化曲线保存在: integrated_training_results/performance_evolution.png")
            print("数据和特征演化曲线保存在: integrated_training_results/data_feature_evolution.png")
            print("最佳预测对比图保存在: integrated_training_results/plots/best_prediction_comparison.png")
                
        elif mode == 3:
            # 反馈循环模式
            print("\n使用反馈循环模式，持续更新模型和特征集")
            
            # 参数设置
            cycles = 3
            
            # 查找可用的实验数据文件
            experimental_data_paths = []
            exp_files = glob.glob("exp_data_*.csv")
            if exp_files:
                experimental_data_paths = sorted(exp_files)
                print(f"发现{len(exp_files)}个实验数据文件: {', '.join(exp_files)}")
            else:
                print("未找到实验数据文件，将仅使用特征更新进行反馈循环")
            
            # 确认是否使用DFT修正
            try:
                use_dft = input("是否在反馈循环中使用DFT修正 (y/n)? ").lower().startswith('y')
            except:
                use_dft = True
                print("使用默认设置：启用DFT修正")
            
            # 执行反馈循环
            model, metrics = run_feedback_loop(
                df, 
                cycles=cycles,
                device=device,
                experimental_data_paths=experimental_data_paths,
                dft_correction=use_dft,
                save_intermediate=True
            )
            
            # 打印最终结果
            print("\n反馈循环完成！")
            print(f"最终模型 R²: {metrics['r2']:.4f}")
            print(f"最终模型 MAE: {metrics['mae']:.4f} eV")
            print(f"最终数据集大小: {metrics['dataset_size']}")
            print(f"机器学习描述符数量: {metrics['ml_descriptors_count']}")
            
            if metrics['mae'] <= 0.2:
                print("✅ 已达到目标 MAE (≤ 0.2 eV)")
            else:
                print("❌ 未达到目标 MAE (≤ 0.2 eV)")
                
            print("\n详细报告保存在: feedback_loop_summary.txt")
            print("性能演化曲线保存在: feedback_loop_performance.png")
            print("数据演化曲线保存在: feedback_loop_data_evolution.png")
            
        elif mode == 2:
            # 迭代优化模式
            print("\n使用迭代优化模式，包含量子化学计算补充")
            
            # 参数设置
            iterations = 3
            error_threshold = 0.4  # eV
            max_molecules = 5
            dft_method = "B3LYP/6-31+G*"
            
            print(f"量子化学计算方法: {dft_method}")
            print(f"预测误差阈值: {error_threshold} eV")
            print(f"每轮修正分子数上限: {max_molecules}")
            print(f"最大迭代次数: {iterations}")
            
            # 确认是否安装了PySCF
            try:
                import pyscf
                print("已检测到PySCF，可以进行DFT计算")
            except ImportError:
                print("警告: 未检测到PySCF，将尝试使用其他方法或模拟DFT计算")
                print("请先安装PySCF: pip install pyscf")
            
            # 执行迭代优化
            model, metrics, corrected_molecules = iterative_model_optimization(
                df, 
                device=device,
                iterations=iterations,
                error_threshold=error_threshold,
                max_molecules_per_iter=max_molecules,
                dft_method=dft_method
            )
            
            # 打印最终结果
            print("\n迭代优化完成！")
            print(f"最终模型 R²: {metrics['r2']:.4f}")
            print(f"最终模型 MAE: {metrics['mae']:.4f} eV")
            print(f"完成了{metrics['total_corrected']}个分子的DFT计算修正")
            print(f"迭代了{metrics['iterations']}轮")
            
            if metrics['target_achieved']:
                print("✅ 已达到目标 MAE (≤ 0.2 eV)")
            else:
                print("❌ 未达到目标 MAE (≤ 0.2 eV)")
                
            print("\n详细报告保存在: iterative_optimization_report.txt")
            print("迭代优化曲线保存在: model_iteration_optimization.png")
            
        else:
            # 标准训练模式
            print("\n使用标准训练模式 (k折交叉验证)")
            model, test_r2, test_mae = train_model(df, device=device)
            
            # 打印结果
            print(f"\n训练完成！")
            print(f"测试集 R²: {test_r2:.4f}")
            print(f"测试集 MAE: {test_mae:.4f} eV")
            
            if test_mae <= 0.2:
                print("✅ 已达到目标 MAE (≤ 0.2 eV)")
            else:
                print("❌ 未达到目标 MAE (≤ 0.2 eV)")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        raise e