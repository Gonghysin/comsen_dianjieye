import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import NNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, EState
import math
import argparse
import os
from rdkit.Chem import rdMolDescriptors  # 用于获取分子式

def get_electronegativity(atomic_num):
    """估算原子的电负性，基于元素周期表位置"""
    en_map = {
        1: 2.20,  # H
        2: 0.0,   # He
        3: 0.98,  # Li
        4: 1.57,  # Be
        5: 2.04,  # B
        6: 2.55,  # C
        7: 3.04,  # N
        8: 3.44,  # O
        9: 3.98,  # F
        10: 0.0,  # Ne
        11: 0.93, # Na
        12: 1.31, # Mg
        13: 1.61, # Al
        14: 1.90, # Si
        15: 2.19, # P
        16: 2.58, # S
        17: 3.16, # Cl
        19: 0.82, # K
        20: 1.00, # Ca
        35: 2.96, # Br
        53: 2.66, # I
    }
    
    if atomic_num not in en_map:
        row = (atomic_num - 1) // 18 + 1
        col = atomic_num % 18
        if col == 0:
            col = 18
        
        if col > 12:
            base_en = 2.0 + (col - 13) * 0.3
        else:
            base_en = 1.0 + (col - 1) * 0.1
        
        row_factor = 1.0 / row
        return base_en * row_factor
    
    return en_map.get(atomic_num, 2.0)

def get_bond_electron_density(bond):
    """计算键的电子密度分布"""
    if bond is None:
        return [0.0] * 4
        
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    
    atom1_num = a1.GetAtomicNum()
    atom2_num = a2.GetAtomicNum()
    
    en1 = get_electronegativity(atom1_num)
    en2 = get_electronegativity(atom2_num)
    en_diff = abs(en1 - en2)
    
    bond_type = bond.GetBondType()
    
    if bond_type == Chem.rdchem.BondType.SINGLE:
        base_density = 2.0
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        base_density = 4.0
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        base_density = 6.0
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        base_density = 3.0
    else:
        base_density = 0.0
    
    if en_diff > 0:
        if en1 > en2:
            density_shift = en_diff * 0.2 * base_density
            density_a1 = base_density/2 + density_shift
            density_a2 = base_density/2 - density_shift
        else:
            density_shift = en_diff * 0.2 * base_density
            density_a1 = base_density/2 - density_shift
            density_a2 = base_density/2 + density_shift
    else:
        density_a1 = base_density/2
        density_a2 = base_density/2
    
    return [base_density, en_diff, density_a1, density_a2, bond.GetIsConjugated() * 1.0]

def get_atom_features(atom):
    """原子特征提取"""
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

    # 电子和结构属性
    features.extend([
        atom.GetTotalNumHs(),         # 总氢原子数
        atom.GetTotalDegree(),        # 总度
        atom.GetTotalValence(),       # 总化合价
        atom.GetExplicitValence(),    # 显式化合价
        atom.GetImplicitValence(),    # 隐式化合价
        atom.GetFormalCharge(),       # 形式电荷
        atom.GetNumRadicalElectrons() # 自由基电子数
    ])

    # 环属性
    features.extend([
        atom.IsInRingSize(3) * 1,     # 3元环
        atom.IsInRingSize(4) * 1,     # 4元环
        atom.IsInRingSize(5) * 1,     # 5元环
        atom.IsInRingSize(6) * 1,     # 6元环
        atom.IsInRingSize(7) * 1,     # 7元环
    ])

    # 手性
    features.extend([
        atom.GetChiralTag() != 0,     # 手性
        atom.HasProp('_CIPCode'),     # CIP构型
    ])

    # 元素(1-118)的独热编码
    atomic_num = atom.GetAtomicNum()
    atom_type = [1 if i == atomic_num else 0 for i in range(1, 119)]
    features.extend(atom_type)

    # 杂化类型独热编码
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hybridization = [1 if atom.GetHybridization() == h else 0 for h in hybridization_types]
    features.extend(hybridization)

    # 电负性和其他化学属性
    try:
        features.extend([
            float(Crippen.MolLogP([atom])),  # 亲脂性
            float(Crippen.MolMR([atom])),    # 摩尔折射率
            float(EState.EStateIndices([atom])[0]),  # E-state指数
        ])
    except:
        features.extend([0, 0, 0])

    # 周期表属性
    try:
        features.extend([
            atom.GetAtomicNum() % 18,  # 主族
            (atom.GetAtomicNum() - 1) // 18 + 1,  # 周期
        ])
    except:
        features.extend([0, 0])
    
    # 电负性特征
    try:
        en = get_electronegativity(atom.GetAtomicNum())
        en_features = [
            en,                          # 原子电负性
            en * atom.GetFormalCharge(), # 电负性与电荷的交互
            en / (atom.GetDegree() + 1), # 归一化电负性
        ]
        features.extend(en_features)
    except:
        features.extend([0, 0, 0])

    return features

def get_bond_features(bond):
    """键特征提取"""
    if bond is None:
        return [0] * 25  # 26维特征

    features = [
        float(bond.GetBondTypeAsDouble()),
        bond.GetIsConjugated() * 1,
        bond.GetIsAromatic() * 1,
        bond.IsInRing() * 1,
        bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE,
    ]

    # 键类型独热编码
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    features.extend([1 if bond.GetBondType() == t else 0 for t in bond_types])

    # 立体构型独热编码
    stereo_types = [
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ]
    features.extend([1 if bond.GetStereo() == s else 0 for s in stereo_types])

    # 环大小特征
    features.extend([
        bond.IsInRingSize(3) * 1,
        bond.IsInRingSize(4) * 1,
        bond.IsInRingSize(5) * 1,
        bond.IsInRingSize(6) * 1,
        bond.IsInRingSize(7) * 1,
        bond.IsInRingSize(8) * 1
    ])

    # 几何特征
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
    
    # 电子密度特征
    electron_density_features = get_bond_electron_density(bond)
    features.extend(electron_density_features)
    
    return features

def smiles_to_graph(smiles):
    """将SMILES转换为分子图"""
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
        edge_features = [[0] * 25]  # 26是键特征的维度

    # 转换为张量
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class EnhancedMolecularGraph(torch.nn.Module):
    """增强型分子图神经网络模型"""
    def __init__(self, num_node_features, num_edge_features, hidden_dim=128):
        super(EnhancedMolecularGraph, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 节点编码器
        self.node_encoder = Sequential(
            Linear(num_node_features, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(0.1),
            Linear(hidden_dim, hidden_dim)
        )
        
        # 边编码器
        edge_network = Sequential(
            Linear(num_edge_features, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(0.1),
            Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        
        # 图卷积层
        self.conv1 = NNConv(hidden_dim, hidden_dim, edge_network, aggr='mean')
        self.conv2 = NNConv(hidden_dim, hidden_dim, edge_network, aggr='mean')

        # 注意力层
        self.gat1 = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1)
        self.gat2 = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1)

        # 输出层
        self.output = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = torch.nn.Dropout(0.2)
        self.batch_norm1 = BatchNorm1d(hidden_dim)
        self.batch_norm2 = BatchNorm1d(hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 节点特征编码
        x = self.node_encoder(x)

        # 第一层卷积+注意力
        x1 = F.elu(self.conv1(x, edge_index, edge_attr))
        x1 = self.batch_norm1(x1)
        x1 = self.dropout(x1)
        x1 = self.gat1(x1, edge_index)

        # 第二层卷积+注意力
        x2 = F.elu(self.conv2(x1, edge_index, edge_attr))
        x2 = self.batch_norm2(x2)
        x2 = self.dropout(x2)
        x2 = self.gat2(x2, edge_index)

        # 全局池化（结合均值和求和）
        global_mean = global_mean_pool(x2, batch)
        global_add = global_add_pool(x2, batch)
        global_features = torch.cat([global_mean, global_add], dim=1)

        # 输出预测
        out = self.output(global_features)
        return out.squeeze(-1)

def predict_molecule(smiles, model_path="best_model.pt", verbose=True):
    """预测单个分子的电离能"""
    try:
        # 设置计算设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if verbose:
            print(f"使用设备: {device}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 找不到模型文件 '{model_path}'")
            return None
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        y_mean = checkpoint['y_mean']
        y_std = checkpoint['y_std']
        
        # 转换SMILES为图
        graph = smiles_to_graph(smiles)
        if graph is None:
            print(f"错误: 无法解析SMILES: {smiles}")
            return None
            
        if verbose:
            print(f"分子图生成: {graph.num_nodes}个原子, {graph.num_edges//2}个键")
        
        # 初始化模型
        model = EnhancedMolecularGraph(
            num_node_features=graph.x.size(1),
            num_edge_features=graph.edge_attr.size(1)
        ).to(device)
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 预测
        graph = graph.to(device)
        with torch.no_grad():
            batch = Batch.from_data_list([graph])
            prediction = model(batch)
            # 反归一化得到实际值
            ionization_energy = prediction.item() * y_std + y_mean
        
        return ionization_energy
        
    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='预测分子的电离能')
    parser.add_argument('--model', type=str, default='best_model.pt', help='模型文件路径 (默认: best_model.pt)')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    
    args = parser.parse_args()
    
    # 显示欢迎信息
    print("\n" + "="*60)
    print("     分子电离能预测工具 - 基于图神经网络 + 电子密度特征     ")
    print("="*60)
    
    while True:
        try:
            # 提示用户输入
            smiles = input("\n请输入分子SMILES（输入'q'退出）: ")
            
            # 检查是否要退出
            if smiles.lower() == 'q' or smiles.lower() == 'quit' or smiles.lower() == 'exit':
                print("\n感谢使用！再见！")
                break
            
            # 检查输入是否为空
            if not smiles.strip():
                print("错误: SMILES不能为空！")
                continue
                
            print(f"正在预测: {smiles}")
            
            # 预测电离能
            energy = predict_molecule(smiles, args.model, args.verbose)
            
            # 输出结果
            if energy is not None:
                print(f"\n预测结果: {energy:.4f} eV")
                
                # 常见分子的参考值显示
                references = {
                    "CH4": 12.61,    # 甲烷
                    "C2H6": 11.52,   # 乙烷
                    "C2H4": 10.51,   # 乙烯
                    "C6H6": 9.24,    # 苯
                    "C2H5OH": 10.48, # 乙醇
                    "H2O": 12.62     # 水
                }
                
                # 查找是否有近似的参考值
                rdkit_mol = Chem.MolFromSmiles(smiles)
                if rdkit_mol:
                    formula = Chem.rdMolDescriptors.CalcMolFormula(rdkit_mol)
                    if formula in references:
                        print(f"参考值: {formula} (文献值): {references[formula]:.2f} eV")
            else:
                print("预测失败！请检查SMILES格式是否正确。")
                
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            print("\n\n操作已取消。感谢使用！")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            print("请重新输入。")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()