import torch
from model import MGCVAE
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import numpy as np
import argparse

# 定义常见的分子片段
FRAGMENTS = {
    'basic': [  # 基础片段类别
        'CC',          # 乙烷
        'CCC',         # 丙烷
        'C=C',         # 乙烯
        'C#C',         # 乙炔
        'CO',          # 甲醇
        'CCO',         # 乙醇
        'CN',          # 甲胺
        'CCN',         # 乙胺
        'CF',          # 氟甲烷
        'CCF',         # 氟乙烷
        'CC(=O)',      # 乙酰基
        'CCl',         # 氯甲烷
        'CBr',         # 溴甲烷
        'C=O',         # 醛基
        'C(=O)O'       # 羧基
    ],
    'cyclic': [
        'C1CCCCC1',    # 环己烷
        'c1ccccc1',    # 苯
        'C1CCCC1',     # 环戊烷
        'C1CCOCC1',    # 四氢吡喃
        'C1COCCO1',    # 1,4-二氧六环
        'C1CCOC1',     # 四氢呋喃
        'C1CCC(=O)O1', # γ-丁内酯
        'C1CCCO1',     # 四氢呋喃
        'C1CCCCO1',    # 四氢吡喃
        'C1COCC1',     # 环氧丁烷
        'C1CC(=O)OC1'  # β-丙内酯
    ],
    'aromatic': [
        'c1ccccc1',  # 苯环
        'c1ccncc1',  # 吡啶
        'c1cccnc1',  # 吡啶
        'c1ccsc1',   # 噻吩
        'c1ccoc1',   # 呋喃
        'c1c2ccccc2ccc1',  # 萘
        'c1ccc2c(c1)Cc1ccccc1-2',  # 芴
        'c1ccc2c(c1)c1ccccc1c2'  # 蒽
    ],
    'polar': [
        'C(=O)O',  # 羧基
        'C(=O)N',  # 酰胺
        'C#N',     # 氰基
        'N(=O)=O', # 硝基
        'S(=O)(=O)O', # 磺酸基
        'C(=O)OC', # 酯基
        'NC(=O)C', # N-乙酰基
        'C(=O)Cl', # 酰氯
        'CN(C)C=O',    # N,N-二甲基甲酰胺
        'CS(=O)(=O)C', # 二甲基亚砜
        'CC#N',        # 乙腈
        'CCS(=O)(=O)C', # 甲基磺酸乙酯
        'CC(=O)N(C)C',  # N,N-二甲基乙酰胺
        'CN(C)S(=O)(=O)C' # N,N-二甲基甲磺酰胺
    ],
    'ether': [
        'COC',     # 甲醚
        'CCOC',    # 乙醚
        'C1COCCO1', # 1,4-二氧六环
        'C1CCOC1',  # 四氢呋喃
        'COCCOC',   # 二甲氧基乙烷
        'C1CCOCC1', # 四氢吡喃
        'CCOCC',       # 二乙醚
        'COCCO',       # 乙二醇单甲醚
        'CCOCCOC',     # 二乙氧基乙烷
        'COCCF',       # 氟代甲氧基乙烷
        'COCCCOC'      # 二甲氧基丙烷
    ],
    'carbonyl': [
        'CC(=O)OC',    # 乙酸甲酯
        'CC(=O)C',     # 丙酮
        'CC(=O)CC',    # 丁酮
        'CC(=O)OCC',   # 乙酸乙酯
        'CC(=O)CF',    # 氟代丙酮
        'CC(=O)CCF',   # 氟代丁酮
        'CC(=O)CF3',   # 三氟乙酰基
        'CC(=O)OC(F)',  # 氟代乙酸酯
        'CC(=O)OCF3'   # 三氟乙酸酯
    ],
    'halogen': [
        'F',      # 氟
        'Cl',     # 氯
        'CF3',    # 三氟甲基
        'CCl3',   # 三氯甲基
        'CF2',    # 二氟亚甲基
        'CCl2',   # 二氯亚甲基
        'CF2CF3',      # 五氟乙基
        'CF2CF2CF3',   # 七氟丙基
        'CF3CF2CF3',   # 全氟丙烷
        'CF3O',        # 三氟甲氧基
        'CF3CF2O'      # 五氟乙氧基
    ],
    'nitrogen': [
        'N',      # 氨基
        'NC',     # 甲胺
        'N1CCCC1', # 吡咯烷
        'N1CCCCC1', # 哌啶
        'N1C=CC=C1', # 吡咯
        'N1C=NC=C1', # 咪唑
        'c1ccncc1',  # 吡啶
        'c1ncncc1',  # 嘧啶
        'N1C=CN=C1', # 咪唑
        'N1N=CN=C1'  # 三唑
    ]
}

def combine_fragments(fragments, max_attempts=100):
    """组合分子片段"""
    if not fragments:
        return None
        
    # 从最大的片段开始
    fragments = sorted(fragments, key=lambda x: len(x), reverse=True)
    base_mol = Chem.MolFromSmiles(fragments[0])
    if base_mol is None:
        return None
        
    # 添加其他片段
    for fragment in fragments[1:]:
        frag_mol = Chem.MolFromSmiles(fragment)
        if frag_mol is None:
            continue
            
        # 尝试不同的连接方式
        success = False
        for _ in range(max_attempts):
            try:
                # 随机选择连接点
                base_atoms = list(range(base_mol.GetNumAtoms()))
                frag_atoms = list(range(frag_mol.GetNumAtoms()))
                random.shuffle(base_atoms)
                random.shuffle(frag_atoms)
                
                # 创建组合分子
                combo = Chem.CombineMols(base_mol, frag_mol)
                editable_mol = Chem.RWMol(combo)
                
                # 尝试添加多个连接键
                num_bonds = random.randint(1, min(3, len(base_atoms), len(frag_atoms)))
                for i in range(num_bonds):
                    if i < len(base_atoms) and i < len(frag_atoms):
                        base_idx = base_atoms[i]
                        frag_idx = frag_atoms[i] + base_mol.GetNumAtoms()
                        editable_mol.AddBond(base_idx, frag_idx, Chem.BondType.SINGLE)
                
                # 尝试清理分子
                mol = editable_mol.GetMol()
                Chem.SanitizeMol(mol)
                
                base_mol = mol
                success = True
                break
                
            except Exception:
                continue
                
        if not success:
            # 如果所有尝试都失败，至少尝试添加一个连接键
            try:
                combo = Chem.CombineMols(base_mol, frag_mol)
                editable_mol = Chem.RWMol(combo)
                base_idx = base_atoms[0]
                frag_idx = frag_atoms[0] + base_mol.GetNumAtoms()
                editable_mol.AddBond(base_idx, frag_idx, Chem.BondType.SINGLE)
                mol = editable_mol.GetMol()
                Chem.SanitizeMol(mol)
                base_mol = mol
            except Exception:
                continue
    
    return base_mol

def load_model(model_path, device='cpu'):
    """加载训练好的模型"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # 创建模型
        input_dim = 5
        hidden_dim = 64
        latent_dim = 32
        model = MGCVAE(input_dim, hidden_dim, latent_dim).to(device)
    
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 获取属性归一化参数
        prop_mean = checkpoint['prop_mean']
        prop_std = checkpoint['prop_std']
        
        return model, prop_mean, prop_std, latent_dim
        
    except Exception as e:
        print(f"\n模型加载过程中出错:")
        print(f"错误类型: {type(e)}")
        print(f"错误信息: {str(e)}")
        print(f"错误堆栈:")
        import traceback
        traceback.print_exc()
        raise e

def denormalize_properties(normalized_props, prop_mean, prop_std):
    """将归一化的属性值转换回原始值"""
    return normalized_props * prop_std + prop_mean

def generate_molecules_vae(model, target_properties, prop_mean, prop_std, latent_dim, num_samples=200):
    """使用VAE模型生成分子"""
    print(f"\n目标属性:")
    print(f"亲和力 (Affinity): {target_properties[0]:.4f}")
    print(f"电离能 (Ionization): {target_properties[1]:.4f}")
    
    # 确保所有张量使用相同的数据类型
    target_properties = torch.tensor(target_properties, dtype=torch.float32)
    prop_mean = prop_mean.to(dtype=torch.float32)
    prop_std = prop_std.to(dtype=torch.float32)
    
    # 为每个样本生成带扰动的目标属性
    perturbed_targets = []
    for _ in range(num_samples):
        perturbation = torch.tensor([
            target_properties[0] * random.uniform(-0.05, 0.05),  # 减小扰动范围到±5%
            target_properties[1] * random.uniform(-0.05, 0.05)   # 减小扰动范围到±5%
        ], dtype=torch.float32)
        perturbed_target = target_properties + perturbation
        perturbed_targets.append(perturbed_target)
    
    perturbed_targets = torch.stack(perturbed_targets)
    
    # 归一化带扰动的目标属性
    normalized_props = (perturbed_targets - prop_mean) / prop_std
    normalized_props = torch.clamp(normalized_props, -3, 3) / 3
    
    # 生成分子
    unique_molecules = {}  # 使用字典存储唯一的分子，键为SMILES
    
    try:
        with torch.no_grad():
            # 使用带扰动的条件向量
            condition = normalized_props
            
            # 生成潜在向量
            z = torch.randn(num_samples, 10, latent_dim)
            
            # 使用模型生成分子特征
            atom_features, bond_features = model.decode(z, condition)
            
            # 对每个样本处理
            for i in range(num_samples):
                try:
                    # 获取当前样本的原子特征和键信息
                    sample_features = atom_features[i]
                    sample_bonds = bond_features[i]
                    
                    # 预测原子类型
                    atom_types = model.atom_type_predictor(sample_features)
                    atom_types = torch.argmax(atom_types, dim=1)
                    
                    # 根据原子类型选择片段
                    fragments = []
                    fragment_counts = {
                        'basic': 0,
                        'cyclic': 0,
                        'aromatic': 0,
                        'polar': 0,
                        'ether': 0,
                        'carbonyl': 0,
                        'halogen': 0,
                        'nitrogen': 0
                    }
                    
                    # 随机选择初始片段类型
                    initial_fragment_type = random.choices(
                        ['basic', 'cyclic', 'aromatic', 'polar', 'ether', 'carbonyl', 'halogen', 'nitrogen'],
                        weights=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]
                    )[0]
                    
                    fragments.append(random.choice(FRAGMENTS[initial_fragment_type]))
                    fragment_counts[initial_fragment_type] += 1
                    
                    # 根据原子类型添加其他片段
                    num_fragments = random.randint(3, 6)  # 增加片段数量范围
                    for _ in range(num_fragments):
                        # 根据原子类型和已有片段数量选择合适的片段类型
                        available_types = []
                        
                        if fragment_counts['basic'] < 4:
                            available_types.extend(['basic'] * 2)
                        if fragment_counts['cyclic'] < 3:
                            available_types.extend(['cyclic'] * 2)
                        if fragment_counts['aromatic'] < 3:
                            available_types.extend(['aromatic'] * 2)
                        if fragment_counts['polar'] < 3:
                            available_types.append('polar')
                        if fragment_counts['ether'] < 3:
                            available_types.append('ether')
                        if fragment_counts['carbonyl'] < 3:
                            available_types.append('carbonyl')
                        if fragment_counts['halogen'] < 3:
                            available_types.append('halogen')
                        if fragment_counts['nitrogen'] < 3:
                            available_types.extend(['nitrogen'] * 2)
                            
                        if available_types:
                            fragment_type = random.choice(available_types)
                            fragments.append(random.choice(FRAGMENTS[fragment_type]))
                            fragment_counts[fragment_type] += 1
                    
                    # 组合片段生成分子
                    mol = combine_fragments(fragments)
                    if mol is not None:
                        # 获取当前样本的预测属性
                        predicted_props = denormalize_properties(normalized_props[i] * 3, prop_mean, prop_std)
                        
                        # 生成规范化的SMILES
                        smiles = Chem.MolToSmiles(mol, canonical=True)
                        
                        # 如果是新的分子，则添加到字典中
                        if smiles not in unique_molecules:
                            unique_molecules[smiles] = (mol, predicted_props)
                        
                except Exception:
                    continue
    
    except Exception as e:
        print(f"生成过程出错: {str(e)}")
    
    # 将字典转换为列表
    molecules = []
    properties = []
    for smiles, (mol, props) in unique_molecules.items():
        molecules.append(mol)
        properties.append(props)
    
    return molecules, properties

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='生成具有目标性质的分子')
    parser.add_argument('--affinity', type=float, help='目标亲和力值')
    parser.add_argument('--ionization', type=float, help='目标电离能值')
    parser.add_argument('--num_samples', type=int, default=200, help='生成分子的数量（默认：200）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认：42）')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt', help='模型文件路径（默认：models/best_model.pt）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有提供参数，提示用户输入
    if args.affinity is None:
        args.affinity = float(input("请输入目标亲和力值: "))
    if args.ionization is None:
        args.ionization = float(input("请输入目标电离能值: "))
    
    print(f"\n目标属性:")
    print(f"亲和力: {args.affinity}")
    print(f"电离能: {args.ionization}")
    print(f"使用模型: {args.model_path}")
    
    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    try:
        # 加载模型
        model, prop_mean, prop_std, latent_dim = load_model(args.model_path)
        
        # 设置目标属性
        target_properties = [args.affinity, args.ionization]
        
        # 生成分子
        molecules, properties = generate_molecules_vae(
            model, 
            target_properties, 
            prop_mean, 
            prop_std, 
            latent_dim,
            num_samples=args.num_samples
        )
        
        # 显示生成的分子
        if molecules:
            print(f"\n成功生成 {len(molecules)} 个唯一分子")
            
            # 计算每个分子的综合误差并创建排序列表
            molecule_errors = []
            for i, (mol, props) in enumerate(zip(molecules, properties)):
                affinity_error = abs(props[0] - args.affinity)
                ionization_error = abs(props[1] - args.ionization)
                total_error = affinity_error + ionization_error
                smiles = Chem.MolToSmiles(mol)
                molecule_errors.append((total_error, affinity_error, ionization_error, smiles, props))
            
            # 按照综合误差排序
            molecule_errors.sort(key=lambda x: x[0])
            
            print("\n生成的分子按误差从小到大排序：")
            for i, (total_error, affinity_error, ionization_error, smiles, props) in enumerate(molecule_errors, 1):
                print(f"\n分子 {i} (总误差: {total_error:.4f}):")
                print(f"SMILES: {smiles}")
                print(f"亲和力: {props[0]:.4f}")
                print(f"电离能: {props[1]:.4f}")
                print(f"亲和力偏差: {affinity_error:.4f}")
                print(f"电离能偏差: {ionization_error:.4f}")
        else:
            print("未能生成有效分子")
            
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        raise e 