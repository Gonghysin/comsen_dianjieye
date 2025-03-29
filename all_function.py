import os
import sys
import torch
from predict_ie import predict_molecule

def predict_molecule_ie(smiles, model_path="best_model.pt", verbose=True):
    """
    预测分子的电离能
    
    Args:
        smiles (str): 分子的SMILES表示
        model_path (str, optional): 模型文件路径. 默认为 "best_model.pt"
        verbose (bool, optional): 是否显示详细输出. 默认为 True
        
    Returns:
        float: 预测的电离能（eV），如果预测失败则返回None
    """
    result = predict_molecule(smiles, model_path, verbose)
    if result is not None:
        return round(result, 3)
    return None

def generate_molecule_by_properties(affinity, ionization_energy, num_samples=3):
    """
    根据目标属性生成分子
    
    Args:
        affinity (float): 目标亲和能
        ionization_energy (float): 目标电离能
        num_samples (int, optional): 生成样本数量. 默认为 200
        
    Returns:
        list: 生成的分子SMILES列表
    """
    # 保存当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    try:
        # 切换到nixiangsheji目录
        nixiangsheji_dir = os.path.join(os.path.dirname(__file__), 'nixiangsheji')
        print(f"切换到目录: {nixiangsheji_dir}")
        os.chdir(nixiangsheji_dir)
        
        # 添加code目录到系统路径
        code_dir = os.path.join(nixiangsheji_dir, 'code')
        print(f"添加路径: {code_dir}")
        sys.path.insert(0, code_dir)  # 插入到路径列表的开头
        
        # 导入生成函数
        print("正在导入generate_simple_rdkit...")
        from generate_simple_rdkit import generate_molecules_vae, load_model
        print("导入成功")
        
        # 加载模型
        print("正在加载模型...")
        model_path = os.path.join(nixiangsheji_dir, 'models', 'best_model.pt')
        print(f"模型文件路径: {model_path}")
        model, prop_mean, prop_std, latent_dim = load_model(model_path)
        print("模型加载成功")
        
        # 目标属性（转换为PyTorch张量）
        print(f"目标属性: 亲和能={affinity}, 电离能={ionization_energy}")
        target_properties = torch.tensor([[affinity, ionization_energy]], dtype=torch.float32)
        print(f"目标属性张量形状: {target_properties.shape}")
        
        # 调用生成函数
        print("开始调用generate_molecules_vae...")
        molecules, properties = generate_molecules_vae(
            model=model,  # 使用加载的模型
            target_properties=target_properties,
            prop_mean=prop_mean,
            prop_std=prop_std,
            latent_dim=latent_dim,
            num_samples=num_samples
        )
        print("generate_molecules_vae调用完成")
        
        # 将分子转换为SMILES字符串
        from rdkit import Chem
        generated_smiles = []
        for mol in molecules:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                generated_smiles.append(smiles)
        
        return generated_smiles
        
    except Exception as e:
        print(f"生成分子时出错: {str(e)}")
        import traceback
        print("错误堆栈:")
        print(traceback.format_exc())
        return None
        
    finally:
        # 恢复原始工作目录
        print(f"恢复工作目录: {current_dir}")
        os.chdir(current_dir)

if __name__ == "__main__":
    # 测试电离能预测
    smiles = "CN(C)c1cccc2c1ON(Cl)O2"
    ie = predict_molecule_ie(smiles)
    print(f"预测的电离能: {ie} eV")
    
    # 测试分子生成
    target_affinity = 6
    target_ie = 10.0
    print("\n目标分子性质:")
    print(f"目标亲和能: {target_affinity}")
    print(f"目标电离能: {target_ie} eV")
    
    print("\n开始生成分子...")
    generated_molecules = generate_molecule_by_properties(target_affinity, target_ie, num_samples=3)
    if generated_molecules:
        print(f"\n生成的分子数量: {len(generated_molecules)}")
        print("\n生成分子及其预测性质:")
        for i, smiles in enumerate(generated_molecules):
            print(f"\n{i+1}. SMILES: {smiles}")
            # 预测生成分子的电离能
            predicted_ie = predict_molecule_ie(smiles)
            print(f"   预测电离能: {predicted_ie} eV")
            print(f"   与目标电离能的偏差: {abs(predicted_ie - target_ie):.2f} eV")
