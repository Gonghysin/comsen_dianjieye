import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import load_and_preprocess_data, MoleculeDataset
from model import MGCVAE, vae_loss
import numpy as np
import random
import torch.nn.functional as F
import os

def compute_loss(recon_atom_features, recon_bond_features, atom_features, adj_matrix, mu, logvar):
    """计算损失函数"""
    # 原子特征重构损失
    recon_loss = F.binary_cross_entropy(recon_atom_features, atom_features, reduction='sum')
    
    # 化学键重构损失
    batch_size = atom_features.size(0)
    num_nodes = atom_features.size(1)
    bond_loss = 0
    
    # 为每个样本计算键损失
    for i in range(batch_size):
        # 创建目标键矩阵
        target_bonds = torch.zeros(num_nodes, num_nodes, 4).to(atom_features.device)
        for j in range(num_nodes):
            for k in range(j+1, num_nodes):
                if adj_matrix[i, j, k] > 0.5:
                    # 根据邻接矩阵的值确定键类型
                    if adj_matrix[i, j, k] > 0.8:
                        bond_type = 3  # 三键
                    elif adj_matrix[i, j, k] > 0.6:
                        bond_type = 2  # 双键
                    else:
                        bond_type = 1  # 单键
                    target_bonds[j, k, bond_type] = 1
                    target_bonds[k, j, bond_type] = 1
                else:
                    target_bonds[j, k, 0] = 1  # 无键
                    target_bonds[k, j, 0] = 1
        
        # 计算预测的键与目标的差异
        for bond in recon_bond_features[i]:
            j, k, bond_type = bond
            if j < num_nodes and k < num_nodes:
                pred_bond = torch.zeros(4).to(atom_features.device)
                pred_bond[bond_type] = 1
                bond_loss += F.binary_cross_entropy(pred_bond, target_bonds[j, k], reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 调整损失权重
    recon_weight = 1.0
    bond_weight = 0.0001  # 进一步降低键损失的权重
    kl_weight = 0.1    # 保持KL散度的权重不变
    
    # 总损失
    total_loss = recon_weight * recon_loss + bond_weight * bond_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, bond_loss, kl_loss

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    recon_loss_total = 0
    bond_loss_total = 0
    kl_loss_total = 0
    batch_count = 0
    
    # 用于计算属性统计
    affinity_errors = []
    ionization_errors = []
    
    for batch in train_loader:
        batch_count += 1
        atom_features = batch['atom_features'].to(device)
        adj_matrix = batch['adj_matrix'].to(device)
        properties = batch['properties'].to(device)
        
        optimizer.zero_grad()
        (recon_atom_features, recon_bond_features), mu, logvar = model(atom_features, adj_matrix, properties)
        loss, recon_loss, bond_loss, kl_loss = compute_loss(recon_atom_features, recon_bond_features, atom_features, adj_matrix, mu, logvar)
        
        # 计算属性误差
        affinity_error = torch.abs(properties[:, 0] - properties[:, 0].mean()).mean().item()
        ionization_error = torch.abs(properties[:, 1] - properties[:, 1].mean()).mean().item()
        affinity_errors.append(affinity_error)
        ionization_errors.append(ionization_error)
        
        loss.backward()
        train_loss += loss.item()
        recon_loss_total += recon_loss.item()
        bond_loss_total += bond_loss.item()
        kl_loss_total += kl_loss.item()
        optimizer.step()
        
        if batch_count % 10 == 0:
            print(f'\nEpoch {epoch}, Batch {batch_count}/{len(train_loader)}')
            print(f'Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, Bond: {bond_loss.item():.4f}, KL: {kl_loss.item():.4f}')
            print(f'亲和力误差: {affinity_error:.4f}, 电离能误差: {ionization_error:.4f}')
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_loss_total / len(train_loader.dataset)
    avg_bond = bond_loss_total / len(train_loader.dataset)
    avg_kl = kl_loss_total / len(train_loader.dataset)
    avg_affinity_error = np.mean(affinity_errors)
    avg_ionization_error = np.mean(ionization_errors)
    
    print(f'\nEpoch {epoch} Summary:')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Average Recon Loss: {avg_recon:.4f}')
    print(f'Average Bond Loss: {avg_bond:.4f}')
    print(f'Average KL Loss: {avg_kl:.4f}')
    print(f'Average Affinity Error: {avg_affinity_error:.4f}')
    print(f'Average Ionization Error: {avg_ionization_error:.4f}')
    
    return avg_loss

def main():
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 训练参数
    batch_size = 16     # 减小批次大小
    epochs = 20        # 减少训练轮数
    max_samples = 1000  # 限制样本数量
    learning_rate = 1e-3
    hidden_dim = 64
    latent_dim = 32
    max_atoms = 50
    
    print(f"训练参数:")
    print(f"最大样本数: {max_samples}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {learning_rate}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"潜在空间维度: {latent_dim}")
    print(f"最大原子数: {max_atoms}")
    
    # 加载数据
    print("\n加载数据...")
    smiles_list, properties, prop_mean, prop_std = load_and_preprocess_data(
        os.path.join('data', 'smiles_affinity.csv'),
        os.path.join('data', 'smiles_ionization.csv'),
        max_samples=max_samples
    )
    
    print(f"\n数据集统计:")
    print(f"总分子数: {len(smiles_list)}")
    print(f"亲和力均值: {prop_mean[0]:.4f}, 标准差: {prop_std[0]:.4f}")
    print(f"电离能均值: {prop_mean[1]:.4f}, 标准差: {prop_std[1]:.4f}")
    
    # 创建数据集和数据加载器
    print("\n创建数据集...")
    dataset = MoleculeDataset(smiles_list, properties, max_atoms=max_atoms)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"数据集大小: {len(dataset)}")
    print(f"批次数量: {len(train_loader)}")
    
    # 创建模型
    print("\n创建模型...")
    input_dim = 5  # 原子特征维度
    model = MGCVAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("\n开始训练...")
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        avg_loss = train(model, train_loader, optimizer, device, epoch)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"\n保存最佳模型 (Loss: {best_loss:.4f})...")
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'prop_mean': torch.tensor(prop_mean),
                'prop_std': torch.tensor(prop_std)
            }, os.path.join('models', 'best_model.pt'))
    
    # 保存最终模型
    print("\n保存最终模型...")
    os.makedirs('models', exist_ok=True)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'prop_mean': torch.tensor(prop_mean),
        'prop_std': torch.tensor(prop_std)
    }, os.path.join('models', 'final_model.pt'))
    
    print("\n训练完成！")
    print(f"最佳模型已保存到 models/best_model.pt (Loss: {best_loss:.4f})")
    print(f"最终模型已保存到 models/final_model.pt")

if __name__ == '__main__':
    main() 