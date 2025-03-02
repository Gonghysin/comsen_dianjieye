import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import NNConv, Set2Set, GATConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
from torch_geometric.data import DataLoader, Data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm
import os
import json

class MolecularGraph(torch.nn.Module):
    def __init__(self, num_features, dim):
        super(MolecularGraph, self).__init__()
        self.lin0 = Linear(num_features, dim)
        
        # 简化边特征网络
        nn = Sequential(Linear(1, 32), ReLU(), Linear(32, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        # 修改GAT层的头数和输出维度
        self.gat = GATConv(dim, dim // 4, heads=4, concat=True)
        self.lin1 = Linear(dim, dim // 2)
        self.lin2 = Linear(dim // 2, 1)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        edge_attr = data.edge_attr.float()
        
        # 减少卷积层数，添加残差连接
        h1 = F.relu(self.conv(out, data.edge_index, edge_attr))
        h1 = self.dropout(h1)
        h1 = F.relu(self.gat(h1, data.edge_index))
        
        h2 = F.relu(self.conv(h1, data.edge_index, edge_attr))
        h2 = self.dropout(h2)
        h2 = F.relu(self.gat(h2, data.edge_index))
        
        # 使用全局平均池化
        out = global_mean_pool(h2, data.batch)
        out = F.relu(self.lin1(out))
        out = self.dropout(out)
        out = self.lin2(out)
        return out.view(-1)

def smiles_to_graph(smiles):
    # 如果SMILES字符串中包含逗号，只取逗号前的部分
    if ',' in smiles:
        smiles = smiles.split(',')[0]
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 获取节点特征和边特征
    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        edge_index.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edge_index.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        edge_attr.append(bond.GetBondTypeAsDouble())
        edge_attr.append(bond.GetBondTypeAsDouble())
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def train_model(dataframe):
    checkpoint_path = 'model_checkpoint.pt'
    training_state_path = 'training_state.json'
    
    print("数据预览：")
    print(dataframe.head())
    print("\n数据统计：")
    print(dataframe.describe())
    
    # 首先创建数据集
    dataset = []
    for index, row in dataframe.iterrows():
        graph = smiles_to_graph(row['x'])
        if graph is not None:
            graph.y = torch.tensor([row['y']], dtype=torch.float)
            dataset.append(graph)
    
    # 检查原始y值的分布
    y_values = torch.tensor([data.y.item() for data in dataset])
    print("\n目标值统计：")
    print(f"最小值: {y_values.min().item():.4f}")
    print(f"最大值: {y_values.max().item():.4f}")
    print(f"均值: {y_values.mean().item():.4f}")
    print(f"标准差: {y_values.std().item():.4f}")
    
    # 使用稳健的标准化方法
    y_median = y_values.median()
    y_iqr = torch.quantile(y_values, 0.75) - torch.quantile(y_values, 0.25)
    y_min = y_values.min()
    y_max = y_values.max()
    
    # 使用Min-Max标准化而不是Z-score标准化
    for data in dataset:
        data.y = (data.y - y_min) / (y_max - y_min)  # 将值缩放到0-1范围
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    indices = torch.randperm(len(dataset))  # 随机打乱数据
    train_dataset = [dataset[i] for i in indices[:train_size]]
    val_dataset = [dataset[i] for i in indices[train_size:]]
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 检查是否存在检查点
    if os.path.exists(checkpoint_path) and os.path.exists(training_state_path):
        print("发现已有检查点，加载模型配置...")
        checkpoint = torch.load(checkpoint_path)
        model_dim = checkpoint['model_state_dict']['lin0.weight'].size(0)
        print(f"使用检查点中的模型维度: {model_dim}")
    else:
        print("未发现检查点，使用默认模型配置...")
        model_dim = 64
    
    model = MolecularGraph(num_features=1, dim=model_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.2,
        patience=10,
        min_lr=1e-6
    )
    
    # 保存标准化参数
    normalization_params = {
        'y_min': y_min.item(),
        'y_max': y_max.item()
    }
    
    # 初始化训练状态
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    min_epochs = 30
    
    # 加载检查点（如果存在）
    if os.path.exists(checkpoint_path) and os.path.exists(training_state_path):
        print("加载检查点状态...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        with open(training_state_path, 'r') as f:
            state = json.load(f)
            start_epoch = state['epoch']
            train_losses = state['train_losses']
            val_losses = state['val_losses']
            best_val_loss = state['best_val_loss']
            patience_counter = state['patience_counter']
        print(f"从第 {start_epoch} 轮继续训练")
    
    pbar = tqdm(range(start_epoch, 200), desc='Training Progress')
    for epoch in pbar:
        # 训练阶段
        model.train()
        train_epoch_losses = []
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch} (Train)', leave=False)
        
        for batch in batch_pbar:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_epoch_losses.append(loss.item())
            batch_pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = np.mean(train_epoch_losses)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_epoch_losses = []
        batch_pbar = tqdm(val_loader, desc=f'Epoch {epoch} (Val)', leave=False)
        
        with torch.no_grad():
            for batch in batch_pbar:
                out = model(batch)
                loss = F.mse_loss(out, batch.y)
                val_epoch_losses.append(loss.item())
                batch_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 保存检查点
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        
        # 保存训练状态
        training_state = {
            'epoch': epoch + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'y_min': y_min.item(),
            'y_max': y_max.item()
        }
        with open(training_state_path, 'w') as f:
            json.dump(training_state, f)
        
        # 修改早停逻辑
        if epoch >= min_epochs:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'y_min': y_min,
                    'y_max': y_max
                }, 'best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}'
        })
        
        if (epoch + 1) % 10 == 0:
            plot_training_progress(train_losses, val_losses)
    
    # 加载最佳模型进行最终评估
    best_checkpoint = torch.load('best_model.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            # 反标准化预测值
            pred = out * (best_checkpoint['y_max'] - best_checkpoint['y_min']) + best_checkpoint['y_min']
            predictions.extend(pred.numpy())
            true_values.extend((batch.y * (best_checkpoint['y_max'] - best_checkpoint['y_min']) + best_checkpoint['y_min']).numpy())
    
    plot_prediction_results(predictions, true_values)

def plot_training_progress(train_losses, val_losses):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

def plot_prediction_results(predictions, true_values):
    """绘制预测结果散点图和相关性"""
    plt.figure(figsize=(10, 6))
    
    # 计算R²分数
    r2 = r2_score(true_values, predictions)
    
    # 绘制散点图
    sns.scatterplot(x=true_values, y=predictions, alpha=0.5)
    
    # 绘制对角线
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Predicted vs Actual Values (R² = {r2:.3f})')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    
    # 添加更多统计信息
    stats_text = f'Predictions:\nMean: {np.mean(predictions):.2f}\nStd: {np.std(predictions):.2f}\n'
    stats_text += f'Min: {np.min(predictions):.2f}\nMax: {np.max(predictions):.2f}\n'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig('prediction_results.png')
    plt.close()

# 添加预测函数
def predict(smiles, model_path):
    """使用保存的模型进行预测"""
    checkpoint = torch.load(model_path)
    model = MolecularGraph(num_features=1, dim=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    y_min = checkpoint['y_min']
    y_max = checkpoint['y_max']
    
    model.eval()
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    
    with torch.no_grad():
        out = model(graph)
        # 反标准化预测值
        prediction = out.item() * (y_max - y_min) + y_min
    return prediction

if __name__ == "__main__":
    # 读取CSV文件，跳过表头行
    df = pd.read_csv('dataset/processed_dataset/processed_qm9.csv', 
                     header=0,
                     dtype={'x': str, 'y': float})
    
    print("原始数据统计：")
    print(df.describe())
    print(f"原始数据量: {len(df)}")
    
    # 移除-5到5范围外的数据
    df = df[(df['y'] >= -5) & (df['y'] <= 5)]
    
    print("\n过滤后数据统计：")
    print(df.describe())
    print(f"过滤后数据量: {len(df)}")
    
    # 绘制过滤后的数据分布
    plt.figure(figsize=(10, 6))
    plt.hist(df['y'], bins=50)
    plt.title('Distribution of y values (filtered)')
    plt.xlabel('y')
    plt.ylabel('Count')
    plt.savefig('y_distribution.png')
    plt.close()
    
    train_model(df)