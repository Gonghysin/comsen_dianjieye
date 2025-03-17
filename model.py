import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import NNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader, Batch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from tqdm import tqdm
import os
import logging
import gc
from torch.cuda.amp import autocast, GradScaler
import psutil
import math
from rdkit.Chem import Crippen, EState

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

def smiles_to_graph(smiles):
    """将SMILES转换为增强的分子图"""
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
    
    # 转换为张量
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class EnhancedMolecularGraph(torch.nn.Module):
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

def plot_prediction_results(true_values, predictions):
    """Plot scatter plot of prediction results with y=x line and correlation analysis"""
    plt.figure(figsize=(12, 8))
    
    # Calculate statistics
    correlation = np.corrcoef(true_values, predictions)[0, 1]
    r2 = r2_score(true_values, predictions)
    mse = np.mean((np.array(true_values) - np.array(predictions)) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(true_values) - np.array(predictions)))
    
    # Set style
    plt.style.use('default')
    
    # Create scatter plot
    plt.scatter(true_values, predictions, 
               alpha=0.5,
               c='blue',
               label='Predictions',
               s=30)
    
    # Add y=x line
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', 
             linewidth=2, 
             label='y=x (Ideal)')
    
    # Add regression line
    z = np.polyfit(true_values, predictions, 1)
    p = np.poly1d(z)
    plt.plot(true_values, p(true_values), 
             "g-", 
             alpha=0.8,
             linewidth=2,
             label=f'Regression (slope={z[0]:.3f})')
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Correlation (r): {correlation:.4f}\n'
    stats_text += f'R² Score: {r2:.4f}\n'
    stats_text += f'RMSE: {rmse:.4f}\n'
    stats_text += f'MAE: {mae:.4f}\n'
    stats_text += f'\nDistribution:\n'
    stats_text += f'Pred Mean: {np.mean(predictions):.2f}\n'
    stats_text += f'True Mean: {np.mean(true_values):.2f}\n'
    stats_text += f'Pred Std: {np.std(predictions):.2f}\n'
    stats_text += f'True Std: {np.std(true_values):.2f}\n'
    stats_text += f'Sample Size: {len(predictions)}'
    
    plt.text(1.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.9,
                      edgecolor='gray'))
    
    plt.title(f'Predicted vs Actual Values\nR² = {r2:.4f}', fontsize=14, pad=20)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    axis_min = min(min_val, min_val)
    axis_max = max(max_val, max_val)
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    
    plt.savefig('prediction_results.png', 
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close()

def plot_training_history(train_losses, val_losses, epochs):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, 'r--', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss Over Time', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig('loss_history.png',
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close()

def plot_y_distribution(y_values):
    """Plot distribution of y values"""
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(y_values, bins=50, density=True, alpha=0.7, color='blue')
    
    # Add kernel density estimation
    from scipy import stats
    kde = stats.gaussian_kde(y_values)
    x_range = np.linspace(min(y_values), max(y_values), 200)
    plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    plt.title('Distribution of Target Values', fontsize=14)
    plt.xlabel('Target Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig('y_distribution.png',
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close()

def train_model(dataframe, device='cuda'):
    """训练模型的主函数"""
    print(f"Using device: {device}")
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    print("Data preview:")
    print(dataframe.head())
    print("\nData statistics:")
    print(dataframe.describe())
    
    # Plot y distribution before training
    plot_y_distribution(dataframe['y'].values)
    
    # Create dataset
    dataset = []
    for _, row in tqdm(dataframe.iterrows(), desc="Processing data"):
        graph = smiles_to_graph(row['x'])
        if graph is not None:
            graph.y = torch.tensor([row['y']], dtype=torch.float)
            dataset.append(graph)
    
    # Data standardization
    y_values = torch.tensor([data.y.item() for data in dataset])
    y_mean = y_values.mean()
    y_std = y_values.std()
    
    print(f"\nTarget value statistics:")
    print(f"Mean: {y_mean:.4f}")
    print(f"Std: {y_std:.4f}")
    print(f"Min: {y_values.min():.4f}")
    print(f"Max: {y_values.max():.4f}")
    
    # Lists to store loss history
    train_losses = []
    val_losses = []
    
    # 数据集划分
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    indices = torch.randperm(len(dataset))
    
    train_dataset = [dataset[i] for i in indices[:train_size]]
    val_dataset = [dataset[i] for i in indices[train_size:train_size+val_size]]
    test_dataset = [dataset[i] for i in indices[train_size+val_size:]]
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, 
                            batch_size=32,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True if device=='cuda' else False)
    
    val_loader = DataLoader(val_dataset,
                          batch_size=32,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True if device=='cuda' else False)
    
    test_loader = DataLoader(test_dataset,
                           batch_size=32,
                           shuffle=False,
                           num_workers=0,
                           pin_memory=True if device=='cuda' else False)
    
    # 初始化模型
    sample_data = dataset[0]
    model = EnhancedMolecularGraph(
        num_node_features=sample_data.x.size(1),
        num_edge_features=sample_data.edge_attr.size(1)
    ).to(device)
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        amsgrad=True
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )
    
    # 初始化混合精度训练
    scaler = GradScaler()
    
    # 训练循环
    best_val_loss = float('inf')
    best_model_state = None
    patience = 15
    patience_counter = 0
    max_epochs = 50  # 修改最大训练轮数为50
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        
        # Training phase
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs} (Train)')
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            with autocast():
                out = model(batch)
                loss = F.mse_loss(out, batch.y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{max_epochs} (Val)'):
                batch = batch.to(device)
                out = model(batch)
                val_loss += F.mse_loss(out, batch.y).item()
                
                pred = out * y_std + y_mean
                true = batch.y * y_std + y_mean
                predictions.extend(pred.cpu().numpy())
                true_values.extend(true.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        r2 = r2_score(true_values, predictions)
        
        print(f'Epoch {epoch+1}/{max_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'R²: {r2:.4f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'y_mean': y_mean,
                'y_std': y_std
            }, 'best_model.pt')
            
            plot_prediction_results(true_values, predictions)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break
        
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Plot training history
    plot_training_history(train_losses, val_losses, len(train_losses))
    
    # 加载最佳模型进行测试
    model.load_state_dict(best_model_state)
    model.eval()
    test_predictions = []
    test_true_values = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            batch = batch.to(device)
            out = model(batch)
            pred = out * y_std + y_mean
            true = batch.y * y_std + y_mean
            test_predictions.extend(pred.cpu().numpy())
            test_true_values.extend(true.cpu().numpy())
    
    test_r2 = r2_score(test_true_values, test_predictions)
    print(f"\nFinal test results:")
    print(f"Test set R²: {test_r2:.4f}")
    
    # 绘制最终的预测结果
    plot_prediction_results(test_true_values, test_predictions)

def predict(smiles, model_path, device='cuda'):
    """使用保存的模型进行预测"""
    checkpoint = torch.load(model_path)
    
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    
    # 加载模型
    model = EnhancedMolecularGraph(
        num_node_features=graph.x.size(1),
        num_edge_features=graph.edge_attr.size(1)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 预测
    model.eval()
    graph = graph.to(device)
    with torch.no_grad():
        out = model(Batch.from_data_list([graph]))
        prediction = out.item() * (checkpoint['y_std'] + checkpoint['y_mean'])
    
    return prediction

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
        
        # 数据过滤
        df = df[(df['y'] >= -5) & (df['y'] <= 5)]
        
        print("\n过滤后数据统计：")
        print(df.describe())
        print(f"过滤后数据量: {len(df)}")
        
        # 开始训练
        train_model(df, device=device)
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        raise e