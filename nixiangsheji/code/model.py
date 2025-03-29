import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, adj):
        # x: [batch_size, num_nodes, in_channels]
        # adj: [batch_size, num_nodes, num_nodes]
        batch_size, num_nodes, _ = x.size()
        
        # 对每个样本进行图卷积操作
        output = []
        for i in range(batch_size):
            output.append(self.linear(torch.matmul(adj[i], x[i])))
        
        return torch.stack(output)

class MGCVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim=2):
        super(MGCVAE, self).__init__()
        
        # 编码器
        self.gc1 = GraphConv(input_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.gc3_mu = GraphConv(hidden_dim, latent_dim)
        self.gc3_logvar = GraphConv(hidden_dim, latent_dim)
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 解码器
        decoder_input_dim = latent_dim + hidden_dim  # 潜在变量 + 条件
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )
        
        # 原子类型预测器
        self.atom_type_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),  # 8种原子类型
            nn.Softmax(dim=-1)
        )
        
        # 化学键预测器
        self.bond_predictor = nn.Sequential(
            nn.Linear(input_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 无键，单键，双键，三键
            nn.Softmax(dim=-1)
        )
        
        # 化学规则编码器
        self.rule_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 最大化合价
        )
        
    def encode(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = F.relu(self.gc2(h, adj))
        return self.gc3_mu(h, adj), self.gc3_logvar(h, adj)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def decode(self, z, condition):
        # z: [batch_size, num_nodes, latent_dim]
        # condition: [batch_size, condition_dim]
        
        batch_size, num_nodes, _ = z.size()
        
        # 编码条件
        condition_encoded = self.condition_encoder(condition)  # [batch_size, hidden_dim]
        
        # 扩展条件以匹配节点数量
        condition_encoded = condition_encoded.unsqueeze(1).expand(-1, num_nodes, -1)
        
        # 合并潜在变量和条件
        decoder_input = torch.cat([z, condition_encoded], dim=-1)
        
        # 生成初始原子特征
        atom_features = []
        atom_types = []
        valences = []
        
        for i in range(batch_size):
            node_features = []
            node_types = []
            node_valences = []
            
            for j in range(num_nodes):
                # 生成原子特征
                features = self.decoder(decoder_input[i, j])
                
                # 预测原子类型
                type_probs = self.atom_type_predictor(features)
                
                # 预测化学规则（最大化合价）
                valence = torch.sigmoid(self.rule_encoder(features)) * 4  # 最大4个键
                
                # 将原子类型概率与特征结合
                combined_features = features * type_probs.mean()
                
                node_features.append(combined_features)
                node_types.append(type_probs)
                node_valences.append(valence)
            
            atom_features.append(torch.stack(node_features))
            atom_types.append(torch.stack(node_types))
            valences.append(torch.stack(node_valences))
        
        atom_features = torch.stack(atom_features)
        atom_types = torch.stack(atom_types)
        valences = torch.stack(valences)
        
        # 生成化学键
        bond_features = []
        for i in range(batch_size):
            bonds = []
            current_valences = valences[i].clone()
            
            # 按原子类型概率排序节点
            type_scores = atom_types[i].max(dim=1)[0]
            sorted_indices = torch.argsort(type_scores, descending=True)
            
            # 优先处理概率高的原子
            for idx1 in sorted_indices:
                if current_valences[idx1].sum() < 0.1:  # 跳过已经用完化合价的原子
                    continue
                    
                for idx2 in sorted_indices:
                    if idx1 >= idx2 or current_valences[idx2].sum() < 0.1:
                        continue
                    
                    # 合并两个原子的特征和条件
                    combined = torch.cat([
                        atom_features[i, idx1],
                        atom_features[i, idx2],
                        condition_encoded[i, 0]  # 使用第一个条件编码
                    ])
                    
                    # 预测键类型
                    bond_probs = self.bond_predictor(combined)
                    
                    # 获取最可能的键类型
                    bond_type = torch.argmax(bond_probs)
                    
                    # 如果预测有键
                    if bond_type > 0:
                        # 检查是否有足够的化合价
                        required_valence = bond_type.item()
                        if (current_valences[idx1][required_valence-1] > 0 and 
                            current_valences[idx2][required_valence-1] > 0):
                            # 添加键
                            bonds.append((idx1.item(), idx2.item(), bond_type.item()))
                            # 更新化合价
                            current_valences[idx1] -= required_valence
                            current_valences[idx2] -= required_valence
            
            bond_features.append(bonds)
        
        return atom_features, bond_features
        
    def forward(self, x, adj, condition):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar
    
    def generate(self, condition, num_samples=1):
        """根据给定条件生成分子"""
        device = next(self.parameters()).device
        batch_size = condition.size(0) if condition.dim() > 1 else 1
        
        # 从标准正态分布采样
        z = torch.randn(batch_size, 20, self.gc3_mu.linear.out_features).to(device)  # 使用较小的原子数
        
        # 确保条件张量形状正确
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        if num_samples > 1 and batch_size == 1:
            condition = condition.repeat(num_samples, 1)
            z = z.repeat(num_samples, 1, 1)
            
        # 生成分子
        return self.decode(z, condition)

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """计算VAE损失函数"""
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss 