import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings(action='ignore')
import argparse
import os

from graph_converter import graph_features, feature_size, graph_adjacency, graph2mol, results

parser = argparse.ArgumentParser(description='Small Molecular Graph Conditional Variational Autoencoder for Multi-objective Optimization (Affinity & Ionization)')
parser.add_argument('--data', type=int, default=80000, help='Sampling (default=80000)')
parser.add_argument('--size', type=int, default=10, help='molecule size (default=10)')
parser.add_argument('--dataset', type=str, default='../data/raw_dataset/molecules.csv', help="dataset path (default='../data/raw_dataset/molecules.csv')")
parser.add_argument('--batch', type=int, default=100, help='batch size (default=100)')
parser.add_argument('--epochs', type=int, default=1000, help='epoch (default=1000)')
parser.add_argument('--test', type=float, default=0.1, help='test set ratio (default=0.1)')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default=0.00005)')
parser.add_argument('--gen', type=int, default=10000, help='number of molecules to be generated (default=10000)')
parser.add_argument('--output', type=str, default='./generated_molecules', help="output files path (default='./generated_molecules')")
args = parser.parse_args()

# 确保输出目录存在
if not os.path.exists(args.output):
    os.makedirs(args.output, exist_ok=True)
    print(f'创建输出目录: {args.output}')

print(f'\n输出目录: {os.path.abspath(args.output)}')
print(f'输出文件前缀: {os.path.basename(args.output)}\n')

print()
print('  ##########   ########## ####    ####      ########    ##########        ####    ########')
print(' ############ ############ ###    ####    ############ ############      ######     ####  ')
print('####     #### ###     #### ###    ####   ####      #### ##     ####      ######     ####  ')
print(' #####        #####       ####    ####  ####           #####            ###  ###    ####  ')
print('  ########     ########   ####    ####  ####             #######   #### ##    ###   ####  ')
print('     #######      ####### ####    ####  ####       ####    ####### #### #########   ####  ')
print('####     #### ###    ##### ###    ####   ####      #### #     #####   ####    ####  ####  ')
print(' ############ ############ ##########     ############ ############  ####      #### ####  ')
print('  #########    #########     ######         ########    #########   ####        #### #####')
print()

print()
print('Small Molecular Graph Variational Autoencoder for Affinity & Ionization Optimization')
print()
print('- Laboratory:')
print('Computational Science and Artificial Intelligence Lab')
print('School of Mechanical Engineering')
print('Soongsil Univ.')
print('Republic of Korea')
print('csailabssu.quv.kr')
print()
print('- Developer:')
print('mhlee216.github.io')
print()
print(f'- Sampling: {args.data}')
print(f'- Molecule size: {args.size}')
print(f'- Dataset: {args.dataset}')
print(f'- Batch size: {args.batch}')
print(f'- Epoch: {args.epochs}')
print(f'- Test set ratio: {args.test}')
print(f'- Learning rate: {args.lr}')
print(f'- Generated molecules: {args.gen}')
print(f'- Output path: {args.output}')
print()


class CVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, c_dim):
        super(CVAE, self).__init__()
        # encoder part
        self.fc1 = nn.Linear(x_dim+c_dim*2, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(c_dim*2+z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
    
    def encoder(self, x, c1, c2):
        concat_input = torch.cat([x, c1, c2], 1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)
    
    def decoder(self, z, c1, c2):
        concat_input = torch.cat([z, c1, c2], 1)
        h = F.relu(self.fc4(concat_input))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x, c1, c2):
        mu, log_var = self.encoder(x.view(-1, out_dim), c1, c2)
        z = self.sampling(mu, log_var)
        return self.decoder(z, c1, c2), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, out_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def one_hot(labels, class_size): 
    targets = torch.zeros(labels.shape[0], class_size)
    for i, label in enumerate(labels):
        targets[i, round(label.item())] = 1
    return Variable(targets)

def train(epoch):
    cvae.train()
    train_loss = 0
    for batch_idx, (graph, affinity, ionization) in enumerate(train_loader):
        if torch.cuda.is_available():
            graph = graph.cuda()
            affinity = one_hot(affinity, cond_dim).cuda()
            ionization = one_hot(ionization, cond_dim).cuda()
        else:
            affinity = one_hot(affinity, cond_dim)
            ionization = one_hot(ionization, cond_dim)
        optimizer.zero_grad()
        recon_batch, mu, log_var = cvae(graph, affinity, ionization)
        loss = loss_function(recon_batch, graph, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)

def test():
    cvae.eval()
    test_loss= 0
    with torch.no_grad():
        for batch_idx, (graph, affinity, ionization) in enumerate(test_loader):
            if torch.cuda.is_available():
                graph = graph.cuda()
                affinity = one_hot(affinity, cond_dim).cuda()
                ionization = one_hot(ionization, cond_dim).cuda()
            else:
                affinity = one_hot(affinity, cond_dim)
                ionization = one_hot(ionization, cond_dim)
            recon, mu, log_var = cvae(graph, affinity, ionization)
            test_loss += loss_function(recon, graph, mu, log_var).item()
    test_loss /= len(test_loader.dataset)
    print('> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


# 修改数据加载和处理部分
affinity_path = args.dataset.replace('molecules.csv', 'smiles_affinity.csv')
ionization_path = args.dataset.replace('molecules.csv', 'smiles_ionization.csv')
print(f'Loading data from:\n- {affinity_path}\n- {ionization_path}')

df_affinity = pd.read_csv(affinity_path)
df_ionization = pd.read_csv(ionization_path)

# 检查并重命名列名
print('\n原始数据集列名:')
print('- Affinity dataset:', df_affinity.columns.tolist())
print('- Ionization dataset:', df_ionization.columns.tolist())

# 重命名列
df_affinity.columns = ['SMILES', 'Affinity']
df_ionization.columns = ['SMILES', 'Ionization']

print('\n重命名后的列名:')
print('- Affinity dataset:', df_affinity.columns.tolist())
print('- Ionization dataset:', df_ionization.columns.tolist())

# 合并数据集
df = pd.merge(df_affinity, df_ionization, on='SMILES')
df['Length'] = df['SMILES'].apply(lambda x: len(Chem.MolFromSmiles(x).GetAtoms()))
df = df[df['Length'] <= args.size].reset_index(drop=True)

print('\n数据集信息:')
print('- 总数据量:', df.shape[0])
try:
    df = df.sample(n=args.data).reset_index(drop=True)
    print('- 采样数据量:', df.shape[0])
except:
    print(f'采样错误: 请将 --data 参数设置小于 {df.shape[0]}')
    quit()
print()

smiles = df['SMILES'].tolist()
data = [Chem.MolFromSmiles(line) for line in smiles]

# 修改性质数据处理
affinity = df['Affinity'].tolist()
ionization = df['Ionization'].tolist()

# 将性质值归一化到0-9的整数范围
affinity_norm = pd.qcut(affinity, q=10, labels=False).tolist()
ionization_norm = pd.qcut(ionization, q=10, labels=False).tolist()

atom_labels = sorted(set([atom.GetAtomicNum() for mol in data for atom in mol.GetAtoms()] + [0]))
atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}

bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType() for mol in data for bond in mol.GetBonds())))
bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}

print('Converting to graphs...')
data_list = []
affinity_list = []
ionization_list = []
atom_number = args.size
for i in range(len(data)):
    try:
        length = [[0] for i in range(args.size)]
        length[int(df['Length'].iloc[i])-1] = [1]
        length = torch.tensor(length)
        data_list.append(torch.cat([length, 
                                    feature_size(data[i], atom_labels, atom_number), 
                                    graph_adjacency(data[i], atom_number, bond_encoder_m)], 1).float())
        affinity_list.append(affinity_norm[i])
        ionization_list.append(ionization_norm[i])
    except:
        print('Error:', df['SMILES'].iloc[i])
        continue

train_list = []
for i in range(len(data_list)):
    train_list.append([np.array([np.array(data_list[i])]), np.array(affinity_list[i]), np.array(ionization_list[i])])

bs = args.batch
tr = 1-args.test
train_loader = torch.utils.data.DataLoader(dataset=train_list[:int(len(train_list)*tr)], batch_size=bs, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=train_list[int(len(train_list)*tr):], batch_size=bs, shuffle=True, drop_last=True)

print()
print('- Train set:', len(train_list[:int(len(train_list)*tr)]))
print('- Test set:', len(train_list[int(len(train_list)*tr):]))
print()

row_dim = train_list[0][0][0].shape[0]
col_dim = train_list[0][0][0].shape[1]
cond_dim = args.size
out_dim = row_dim*col_dim
z_dim = 128

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n使用设备: {device}')

# 初始化模型
cvae = CVAE(x_dim=out_dim, h_dim1=512, h_dim2=256, z_dim=z_dim, c_dim=cond_dim)
cvae = cvae.to(device)  # 将模型移动到设备上

optimizer = optim.Adam(cvae.parameters(), lr=args.lr)

print('开始训练模型...')
train_loss_list = []
test_loss_list = []
for epoch in range(1, args.epochs+1):
    train_loss = train(epoch)
    train_loss_list.append(train_loss)
    test_loss = test()
    test_loss_list.append(test_loss)

print()
print('生成分子...')
# 生成不同条件组合的分子
affinity_ranges = range(-1, 4)  # -1到3，共5个水平
ionization_ranges = range(3, 7)  # 3到6，共4个水平
print(f'\n生成范围:')
print(f'- Affinity范围: {list(affinity_ranges)}')
print(f'- Ionization范围: {list(ionization_ranges)}')
print(f'- 总共将生成 {len(affinity_ranges) * len(ionization_ranges)} 个文件\n')

for affinity_level in affinity_ranges:
    for ionization_level in ionization_ranges:
        cvae_df = results(cvae, affinity_level, ionization_level, args.gen, z_dim, cond_dim, 
                          atom_number, atom_labels, row_dim, col_dim, atom_decoder_m, bond_decoder_m)
        output_file = os.path.join(args.output, f'{affinity_level}_{ionization_level}.csv')
        cvae_df.to_csv(output_file, index=False)
        print(f'保存文件 {output_file} (生成了 {cvae_df.shape[0]} 个分子)...')

print()
print('完成!')
print()
