# 电解液分子生成器

本文件是一个基于深度学习的电解液分子生成器，可以根据指定的目标属性（亲和力和电离能）生成符合要求的分子结构。

## 项目结构

```
.
├── 代码/
│   ├── generate_simple_rdkit.py  # 主要的分子生成程序
│   ├── data_utils.py            # 数据处理
|   |—— model.py                #模型框架
|   |—— train_simple.py         #模型训练
|── models/                  # 训练好的模型
├── data/                        # 数据集目录
└── README.md                    # 本文件
```



## 使用方法

### 1. 运行分子生成器

基本命令格式：
```bash
python 代码/generate_simple_rdkit.py --affinity <目标亲和力> --ionization <目标电离能>
```

参数说明：
- `--affinity`：目标亲和力值
- `--ionization`：目标电离能值

示例：
```bash
python 代码/generate_simple_rdkit.py --affinity 4.0 --ionization 6.2
```

### 2. 输出说明

程序会输出：
1. 生成的分子的SMILES表示
2. 预测的亲和力和电离能值
3. 与目标值的偏差

生成的分子按照总误差（亲和力偏差 + 电离能偏差）从小到大排序。
