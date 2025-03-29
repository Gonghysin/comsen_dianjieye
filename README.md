# 分子性质预测与生成 API

这是一个用于预测分子电离能和生成具有特定性质分子的 API 服务。

## 安装依赖

```bash
pip install flask flask-cors rdkit torch numpy
```

## 启动服务

```bash
python api.py
```

服务将在 `http://localhost:5714` 启动。

## API 端点

### 1. 健康检查
- **URL**: `/api/health`
- **方法**: GET
- **描述**: 检查服务是否正常运行
- **响应示例**:
```json
{
    "status": "healthy"
}
```

### 2. 预测分子电离能
- **URL**: `/api/predict`
- **方法**: POST
- **描述**: 预测给定分子的电离能
- **请求体**:
```json
{
    "smiles": "C1=CC=CC=C1"
}
```
- **响应示例**:
```json
{
    "energy": 9.234,
    "confidence": 0.95
}
```

### 3. 生成分子
- **URL**: `/api/generate`
- **方法**: POST
- **描述**: 根据目标性质生成分子
- **请求体**:
```json
{
    "affinity": 0.5,
    "ionization_energy": 10.0,
    "num_samples": 3
}
```
- **响应示例**:
```json
[
    {
        "smiles": "C1=CC=CC=C1",
        "properties": {
            "ionization_energy": 9.234,
            "affinity": 0.5
        }
    },
    {
        "smiles": "CC(=O)O",
        "properties": {
            "ionization_energy": 10.123,
            "affinity": 0.5
        }
    }
]
```

## 使用示例

### 使用 curl 预测分子电离能
```bash
curl -X POST http://localhost:5714/api/predict \
     -H "Content-Type: application/json" \
     -d '{"smiles": "C1=CC=CC=C1"}'
```

### 使用 curl 生成分子
```bash
curl -X POST http://localhost:5714/api/generate \
     -H "Content-Type: application/json" \
     -d '{"affinity": 0.5, "ionization_energy": 10.0, "num_samples": 3}'
```

### 使用 Python requests 库
```python
import requests

# 预测电离能
response = requests.post(
    'http://localhost:5714/api/predict',
    json={'smiles': 'C1=CC=CC=C1'}
)
print(response.json())

# 生成分子
response = requests.post(
    'http://localhost:5714/api/generate',
    json={
        'affinity': 0.5,
        'ionization_energy': 10.0,
        'num_samples': 3
    }
)
print(response.json())
```

## 错误处理

所有 API 端点都会返回适当的 HTTP 状态码和错误信息：

- 400: 请求参数错误
- 500: 服务器内部错误

错误响应示例：
```json
{
    "error": "错误信息描述"
}
```

## 项目简介
已经初步实现了用数据集来训练一个模型，用来预测分子的电离能和亲和能
训练用的数据集是QM9数据集
输入的是一个smiles字符串（分支化学式），输出的这是预测的电离能or亲和能
我目前训练的这个模型，是用来预测亲和能的

1

项目简介详见：[项目简介](docs/项目简介.md)

接下来的任务见：[接下来的任务](docs/接下来的任务.md)

github规则见：[github规则](docs/git约定.md)

项目日志见：[项目日志](docs/项目日志.md)

