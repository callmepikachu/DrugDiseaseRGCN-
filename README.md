# DrugDiseaseRGCN - 基于PrimeKG的药物疾病关系预测

本项目使用关系图卷积网络(RGCN)在PrimeKG数据集上进行药物-疾病关系预测。

## 项目概述

- **数据集**: PrimeKG (Precision Medicine Knowledge Graph)
- **模型**: Relational Graph Convolutional Network (RGCN)
- **任务**: 药物-疾病关系预测
- **框架**: PyTorch Geometric

## 项目结构

```
DrugDiseaseRGCN/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   └── splits/             # 训练/验证/测试集划分
├── src/                    # 源代码
│   ├── data_loader.py      # 数据加载和预处理
│   ├── model.py           # RGCN模型定义
│   ├── train.py           # 训练脚本
│   ├── evaluate.py        # 评估脚本
│   └── utils.py           # 工具函数
├── notebooks/             # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── model_analysis.ipynb
├── configs/               # 配置文件
│   └── config.yaml
├── requirements.txt       # 依赖包
└── README.md             # 项目说明
```

## 快速开始

### 1. 环境设置

```bash
# 创建虚拟环境
conda create -n drugdisease python=3.9
conda activate drugdisease

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据下载

```bash
# 下载PrimeKG数据集
python src/data_loader.py --download
```

### 3. 数据处理

```bash
# 处理PrimeKG数据，创建图结构
python src/data_loader.py --process
```

### 4. 训练模型

```bash
# 训练RGCN模型
python src/train.py --config configs/config.yaml
```

### 5. 评估模型

```bash
# 评估模型性能
python src/evaluate.py --model_path checkpoints/best_model.pth
```

## 数据集信息

### PrimeKG统计信息
- 节点数: 129,375
- 边数: 4,050,249
- 节点类型: 10种 (药物、疾病、基因、蛋白质等)
- 关系类型: 29种

### 药物-疾病关系类型
- indication (适应症)
- contraindication (禁忌症)
- off-label use (超说明书用药)

## 模型架构

### RGCN (Relational Graph Convolutional Network)
- 处理异构图的图神经网络
- 支持多种关系类型
- 适合知识图谱上的链接预测任务

## 实验结果

| 模型 | AUC | AP | Precision@10 | Recall@10 |
|------|-----|----|--------------|-----------|
| RGCN | TBD | TBD | TBD | TBD |

## 引用

如果使用本项目，请引用：

```bibtex
@article{chandak2022building,
  title={Building a knowledge graph to enable precision medicine},
  author={Chandak, Payal and Huang, Kexin and Zitnik, Marinka},
  journal={Nature Scientific Data},
  year={2023}
}
```

## 许可证

MIT License