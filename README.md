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
├── README.md                # 项目说明文档
├── requirements.txt         # Python依赖包列表
├── configs/                 # 配置文件目录
│   └── config.yaml         # 主配置文件
├── src/                     # 源代码目录
│   ├── data_loader.py      # PrimeKG数据加载和预处理
│   ├── model.py            # 多任务RGCN模型定义
│   ├── train.py            # 多任务训练脚本
│   ├── evaluate.py         # 模型评估脚本
│   └── utils.py            # 工具函数和辅助方法
├── data/                    # 数据存储目录
│   ├── raw/                # 原始PrimeKG数据 (kg.csv)
│   ├── processed/          # 处理后的图数据 (processed_data.pkl)
│   └── splits/             # 数据集划分 (自动生成)
├── checkpoints/            # 模型检查点目录
│   └── drugdisease_rgcn_best.pth  # 最佳模型权重
└── logs/                   # 日志和结果目录
    ├── drugdisease_rgcn.log      # 训练日志
    ├── predictions.csv           # 预测结果
    └── evaluation_plots.png      # 评估可视化图表
```

## 快速开始

### 1. 环境设置

```bash
# 创建虚拟环境
conda create -n drugdisease python=3.9
conda activate drugdisease

# 步骤1: 安装PyTorch (CUDA版本)
pip install torch==2.7.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu128

# 步骤2: 安装PyTorch Geometric扩展
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# 步骤3: 安装PyTorch Geometric
pip install torch-geometric

# 步骤4: 安装其他依赖
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

### 多任务学习结果

我们的多任务RGCN模型同时预测**关系存在性**和**关系类型**，在PrimeKG数据集上取得了优异的性能：

#### 关系存在性预测

| 指标 | 验证集 | 测试集 | 说明 |Baseline(arXiv:2501.01644)
|------|--------|--------|------|------|
| AUC | 0.9816 | 0.9816 | ROC曲线下面积 |-|
| AP | 0.9885 | 0.9881 | 平均精度 |0.980 (Random Init) / 0.993 (LM Embedding)|
| Precision@10 | 1.0000 | 1.0000 | Top-10预测 |-|
| Precision@50 | 1.0000 | 1.0000 | Top-50预测 |-|
| Precision@100 | 1.0000 | 1.0000 | Top-100 |-|
| 整体准确率 | - | 0.9400 | 总体分类准确率 |-|

#### 关系类型预测

| 指标 | 验证集 | 测试集 | 说明 |
|------|--------|--------|------|
| 准确率 | 0.8785 | 0.8751 | 在存在关系的情况下正确预测关系类型 |

#### 详细分类性能

**混淆矩阵 (测试集)**:
- 真负例: 7,893 | 假正例: 633
- 假负例: 1,053 | 真正例: 16,000

**分类报告 (测试集)**:
| 类别 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| 负样本 | 0.91 | 0.93 | 0.92 | 8,526 |
| 正样本 | 0.96 | 0.95 | 0.96 | 17,053 |
| **加权平均** | **0.94** | **0.94** | **0.94** | **25,579** |

### 数据集统计
- **训练集**: 89,523 个样本
- **验证集**: 12,790 个样本
- **测试集**: 25,579 个样本
- **药物-疾病关系**: 85,262 条
- **关系类型**: 3种主要类型 (indication, contraindication, off-label use)

### 模型配置
- **隐藏维度**: 64
- **RGCN层数**: 2
- **批大小**: 512
- **训练轮数**: 50
- **负采样比例**: 0.5

