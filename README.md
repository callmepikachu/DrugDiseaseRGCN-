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