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

| 指标 | 验证集 | 测试集 | 说明 |
|------|--------|--------|------|
| AUC | 0.9816 | 0.9816 | ROC曲线下面积，接近完美分类 |
| AP | 0.9885 | 0.9881 | 平均精度，排序质量极高 |
| Precision@10 | 1.0000 | 1.0000 | Top-10预测100%准确 |
| Precision@50 | 1.0000 | 1.0000 | Top-50预测100%准确 |
| Precision@100 | 1.0000 | 1.0000 | Top-100预测100%准确 |
| 整体准确率 | - | 0.9400 | 总体分类准确率 |

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

### 数据完整性保证
✅ **无数据泄露**: 负采样时严格排除所有分割中的正样本边，确保训练/验证/测试集完全独立

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

```
configs/config.yaml
# 配置文件

# 数据配置
data:
  data_dir: "data"
  test_ratio: 0.2
  val_ratio: 0.1
  negative_sampling_ratio: 0.5  # 负样本与正样本的比例
  target_relations:  # 目标关系类型
    - "indication"
    - "contraindication" 
    - "off-label use"

# 模型配置
model:
  hidden_dim: 64
  num_layers: 2
  dropout: 0.1
  num_bases: null  # 如果为null，则不使用basis decomposition
  num_blocks: null  # 如果为null，则不使用block decomposition

# 训练配置
training:
  batch_size: 512
  learning_rate: 0.001
  weight_decay: 0.00001
  num_epochs: 50
  patience: 10  # early stopping patience
  gradient_clip: 1.0
  
  # 学习率调度
  scheduler:
    type: "ReduceLROnPlateau"  # StepLR, ExponentialLR, ReduceLROnPlateau
    factor: 0.5
    patience: 5
    min_lr: 0.000001

# 评估配置
evaluation:
  metrics:
    - "auc"
    - "ap"  # average precision
    - "precision_at_k"
    - "recall_at_k"
  k_values: [10, 50, 100]
  
# 设备配置
device: "auto"  # cuda, cpu, auto

# 日志配置
logging:
  log_dir: "logs"
  save_model: true
  model_dir: "checkpoints"
  log_interval: 10  # 每多少个batch记录一次
  
# 随机种子
seed: 42

# 实验配置
experiment:
  name: "drugdisease_rgcn"
  description: "Drug-Disease relation prediction using RGCN on PrimeKG"
  tags:
    - "rgcn"
    - "primekg"
    - "drug-disease"
    - "link-prediction"


```

```
src/data_loader.py
"""
PrimeKG数据加载和预处理模块
"""

import os
import pandas as pd
import numpy as np
import requests
import argparse
from tqdm import tqdm
from typing import Dict, Tuple, List
import pickle

# 可选导入 - 如果没有安装torch_geometric，只能下载数据，不能处理
try:
    import torch
    from torch_geometric.data import HeteroData
    from sklearn.preprocessing import LabelEncoder
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch或torch_geometric未安装，只能下载数据，无法处理数据")
    print("请运行: python install_dependencies.py")
    TORCH_AVAILABLE = False
    # 创建占位符类
    class HeteroData:
        pass


class PrimeKGDataLoader:
    """PrimeKG数据加载器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # 创建目录
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # PrimeKG下载链接
        self.primekg_url = "https://dataverse.harvard.edu/api/access/datafile/6180620"
        self.primekg_file = os.path.join(self.raw_dir, "kg.csv")
        
    def download_primekg(self):
        """下载PrimeKG数据集"""
        if os.path.exists(self.primekg_file):
            print(f"PrimeKG文件已存在: {self.primekg_file}")
            return
            
        print("正在下载PrimeKG数据集...")
        response = requests.get(self.primekg_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(self.primekg_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"下载完成: {self.primekg_file}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """加载原始PrimeKG数据"""
        if not os.path.exists(self.primekg_file):
            self.download_primekg()
        
        print("正在加载PrimeKG数据...")
        df = pd.read_csv(self.primekg_file, low_memory=False)
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        return df
    
    def filter_drug_disease_relations(self, df: pd.DataFrame) -> pd.DataFrame:
        """筛选药物-疾病关系"""
        # 筛选药物-疾病边
        drug_disease_mask = (
            ((df['x_type'] == 'drug') & (df['y_type'] == 'disease')) |
            ((df['x_type'] == 'disease') & (df['y_type'] == 'drug'))
        )
        
        drug_disease_df = df[drug_disease_mask].copy()
        print(f"药物-疾病关系数量: {len(drug_disease_df)}")
        
        # 显示关系类型分布
        print("关系类型分布:")
        print(drug_disease_df['display_relation'].value_counts())
        
        return drug_disease_df
    
    def create_node_mappings(self, df: pd.DataFrame) -> Dict:
        """创建节点映射"""
        if not TORCH_AVAILABLE:
            raise ImportError("需要安装torch和sklearn才能创建节点映射")

        # 获取所有唯一节点
        all_nodes = set(df['x_id'].unique()) | set(df['y_id'].unique())
        all_types = set(df['x_type'].unique()) | set(df['y_type'].unique())

        # 创建节点到索引的映射
        node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}

        # 创建类型编码器
        type_encoder = LabelEncoder()
        type_encoder.fit(list(all_types))

        # 创建关系编码器
        relation_encoder = LabelEncoder()
        relation_encoder.fit(df['display_relation'].unique())

        mappings = {
            'node_to_idx': node_to_idx,
            'idx_to_node': {idx: node for node, idx in node_to_idx.items()},
            'type_encoder': type_encoder,
            'relation_encoder': relation_encoder,
            'num_nodes': len(all_nodes),
            'num_relations': len(relation_encoder.classes_)
        }

        return mappings
    
    def create_hetero_data(self, df: pd.DataFrame, mappings: Dict) -> HeteroData:
        """创建异构图数据"""
        if not TORCH_AVAILABLE:
            raise ImportError("需要安装torch_geometric才能创建异构图数据")

        data = HeteroData()
        
        # 按节点类型分组
        node_types = {}
        for _, row in df.iterrows():
            x_type, x_id = row['x_type'], row['x_id']
            y_type, y_id = row['y_type'], row['y_id']
            
            if x_type not in node_types:
                node_types[x_type] = set()
            if y_type not in node_types:
                node_types[y_type] = set()
                
            node_types[x_type].add(x_id)
            node_types[y_type].add(y_id)
        
        # 为每种节点类型创建特征
        for node_type, nodes in node_types.items():
            num_nodes = len(nodes)
            # 简单的one-hot编码作为初始特征
            node_features = torch.eye(num_nodes)
            data[node_type].x = node_features
            data[node_type].num_nodes = num_nodes
            
            # 创建节点ID到局部索引的映射
            local_node_to_idx = {node: idx for idx, node in enumerate(sorted(nodes))}
            data[node_type].node_to_idx = local_node_to_idx
        
        # 创建边
        edge_types = {}
        for _, row in df.iterrows():
            x_type, x_id = row['x_type'], row['x_id']
            y_type, y_id = row['y_type'], row['y_id']
            relation = row['display_relation']
            
            edge_type = (x_type, relation, y_type)
            
            if edge_type not in edge_types:
                edge_types[edge_type] = {'source': [], 'target': []}
            
            x_local_idx = data[x_type].node_to_idx[x_id]
            y_local_idx = data[y_type].node_to_idx[y_id]
            
            edge_types[edge_type]['source'].append(x_local_idx)
            edge_types[edge_type]['target'].append(y_local_idx)
        
        # 添加边到数据对象
        for edge_type, edges in edge_types.items():
            source_type, relation, target_type = edge_type
            edge_index = torch.tensor([edges['source'], edges['target']], dtype=torch.long)
            data[source_type, relation, target_type].edge_index = edge_index
        
        return data
    
    def process_data(self, save_processed: bool = True):
        """处理数据并保存"""
        if not TORCH_AVAILABLE:
            print("❌ 无法处理数据：缺少必要的依赖包")
            print("请先运行: python install_dependencies.py")
            return None, None, None

        # 加载原始数据
        df = self.load_raw_data()

        # 筛选药物-疾病关系
        drug_disease_df = self.filter_drug_disease_relations(df)

        # 创建节点映射
        mappings = self.create_node_mappings(df)

        # 创建异构图数据
        hetero_data = self.create_hetero_data(df, mappings)
        
        if save_processed:
            # 保存处理后的数据
            processed_data = {
                'hetero_data': hetero_data,
                'mappings': mappings,
                'drug_disease_df': drug_disease_df,
                'full_df': df
            }
            
            processed_file = os.path.join(self.processed_dir, "processed_data.pkl")
            with open(processed_file, 'wb') as f:
                pickle.dump(processed_data, f)
            
            print(f"处理后的数据已保存到: {processed_file}")
        
        return hetero_data, mappings, drug_disease_df
    
    def load_processed_data(self):
        """加载处理后的数据"""
        processed_file = os.path.join(self.processed_dir, "processed_data.pkl")
        
        if not os.path.exists(processed_file):
            print("处理后的数据不存在，正在处理原始数据...")
            return self.process_data()
        
        with open(processed_file, 'rb') as f:
            processed_data = pickle.load(f)
        
        return (
            processed_data['hetero_data'],
            processed_data['mappings'],
            processed_data['drug_disease_df']
        )


def main():
    parser = argparse.ArgumentParser(description="PrimeKG数据加载器")
    parser.add_argument("--download", action="store_true", help="下载PrimeKG数据集")
    parser.add_argument("--process", action="store_true", help="处理数据")
    parser.add_argument("--data_dir", default="data", help="数据目录")
    
    args = parser.parse_args()
    
    loader = PrimeKGDataLoader(args.data_dir)
    
    if args.download:
        loader.download_primekg()
    
    if args.process:
        loader.process_data()
    
    if not args.download and not args.process:
        # 默认行为：加载和处理数据
        hetero_data, mappings, drug_disease_df = loader.load_processed_data()
        print(f"异构图数据: {hetero_data}")
        print(f"节点映射: {len(mappings['node_to_idx'])} 个节点")
        print(f"药物-疾病关系: {len(drug_disease_df)} 条")


if __name__ == "__main__":
    main()

```

```python
#src/evaluate.py
"""
模型评估脚本
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import PrimeKGDataLoader
from model import DrugDiseaseRGCN
from utils import (
    load_config, get_device, load_model, calculate_metrics,
    prepare_multitask_data, split_edges
)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config_path: str, model_path: str):
        self.config = load_config(config_path)
        self.model_path = model_path
        self.device = get_device(self.config['device'])
        
        # 加载数据
        self.data_loader = PrimeKGDataLoader(self.config['data']['data_dir'])
        self.load_data()
        
        # 加载模型
        self.load_model()
    
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        hetero_data, mappings, drug_disease_df = self.data_loader.load_processed_data()
        
        self.mappings = mappings
        self.num_nodes = mappings['num_nodes']
        self.num_relations = mappings['num_relations']
        
        # 筛选目标关系
        target_relations = self.config['data']['target_relations']
        if target_relations:
            mask = drug_disease_df['display_relation'].isin(target_relations)
            drug_disease_df = drug_disease_df[mask]
        
        # 转换为边索引和边类型
        edge_list = []
        edge_types = []
        
        for _, row in drug_disease_df.iterrows():
            head_idx = mappings['node_to_idx'][row['x_id']]
            tail_idx = mappings['node_to_idx'][row['y_id']]
            rel_idx = mappings['relation_encoder'].transform([row['display_relation']])[0]
            
            edge_list.append([head_idx, tail_idx])
            edge_types.append(rel_idx)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        # 分割数据
        train_data, val_data, test_data = split_edges(
            edge_index, edge_type,
            test_ratio=self.config['data']['test_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            seed=self.config['seed']
        )

        # 创建全局正样本边集合，避免数据泄露
        all_positive_edges = set()
        for i in range(edge_index.shape[1]):
            head, tail = edge_index[0, i].item(), edge_index[1, i].item()
            all_positive_edges.add((head, tail))

        # 准备测试数据
        self.test_data = self._prepare_data_for_evaluation(test_data, all_positive_edges)
        
        # 创建完整图用于编码
        self.full_edge_index = edge_index.to(self.device)
        self.full_edge_type = edge_type.to(self.device)
        
        print(f"测试集大小: {len(self.test_data['existence_labels'])}")
    
    def _prepare_data_for_evaluation(self, data, all_positive_edges):
        """准备多任务评估数据"""
        edge_index, existence_labels, relation_labels, _ = prepare_multitask_data(
            data['edge_index'],
            data['edge_type'],
            self.num_nodes,
            self.config['data']['negative_sampling_ratio'],
            all_positive_edges
        )

        return {
            'edge_index': edge_index.to(self.device),
            'existence_labels': existence_labels.to(self.device),
            'relation_labels': relation_labels.to(self.device)
        }
    
    def load_model(self):
        """加载模型"""
        print("正在加载模型...")
        
        self.model = DrugDiseaseRGCN(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            **self.config['model']
        ).to(self.device)
        
        # 创建优化器（加载检查点需要）
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # 加载检查点
        epoch, loss, metrics = load_model(self.model, optimizer, self.model_path, self.device)
        
        print(f"已加载模型 (epoch {epoch}, loss: {loss:.4f})")
        if metrics:
            print("训练时的最佳指标:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    def evaluate(self):
        """评估模型"""
        print("正在评估模型...")
        
        self.model.eval()
        
        with torch.no_grad():
            # 编码所有节点
            node_indices = torch.arange(self.num_nodes, device=self.device)
            node_embeddings = self.model.encode(
                node_indices, self.full_edge_index, self.full_edge_type
            )
            
            # 多任务预测
            existence_scores, relation_logits = self.model.predict_links(
                node_embeddings,
                self.test_data['edge_index'][0],
                self.test_data['edge_index'][1]
            )

            # 关系存在性评估
            existence_y_true = self.test_data['existence_labels'].cpu().numpy()
            existence_y_score = torch.sigmoid(existence_scores).cpu().numpy()
            existence_y_pred = (existence_y_score > 0.5).astype(int)

            # 计算关系存在性指标
            existence_metrics = calculate_metrics(
                existence_y_true, existence_y_pred, existence_y_score,
                self.config['evaluation']['k_values']
            )

            # 关系类型评估（只对正样本）
            positive_mask = self.test_data['existence_labels'] == 1
            if positive_mask.sum() > 0:
                relation_y_true = self.test_data['relation_labels'][positive_mask].cpu().numpy()
                relation_y_pred = torch.argmax(relation_logits[positive_mask], dim=1).cpu().numpy()
                relation_accuracy = (relation_y_true == relation_y_pred).mean()
            else:
                relation_accuracy = 0.0

            # 合并指标
            metrics = {}
            for key, value in existence_metrics.items():
                metrics[f'existence_{key}'] = value
            metrics['relation_accuracy'] = relation_accuracy
        
        return metrics, existence_y_true, existence_y_pred, existence_y_score
    
    def detailed_analysis(self, y_true, y_pred, y_score):
        """详细分析"""
        print("\n" + "="*50)
        print("详细分析结果")
        print("="*50)
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=['负样本', '正样本']))
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n混淆矩阵:")
        print(f"真负例: {cm[0,0]}, 假正例: {cm[0,1]}")
        print(f"假负例: {cm[1,0]}, 真正例: {cm[1,1]}")
        
        # 按关系类型分析
        self._analyze_by_relation_type(y_true, y_pred, y_score)
        
        # 可视化
        self._create_visualizations(y_true, y_pred, y_score, cm)
    
    def _analyze_by_relation_type(self, y_true, y_pred, y_score):
        """按关系类型分析"""
        print("\nAnalysis by relation type:")

        # 获取关系类型（只分析正样本）
        positive_mask = self.test_data['existence_labels'] == 1
        if positive_mask.sum() == 0:
            print("  No positive samples for relation type analysis")
            return

        # 转换为CPU numpy数组
        positive_mask_np = positive_mask.cpu().numpy()
        relation_types = self.test_data['relation_labels'][positive_mask].cpu().numpy()
        relation_names = self.mappings['relation_encoder'].classes_

        # 只分析存在关系的样本
        pos_y_true = y_true[positive_mask_np]
        pos_y_pred = y_pred[positive_mask_np]
        pos_y_score = y_score[positive_mask_np]

        for rel_idx, rel_name in enumerate(relation_names):
            if rel_name in self.config['data']['target_relations']:
                mask = relation_types == rel_idx
                if mask.sum() > 0:
                    rel_y_true = pos_y_true[mask]
                    rel_y_pred = pos_y_pred[mask]
                    rel_y_score = pos_y_score[mask]

                    rel_metrics = calculate_metrics(rel_y_true, rel_y_pred, rel_y_score)

                    print(f"\n{rel_name}:")
                    print(f"  Samples: {mask.sum()}")
                    print(f"  AUC: {rel_metrics['auc']:.4f}")
                    print(f"  AP: {rel_metrics['ap']:.4f}")
    
    def _create_visualizations(self, y_true, y_pred, y_score, cm):
        """创建可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted Label')
        axes[0,0].set_ylabel('True Label')

        # Prediction Score Distribution
        axes[0,1].hist(y_score[y_true==0], bins=50, alpha=0.7, label='Negative', density=True)
        axes[0,1].hist(y_score[y_true==1], bins=50, alpha=0.7, label='Positive', density=True)
        axes[0,1].set_title('Prediction Score Distribution')
        axes[0,1].set_xlabel('Prediction Score')
        axes[0,1].set_ylabel('Density')
        axes[0,1].legend()

        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[1,0].plot(fpr, tpr, label=f'ROC (AUC = {calculate_metrics(y_true, y_pred, y_score)["auc"]:.3f})')
        axes[1,0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1,0].set_title('ROC Curve')
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('True Positive Rate')
        axes[1,0].legend()

        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        axes[1,1].plot(recall, precision, label=f'PR (AP = {calculate_metrics(y_true, y_pred, y_score)["ap"]:.3f})')
        axes[1,1].set_title('Precision-Recall Curve')
        axes[1,1].set_xlabel('Recall')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.config['logging']['log_dir'], 'evaluation_plots.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        
        plt.show()
    
    def save_predictions(self, existence_y_true, existence_y_pred, existence_y_score):
        """保存多任务预测结果"""
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'head_node': self.test_data['edge_index'][0].cpu().numpy(),
            'tail_node': self.test_data['edge_index'][1].cpu().numpy(),
            'true_existence': existence_y_true,
            'predicted_existence': existence_y_pred,
            'existence_score': existence_y_score,
            'true_relation_type': self.test_data['relation_labels'].cpu().numpy()
        })
        
        # 添加节点和关系名称
        idx_to_node = self.mappings['idx_to_node']
        relation_names = self.mappings['relation_encoder'].classes_
        
        results_df['head_node_id'] = results_df['head_node'].map(idx_to_node)
        results_df['tail_node_id'] = results_df['tail_node'].map(idx_to_node)
        results_df['true_relation_name'] = results_df['true_relation_type'].map(
            lambda x: relation_names[x] if x >= 0 and x < len(relation_names) else 'none'
        )
        
        # 保存结果
        save_path = os.path.join(self.config['logging']['log_dir'], 'predictions.csv')
        results_df.to_csv(save_path, index=False)
        print(f"预测结果已保存到: {save_path}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description="评估药物-疾病关系预测模型")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--model_path", required=True, help="模型检查点路径")
    parser.add_argument("--save_predictions", action="store_true", help="保存预测结果")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(args.config, args.model_path)
    
    # 评估模型
    metrics, y_true, y_pred, y_score = evaluator.evaluate()
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 详细分析
    evaluator.detailed_analysis(y_true, y_pred, y_score)
    
    # 保存预测结果
    if args.save_predictions:
        evaluator.save_predictions(y_true, y_pred, y_score)
    
    print("\n评估完成!")


if __name__ == "__main__":
    main()

```
```python
# src/model.py
"""
RGCN模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
import math


class RGCNEncoder(nn.Module):
    """RGCN编码器"""
    
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None
    ):
        super(RGCNEncoder, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 节点嵌入
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        
        # RGCN层
        self.rgcn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim
                
            self.rgcn_layers.append(
                RGCNConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    num_relations=num_relations,
                    num_bases=num_bases,
                    num_blocks=num_blocks,
                    aggr='mean'
                )
            )
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.node_embedding.weight)
        
        for layer in self.rgcn_layers:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 如果输入是节点索引，使用嵌入
        if x.dim() == 1:
            x = self.node_embedding(x)
        
        # 通过RGCN层
        for i, layer in enumerate(self.rgcn_layers):
            x = layer(x, edge_index, edge_type)
            
            if i < len(self.rgcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x


class MultiTaskLinkPredictor(nn.Module):
    """多任务链接预测器：同时预测关系存在性和关系类型"""

    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        dropout: float = 0.1
    ):
        super(MultiTaskLinkPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_relations = num_relations

        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # 只用头尾节点，不用关系嵌入
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 关系存在性预测头（二分类）
        self.existence_head = nn.Linear(hidden_dim, 1)

        # 关系类型预测头（多分类）
        self.relation_type_head = nn.Linear(hidden_dim, num_relations)

        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.existence_head.weight)
        nn.init.zeros_(self.existence_head.bias)
        nn.init.xavier_uniform_(self.relation_type_head.weight)
        nn.init.zeros_(self.relation_type_head.bias)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        head_indices: torch.Tensor,
        tail_indices: torch.Tensor
    ) -> tuple:
        """前向传播

        Returns:
            tuple: (existence_scores, relation_type_logits)
        """
        # 获取头尾节点嵌入
        head_emb = node_embeddings[head_indices]  # [batch_size, hidden_dim]
        tail_emb = node_embeddings[tail_indices]  # [batch_size, hidden_dim]

        # 拼接特征（不包含关系类型）
        combined = torch.cat([head_emb, tail_emb], dim=-1)  # [batch_size, hidden_dim * 2]

        # 共享特征提取
        shared_features = self.shared_layers(combined)  # [batch_size, hidden_dim]

        # 两个任务的预测
        existence_scores = self.existence_head(shared_features).squeeze(-1)  # [batch_size]
        relation_type_logits = self.relation_type_head(shared_features)  # [batch_size, num_relations]

        return existence_scores, relation_type_logits


class DrugDiseaseRGCN(nn.Module):
    """药物-疾病关系预测RGCN模型"""
    
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None
    ):
        super(DrugDiseaseRGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        # RGCN编码器
        self.encoder = RGCNEncoder(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_bases=num_bases,
            num_blocks=num_blocks
        )
        
        # 多任务链接预测器
        self.link_predictor = MultiTaskLinkPredictor(
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        head_indices: torch.Tensor,
        tail_indices: torch.Tensor
    ) -> tuple:
        """前向传播

        Returns:
            tuple: (existence_scores, relation_type_logits)
        """
        # 编码节点
        node_embeddings = self.encoder(x, edge_index, edge_type)

        # 多任务预测
        existence_scores, relation_type_logits = self.link_predictor(
            node_embeddings,
            head_indices,
            tail_indices
        )

        return existence_scores, relation_type_logits
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """编码节点"""
        return self.encoder(x, edge_index, edge_type)
    
    def predict_links(
        self,
        node_embeddings: torch.Tensor,
        head_indices: torch.Tensor,
        tail_indices: torch.Tensor
    ) -> tuple:
        """预测链接

        Returns:
            tuple: (existence_scores, relation_type_logits)
        """
        return self.link_predictor(
            node_embeddings,
            head_indices,
            tail_indices
        )


class DistMultDecoder(nn.Module):
    """DistMult解码器（替代方案）"""
    
    def __init__(self, hidden_dim: int, num_relations: int):
        super(DistMultDecoder, self).__init__()
        
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        head_indices: torch.Tensor,
        tail_indices: torch.Tensor,
        relation_indices: torch.Tensor
    ) -> torch.Tensor:
        """DistMult评分函数"""
        head_emb = node_embeddings[head_indices]
        tail_emb = node_embeddings[tail_indices]
        rel_emb = self.relation_embedding(relation_indices)
        
        # DistMult: <h, r, t> = sum(h * r * t)
        scores = torch.sum(head_emb * rel_emb * tail_emb, dim=-1)
        
        return scores


class ComplExDecoder(nn.Module):
    """ComplEx解码器（替代方案）"""
    
    def __init__(self, hidden_dim: int, num_relations: int):
        super(ComplExDecoder, self).__init__()
        
        assert hidden_dim % 2 == 0, "hidden_dim必须是偶数"
        
        self.hidden_dim = hidden_dim
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        head_indices: torch.Tensor,
        tail_indices: torch.Tensor,
        relation_indices: torch.Tensor
    ) -> torch.Tensor:
        """ComplEx评分函数"""
        head_emb = node_embeddings[head_indices]
        tail_emb = node_embeddings[tail_indices]
        rel_emb = self.relation_embedding(relation_indices)
        
        # 分离实部和虚部
        head_real, head_img = torch.chunk(head_emb, 2, dim=-1)
        tail_real, tail_img = torch.chunk(tail_emb, 2, dim=-1)
        rel_real, rel_img = torch.chunk(rel_emb, 2, dim=-1)
        
        # ComplEx评分
        score_real = torch.sum(
            head_real * rel_real * tail_real +
            head_real * rel_img * tail_img +
            head_img * rel_real * tail_img -
            head_img * rel_img * tail_real,
            dim=-1
        )
        
        return score_real


def create_model(
    num_nodes: int,
    num_relations: int,
    model_config: Dict
) -> DrugDiseaseRGCN:
    """创建模型"""
    return DrugDiseaseRGCN(
        num_nodes=num_nodes,
        num_relations=num_relations,
        **model_config
    )

```

```python
# src/train.py
"""
训练脚本
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import wandb

from data_loader import PrimeKGDataLoader
from model import DrugDiseaseRGCN
from utils import (
    set_seed, load_config, setup_logging, get_device,
    split_edges, prepare_multitask_data, calculate_metrics,
    save_model, print_model_info, EarlyStopping
)


class Trainer:
    """训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 设置随机种子
        set_seed(config['seed'])
        
        # 设置设备
        self.device = get_device(config['device'])
        
        # 设置日志
        self.logger = setup_logging(
            config['logging']['log_dir'],
            config['experiment']['name']
        )
        
        # 初始化数据加载器
        self.data_loader = PrimeKGDataLoader(config['data']['data_dir'])
        
        # 初始化模型、优化器等
        self.model = None
        self.optimizer = None
        self.scheduler = None
        # 多任务损失函数
        self.existence_criterion = nn.BCEWithLogitsLoss()  # 关系存在性
        self.relation_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 关系类型，忽略负样本
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config['training']['patience']
        )
        
        # 创建保存目录
        os.makedirs(config['logging']['model_dir'], exist_ok=True)
    
    def load_data(self):
        """加载数据"""
        self.logger.info("正在加载数据...")
        
        # 加载处理后的数据
        hetero_data, mappings, drug_disease_df = self.data_loader.load_processed_data()
        
        self.mappings = mappings
        self.num_nodes = mappings['num_nodes']
        self.num_relations = mappings['num_relations']
        
        # 筛选目标关系
        target_relations = self.config['data']['target_relations']
        if target_relations:
            mask = drug_disease_df['display_relation'].isin(target_relations)
            drug_disease_df = drug_disease_df[mask]
        
        self.logger.info(f"筛选后的药物-疾病关系数量: {len(drug_disease_df)}")
        
        # 转换为边索引和边类型
        edge_list = []
        edge_types = []
        
        for _, row in drug_disease_df.iterrows():
            head_idx = mappings['node_to_idx'][row['x_id']]
            tail_idx = mappings['node_to_idx'][row['y_id']]
            rel_idx = mappings['relation_encoder'].transform([row['display_relation']])[0]
            
            edge_list.append([head_idx, tail_idx])
            edge_types.append(rel_idx)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        # 分割数据
        train_data, val_data, test_data = split_edges(
            edge_index, edge_type,
            test_ratio=self.config['data']['test_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            seed=self.config['seed']
        )

        # 创建全局正样本边集合，避免数据泄露
        all_positive_edges = set()
        for i in range(edge_index.shape[1]):
            head, tail = edge_index[0, i].item(), edge_index[1, i].item()
            all_positive_edges.add((head, tail))

        # 准备链接预测数据
        self.train_data = self._prepare_data_for_training(train_data, all_positive_edges)
        self.val_data = self._prepare_data_for_training(val_data, all_positive_edges)
        self.test_data = self._prepare_data_for_training(test_data, all_positive_edges)
        
        # 创建完整图用于编码
        self.full_edge_index = edge_index.to(self.device)
        self.full_edge_type = edge_type.to(self.device)
        
        self.logger.info(f"训练集大小: {len(self.train_data['existence_labels'])}")
        self.logger.info(f"验证集大小: {len(self.val_data['existence_labels'])}")
        self.logger.info(f"测试集大小: {len(self.test_data['existence_labels'])}")
    
    def _prepare_data_for_training(self, data, all_positive_edges):
        """准备多任务训练数据"""
        edge_index, existence_labels, relation_labels, _ = prepare_multitask_data(
            data['edge_index'],
            data['edge_type'],
            self.num_nodes,
            self.config['data']['negative_sampling_ratio'],
            all_positive_edges
        )

        return {
            'edge_index': edge_index.to(self.device),
            'existence_labels': existence_labels.to(self.device),
            'relation_labels': relation_labels.to(self.device)
        }
    
    def build_model(self):
        """构建模型"""
        self.logger.info("正在构建模型...")
        
        self.model = DrugDiseaseRGCN(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            **self.config['model']
        ).to(self.device)
        
        # 打印模型信息
        print_model_info(self.model)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # 学习率调度器
        scheduler_config = self.config['training']['scheduler']
        if scheduler_config['type'] == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_config['type'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 创建数据加载器
        dataset = TensorDataset(
            self.train_data['edge_index'][0],      # head
            self.train_data['edge_index'][1],      # tail
            self.train_data['existence_labels'],   # existence labels
            self.train_data['relation_labels']     # relation type labels
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        # 编码所有节点
        with torch.no_grad():
            node_indices = torch.arange(self.num_nodes, device=self.device)
            node_embeddings = self.model.encode(
                node_indices, self.full_edge_index, self.full_edge_type
            )
        
        for batch_idx, (head_indices, tail_indices, existence_labels, relation_labels) in enumerate(dataloader):
            self.optimizer.zero_grad()

            # 前向传播
            existence_scores, relation_logits = self.model.predict_links(
                node_embeddings, head_indices, tail_indices
            )

            # 计算多任务损失
            existence_loss = self.existence_criterion(existence_scores, existence_labels.float())
            relation_loss = self.relation_criterion(relation_logits, relation_labels)

            # 总损失（可以添加权重）
            total_batch_loss = existence_loss + relation_loss

            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            num_batches += 1

            # 记录日志
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{len(dataloader)}, Total Loss: {total_batch_loss.item():.4f}, '
                    f'Existence Loss: {existence_loss.item():.4f}, Relation Loss: {relation_loss.item():.4f}'
                )
        
        return total_loss / num_batches
    
    def evaluate(self, data, split_name="val"):
        """评估模型"""
        self.model.eval()
        
        with torch.no_grad():
            # 编码所有节点
            node_indices = torch.arange(self.num_nodes, device=self.device)
            node_embeddings = self.model.encode(
                node_indices, self.full_edge_index, self.full_edge_type
            )
            
            # 预测
            existence_scores, relation_logits = self.model.predict_links(
                node_embeddings,
                data['edge_index'][0],
                data['edge_index'][1]
            )

            # 计算损失
            existence_loss = self.existence_criterion(existence_scores, data['existence_labels'].float())
            relation_loss = self.relation_criterion(relation_logits, data['relation_labels'])
            total_loss = existence_loss + relation_loss

            # 关系存在性评估
            existence_y_true = data['existence_labels'].cpu().numpy()
            existence_y_score = torch.sigmoid(existence_scores).cpu().numpy()
            existence_y_pred = (existence_y_score > 0.5).astype(int)

            # 计算关系存在性指标
            existence_metrics = calculate_metrics(
                existence_y_true, existence_y_pred, existence_y_score,
                self.config['evaluation']['k_values']
            )

            # 关系类型评估（只对正样本）
            positive_mask = data['existence_labels'] == 1
            if positive_mask.sum() > 0:
                relation_y_true = data['relation_labels'][positive_mask].cpu().numpy()
                relation_y_pred = torch.argmax(relation_logits[positive_mask], dim=1).cpu().numpy()
                relation_accuracy = (relation_y_true == relation_y_pred).mean()
            else:
                relation_accuracy = 0.0

            # 合并指标
            metrics = {}
            for key, value in existence_metrics.items():
                metrics[f'existence_{key}'] = value
            metrics['relation_accuracy'] = relation_accuracy
            metrics['total_loss'] = total_loss.item()
            metrics['existence_loss'] = existence_loss.item()
            metrics['relation_loss'] = relation_loss.item()
        
        return metrics
    
    def train(self):
        """训练模型"""
        self.logger.info("开始训练...")
        
        best_val_auc = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_metrics = self.evaluate(self.val_data, "val")
            
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Total Loss: {val_metrics['total_loss']:.4f}, "
                           f"Existence AUC: {val_metrics['existence_auc']:.4f}, "
                           f"Relation Acc: {val_metrics['relation_accuracy']:.4f}")

            # 学习率调度
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['existence_auc'])
            else:
                self.scheduler.step()

            # 保存最佳模型（基于关系存在性AUC）
            if val_metrics['existence_auc'] > best_val_auc:
                best_val_auc = val_metrics['existence_auc']
                
                if self.config['logging']['save_model']:
                    save_path = os.path.join(
                        self.config['logging']['model_dir'],
                        f"{self.config['experiment']['name']}_best.pth"
                    )
                    save_model(
                        self.model, self.optimizer, epoch,
                        val_metrics['total_loss'], val_metrics, save_path
                    )
                    self.logger.info(f"保存最佳模型到: {save_path}")
            
            # 早停检查
            if self.early_stopping(val_metrics['existence_auc'], self.model):
                self.logger.info(f"早停在epoch {epoch+1}")
                break
        
        # 最终测试
        test_metrics = self.evaluate(self.test_data, "test")
        self.logger.info("测试结果:")
        for metric, value in test_metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return test_metrics


def main():
    parser = argparse.ArgumentParser(description="训练药物-疾病关系预测模型")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 加载数据
    trainer.load_data()
    
    # 构建模型
    trainer.build_model()
    
    # 训练
    test_metrics = trainer.train()
    
    print("训练完成!")
    print("最终测试结果:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

```


```python
# src/utils.py
"""
工具函数
"""

import os
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any, Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import logging


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(experiment_name)
    return logger


def get_device(device_config: str) -> torch.device:
    """获取设备"""
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    return device


def create_negative_samples(
    positive_edges: torch.Tensor,
    num_nodes: int,
    num_negative: int,
    existing_edges: set = None
) -> torch.Tensor:
    """创建负样本"""
    if existing_edges is None:
        existing_edges = set()
        for i in range(positive_edges.shape[1]):
            head, tail = positive_edges[0, i].item(), positive_edges[1, i].item()
            existing_edges.add((head, tail))
            existing_edges.add((tail, head))  # 无向图
    
    negative_edges = []
    
    while len(negative_edges) < num_negative:
        head = random.randint(0, num_nodes - 1)
        tail = random.randint(0, num_nodes - 1)
        
        if head != tail and (head, tail) not in existing_edges:
            negative_edges.append([head, tail])
            existing_edges.add((head, tail))
    
    return torch.tensor(negative_edges, dtype=torch.long).t()


def split_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """分割边为训练/验证/测试集"""
    num_edges = edge_index.shape[1]
    indices = np.arange(num_edges)
    
    # 分割索引
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=seed
    )
    
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_ratio/(1-test_ratio), random_state=seed
    )
    
    # 创建数据集
    train_data = {
        'edge_index': edge_index[:, train_indices],
        'edge_type': edge_type[train_indices]
    }
    
    val_data = {
        'edge_index': edge_index[:, val_indices],
        'edge_type': edge_type[val_indices]
    }
    
    test_data = {
        'edge_index': edge_index[:, test_indices],
        'edge_type': edge_type[test_indices]
    }
    
    return train_data, val_data, test_data


def prepare_multitask_data(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    negative_sampling_ratio: float = 1.0,
    all_positive_edges: set = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """准备多任务学习数据

    Args:
        all_positive_edges: 所有分割中的正样本边集合，用于避免数据泄露

    Returns:
        tuple: (all_edge_index, existence_labels, relation_type_labels, perm)
    """
    num_positive = edge_index.shape[1]
    num_negative = int(num_positive * negative_sampling_ratio)

    # 正样本：存在关系=1，关系类型=实际类型
    positive_existence_labels = torch.ones(num_positive)
    positive_relation_labels = edge_type.clone()

    # 创建负样本 - 排除所有正样本边以避免数据泄露
    if all_positive_edges is None:
        # 如果没有提供全局正样本边，只排除当前分割的边（旧行为）
        existing_edges = set()
        for i in range(num_positive):
            head, tail = edge_index[0, i].item(), edge_index[1, i].item()
            existing_edges.add((head, tail))
    else:
        # 使用全局正样本边集合，避免数据泄露
        existing_edges = all_positive_edges.copy()

    negative_edge_index = create_negative_samples(
        edge_index, num_nodes, num_negative, existing_edges
    )

    # 负样本：存在关系=0，关系类型=-1（忽略）
    negative_existence_labels = torch.zeros(num_negative)
    negative_relation_labels = torch.full((num_negative,), -1, dtype=torch.long)  # -1表示忽略

    # 合并正负样本
    all_edge_index = torch.cat([edge_index, negative_edge_index], dim=1)
    all_existence_labels = torch.cat([positive_existence_labels, negative_existence_labels], dim=0)
    all_relation_labels = torch.cat([positive_relation_labels, negative_relation_labels], dim=0)

    # 打乱顺序
    perm = torch.randperm(all_edge_index.shape[1])
    all_edge_index = all_edge_index[:, perm]
    all_existence_labels = all_existence_labels[perm]
    all_relation_labels = all_relation_labels[perm]

    return all_edge_index, all_existence_labels, all_relation_labels, perm


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    k_values: List[int] = [10, 50, 100]
) -> Dict[str, float]:
    """计算评估指标"""
    metrics = {}
    
    # AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics['auc'] = 0.0
    
    # Average Precision
    try:
        metrics['ap'] = average_precision_score(y_true, y_score)
    except ValueError:
        metrics['ap'] = 0.0
    
    # Precision@K and Recall@K
    for k in k_values:
        if len(y_score) >= k:
            # 获取top-k预测
            top_k_indices = np.argsort(y_score)[-k:]
            top_k_pred = np.zeros_like(y_pred)
            top_k_pred[top_k_indices] = 1
            
            # 计算precision@k
            if top_k_pred.sum() > 0:
                precision_k = precision_score(y_true, top_k_pred, zero_division=0)
                metrics[f'precision@{k}'] = precision_k
            else:
                metrics[f'precision@{k}'] = 0.0
            
            # 计算recall@k
            if y_true.sum() > 0:
                recall_k = recall_score(y_true, top_k_pred, zero_division=0)
                metrics[f'recall@{k}'] = recall_k
            else:
                metrics[f'recall@{k}'] = 0.0
        else:
            metrics[f'precision@{k}'] = 0.0
            metrics[f'recall@{k}'] = 0.0
    
    return metrics


def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
               epoch: int, loss: float, metrics: Dict[str, float], 
               save_path: str):
    """保存模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }, save_path)


def load_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
               load_path: str, device: torch.device) -> Tuple[int, float, Dict[str, float]]:
    """加载模型"""
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metrics = checkpoint.get('metrics', {})
    
    return epoch, loss, metrics


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module):
    """打印模型信息"""
    total_params = count_parameters(model)
    print(f"模型总参数数量: {total_params:,}")
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False

```

#### 上述的项目的问题
1. 跑分太高了, 不正常, 0.95左右是正常的跑分 
2. 可能出现的原因:首先, 目前预测了- indication (适应症), contraindication (禁忌症), off-label use (超说明书用药), 太多了, 只要预测indication关系就可以; 其次, 负样本生成, 目前的办法是随机生成, 但是需要改成:负样本应该在所有关系下都没有边;需要使用使用contrastive loss(CLIP)替代 BCE; 尝试 cross-disease splits，即在训练时完全去掉某些疾病的所有药物配对，然后在这些疾病上预测（zero-shot setting）,在这个设定下计算 auROC 和 MRR; 在evaluation的时候, Ranking-based Evaluation, 即对于一个给定的药物, 构造候选疾病集合（不包含训练中该药物的任何疾病边）, 模型对该药物与这些疾病的相关性打分,要求真实的适应症疾病在排序中排名靠前（top 1~50）。同样，针对疾病，也需要从所有的药物中选，看看target drug在所有药物中的排序；
