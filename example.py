#!/usr/bin/env python3
"""
PrimeKG药物疾病关系预测示例
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.append('src')

from data_loader import PrimeKGDataLoader
from model import DrugDiseaseRGCN
from utils import set_seed, calculate_metrics


def example_1_data_loading():
    """示例1: 数据加载和探索"""
    print("=" * 60)
    print("示例1: 数据加载和探索")
    print("=" * 60)
    
    # 初始化数据加载器
    loader = PrimeKGDataLoader("data")
    
    # 检查是否已有处理后的数据
    processed_file = "data/processed/processed_data.pkl"
    if not os.path.exists(processed_file):
        print("首次运行，正在下载和处理数据...")
        # 下载数据（如果需要）
        loader.download_primekg()
        # 处理数据
        hetero_data, mappings, drug_disease_df = loader.process_data()
    else:
        print("加载已处理的数据...")
        hetero_data, mappings, drug_disease_df = loader.load_processed_data()
    
    # 数据统计
    print(f"\n数据统计:")
    print(f"- 总节点数: {mappings['num_nodes']:,}")
    print(f"- 总关系类型数: {mappings['num_relations']}")
    print(f"- 药物-疾病关系数: {len(drug_disease_df):,}")
    
    # 关系类型分布
    print(f"\n药物-疾病关系类型分布:")
    relation_counts = drug_disease_df['display_relation'].value_counts()
    for relation, count in relation_counts.items():
        print(f"- {relation}: {count:,}")
    
    return hetero_data, mappings, drug_disease_df


def example_2_simple_prediction():
    """示例2: 简单的关系预测"""
    print("\n" + "=" * 60)
    print("示例2: 简单的关系预测")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(42)
    
    # 加载数据
    loader = PrimeKGDataLoader("data")
    hetero_data, mappings, drug_disease_df = loader.load_processed_data()
    
    # 创建简单的训练数据
    # 选择前1000个药物-疾病关系作为示例
    sample_df = drug_disease_df.head(1000)
    
    # 转换为模型输入格式
    edge_list = []
    edge_types = []
    
    for _, row in sample_df.iterrows():
        head_idx = mappings['node_to_idx'][row['x_id']]
        tail_idx = mappings['node_to_idx'][row['y_id']]
        rel_idx = mappings['relation_encoder'].transform([row['display_relation']])[0]
        
        edge_list.append([head_idx, tail_idx])
        edge_types.append(rel_idx)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    print(f"示例数据: {edge_index.shape[1]} 条边")
    
    # 创建简单模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = DrugDiseaseRGCN(
        num_nodes=mappings['num_nodes'],
        num_relations=mappings['num_relations'],
        hidden_dim=64,  # 较小的维度用于快速演示
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 简单的前向传播测试
    model.eval()
    with torch.no_grad():
        # 编码节点
        node_indices = torch.arange(min(1000, mappings['num_nodes']), device=device)
        edge_index_sample = edge_index[:, :100].to(device)  # 使用前100条边
        edge_type_sample = edge_type[:100].to(device)
        
        try:
            node_embeddings = model.encode(node_indices, edge_index_sample, edge_type_sample)
            print(f"节点嵌入形状: {node_embeddings.shape}")
            
            # 预测示例
            head_indices = edge_index_sample[0, :10]
            tail_indices = edge_index_sample[1, :10]
            relation_indices = edge_type_sample[:10]
            
            scores = model.predict_links(
                node_embeddings, head_indices, tail_indices, relation_indices
            )
            print(f"预测分数: {scores[:5].cpu().numpy()}")
            
        except Exception as e:
            print(f"模型测试出错: {e}")
            print("这可能是由于数据格式或设备问题，请检查配置")


def example_3_model_comparison():
    """示例3: 不同模型配置比较"""
    print("\n" + "=" * 60)
    print("示例3: 不同模型配置比较")
    print("=" * 60)
    
    # 加载数据
    loader = PrimeKGDataLoader("data")
    hetero_data, mappings, drug_disease_df = loader.load_processed_data()
    
    # 不同的模型配置
    configs = [
        {"hidden_dim": 32, "num_layers": 1, "name": "Small"},
        {"hidden_dim": 64, "num_layers": 2, "name": "Medium"},
        {"hidden_dim": 128, "num_layers": 2, "name": "Large"},
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("模型配置比较:")
    print("-" * 50)
    
    for config in configs:
        model = DrugDiseaseRGCN(
            num_nodes=mappings['num_nodes'],
            num_relations=mappings['num_relations'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=0.1
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        memory_mb = param_count * 4 / (1024 * 1024)  # 假设float32
        
        print(f"{config['name']} 模型:")
        print(f"  - 隐藏维度: {config['hidden_dim']}")
        print(f"  - 层数: {config['num_layers']}")
        print(f"  - 参数数量: {param_count:,}")
        print(f"  - 估计内存: {memory_mb:.1f} MB")
        print()


def example_4_data_analysis():
    """示例4: 数据分析"""
    print("\n" + "=" * 60)
    print("示例4: 数据分析")
    print("=" * 60)
    
    # 加载数据
    loader = PrimeKGDataLoader("data")
    hetero_data, mappings, drug_disease_df = loader.load_processed_data()
    
    # 节点度分析
    print("节点度分析:")
    
    # 计算节点度
    node_degrees = {}
    for _, row in drug_disease_df.iterrows():
        x_id, y_id = row['x_id'], row['y_id']
        node_degrees[x_id] = node_degrees.get(x_id, 0) + 1
        node_degrees[y_id] = node_degrees.get(y_id, 0) + 1
    
    degrees = list(node_degrees.values())
    print(f"- 平均度: {np.mean(degrees):.2f}")
    print(f"- 度中位数: {np.median(degrees):.2f}")
    print(f"- 最大度: {max(degrees)}")
    print(f"- 最小度: {min(degrees)}")
    
    # 高度节点
    top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n度最高的10个节点:")
    for node_id, degree in top_nodes:
        # 查找节点类型
        node_type = "unknown"
        for _, row in drug_disease_df.iterrows():
            if row['x_id'] == node_id:
                node_type = row['x_type']
                break
            elif row['y_id'] == node_id:
                node_type = row['y_type']
                break
        print(f"- {node_id} ({node_type}): {degree}")
    
    # 关系分析
    print(f"\n关系分析:")
    relation_stats = drug_disease_df.groupby('display_relation').agg({
        'x_id': 'nunique',
        'y_id': 'nunique'
    }).rename(columns={'x_id': 'unique_sources', 'y_id': 'unique_targets'})
    
    for relation, stats in relation_stats.iterrows():
        count = len(drug_disease_df[drug_disease_df['display_relation'] == relation])
        print(f"- {relation}:")
        print(f"  关系数量: {count}")
        print(f"  唯一源节点: {stats['unique_sources']}")
        print(f"  唯一目标节点: {stats['unique_targets']}")


def example_5_custom_training():
    """示例5: 自定义训练循环"""
    print("\n" + "=" * 60)
    print("示例5: 自定义训练循环（演示）")
    print("=" * 60)
    
    # 这里只演示训练循环的结构，不实际训练
    print("训练循环结构演示:")
    print("""
    # 1. 数据准备
    loader = PrimeKGDataLoader("data")
    hetero_data, mappings, drug_disease_df = loader.load_processed_data()
    
    # 2. 模型初始化
    model = DrugDiseaseRGCN(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 3. 训练循环
    for epoch in range(num_epochs):
        model.train()
        
        # 前向传播
        node_embeddings = model.encode(...)
        scores = model.predict_links(...)
        
        # 计算损失
        loss = criterion(scores, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 验证
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_scores = model.predict_links(...)
                val_metrics = calculate_metrics(...)
                print(f"Epoch {epoch}, Val AUC: {val_metrics['auc']:.4f}")
    """)
    
    print("完整的训练代码请参考 src/train.py")


def main():
    """主函数"""
    print("PrimeKG药物疾病关系预测示例")
    print("=" * 60)
    
    # 检查环境
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
    
    # 创建必要的目录
    Path("data").mkdir(exist_ok=True)
    Path("data/raw").mkdir(exist_ok=True)
    Path("data/processed").mkdir(exist_ok=True)
    
    try:
        # 运行示例
        example_1_data_loading()
        example_2_simple_prediction()
        example_3_model_comparison()
        example_4_data_analysis()
        example_5_custom_training()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        print("\n接下来你可以:")
        print("1. 运行完整训练: python src/train.py")
        print("2. 数据探索: jupyter notebook notebooks/data_exploration.ipynb")
        print("3. 快速开始: python quick_start.py")
        print("4. 查看使用指南: USAGE_GUIDE.md")
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        print("\n可能的解决方案:")
        print("1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("2. 检查网络连接（下载数据需要）")
        print("3. 确保有足够的磁盘空间")
        print("4. 查看详细错误信息并参考 USAGE_GUIDE.md")


if __name__ == "__main__":
    main()
