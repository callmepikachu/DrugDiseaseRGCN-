#!/usr/bin/env python3
"""
Stage II 三元关系预测评估脚本
"""

import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# 添加父目录以导入Stage I的模块
sys.path.append('../src')
from data_loader import PrimeKGDataLoader
from utils import get_device

# 导入Stage II模块
from triple_model import TripleRelationRGCN


class TripleRelationEvaluator:
    """三元关系预测评估器"""
    
    def __init__(self, config_path: str, model_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = model_path
        self.device = get_device(self.config['device'])
        
        # 创建结果目录
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "predictions").mkdir(exist_ok=True)
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        
        # 加载数据
        self.load_data()
        
        # 加载模型
        self.load_model()
    
    def load_data(self):
        """加载数据"""
        print("Loading data...")
        
        # 加载Stage I处理的数据
        data_loader = PrimeKGDataLoader(self.config['data']['data_dir'])
        hetero_data, mappings, drug_disease_df = data_loader.load_processed_data()
        
        self.mappings = mappings
        self.num_nodes = mappings['num_nodes']
        self.num_relations = mappings['num_relations']
        
        # 加载三元关系数据
        triple_file = Path("results/triple_relations.csv")
        if not triple_file.exists():
            raise FileNotFoundError("Triple relations data not found. Please run data_analyzer.py first.")
        
        self.triple_df = pd.read_csv(triple_file)
        print(f"Loaded {len(self.triple_df)} triple relations")
        
        # 准备图数据 - 基于三元关系构建
        edges = []
        edge_types = []

        for _, row in self.triple_df.iterrows():
            drug_idx = self.mappings['node_to_idx'][str(row['drug_id'])]
            protein_idx = self.mappings['node_to_idx'][str(row['protein_id'])]
            disease_idx = self.mappings['node_to_idx'][str(row['disease_id'])]

            edges.append([drug_idx, protein_idx])
            edge_types.append(0)
            edges.append([protein_idx, disease_idx])
            edge_types.append(1)

        if len(edges) > 0:
            self.edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
            self.edge_type = torch.tensor(edge_types, dtype=torch.long, device=self.device)
        else:
            self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=self.device).t()
            self.edge_type = torch.tensor([0, 0], dtype=torch.long, device=self.device)
        
        # 准备测试数据
        self.prepare_test_data()
    
    def prepare_test_data(self):
        """准备测试数据"""
        print("Preparing test data...")

        # 创建与训练时相同的节点映射
        all_nodes = set()
        for _, row in self.triple_df.iterrows():
            all_nodes.add(str(row['drug_id']))
            all_nodes.add(str(row['protein_id']))
            all_nodes.add(str(row['disease_id']))

        sorted_nodes = sorted(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}

        # 更新映射信息
        self.mappings['node_to_idx'] = node_to_idx
        self.mappings['idx_to_node'] = {idx: node for node, idx in node_to_idx.items()}
        self.num_nodes = len(node_to_idx)

        print(f"Created mapping for {len(node_to_idx)} unique nodes")

        # 转换所有三元组
        valid_triples = []
        for _, row in self.triple_df.iterrows():
            valid_triples.append({
                'drug_idx': node_to_idx[str(row['drug_id'])],
                'protein_idx': node_to_idx[str(row['protein_id'])],
                'disease_idx': node_to_idx[str(row['disease_id'])],
                'drug_id': row['drug_id'],
                'protein_id': row['protein_id'],
                'disease_id': row['disease_id'],
                'drug_protein_relation': row['drug_protein_relation'],
                'protein_disease_relation': row['protein_disease_relation']
            })

        print(f"Valid triples for evaluation: {len(valid_triples)}")
        
        # 转换为张量
        valid_df = pd.DataFrame(valid_triples)
        
        # 使用后20%作为测试数据
        test_size = int(len(valid_df) * 0.2)
        test_df = valid_df.tail(test_size)
        
        self.test_data = {
            'drug_indices': torch.tensor(test_df['drug_idx'].values, dtype=torch.long, device=self.device),
            'protein_indices': torch.tensor(test_df['protein_idx'].values, dtype=torch.long, device=self.device),
            'disease_indices': torch.tensor(test_df['disease_idx'].values, dtype=torch.long, device=self.device),
            'drug_ids': test_df['drug_id'].values,
            'protein_ids': test_df['protein_id'].values,
            'disease_ids': test_df['disease_id'].values,
            'drug_protein_relations': test_df['drug_protein_relation'].values,
            'protein_disease_relations': test_df['protein_disease_relation'].values
        }
        
        # 创建标签（所有测试样本都是正样本）
        num_test = len(test_df)
        self.test_labels = {
            'existence_labels': torch.ones(num_test, dtype=torch.float, device=self.device),
            'protein_importance': torch.ones(num_test, dtype=torch.float, device=self.device),
            'pathway_labels': torch.randint(0, 10, (num_test,), dtype=torch.long, device=self.device),
            'mechanism_labels': torch.randint(0, 5, (num_test,), dtype=torch.long, device=self.device)
        }
        
        print(f"Test data prepared: {num_test} samples")
    
    def load_model(self):
        """加载模型"""
        print("Loading model...")

        # 加载检查点以获取训练时的配置
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # 从检查点获取训练时的配置
        if 'config' in checkpoint:
            train_config = checkpoint['config']
            print(f"Using training config from checkpoint")
        else:
            train_config = self.config
            print(f"Using current config (no training config in checkpoint)")

        # 创建模型 - 使用与训练时相同的节点数量
        self.model = TripleRelationRGCN(
            num_nodes=self.num_nodes,
            num_relations=max(self.num_relations, 2),
            num_pathways=train_config['model'].get('num_pathways', 100),
            hidden_dim=train_config['model']['hidden_dim'],
            num_layers=train_config['model']['num_layers'],
            fusion_dim=train_config['model']['triple_fusion_dim'],
            dropout=train_config['model']['dropout']
        ).to(self.device)

        print(f"Created model with {self.num_nodes} nodes, {self.num_relations} relations")

        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from epoch {checkpoint['epoch']}")
        if 'val_metrics' in checkpoint:
            print("Training validation metrics:")
            for metric, value in checkpoint['val_metrics'].items():
                print(f"  {metric}: {value:.4f}")
    
    def evaluate_model(self):
        """评估模型"""
        print("Evaluating model...")
        
        self.model.eval()
        
        with torch.no_grad():
            # 编码所有节点
            node_indices = torch.arange(self.num_nodes, device=self.device)

            # 检查模型的节点数量是否匹配
            model_num_nodes = self.model.num_nodes
            if self.num_nodes != model_num_nodes:
                print(f"Warning: Data nodes ({self.num_nodes}) != Model nodes ({model_num_nodes})")
                # 使用较小的数量
                actual_num_nodes = min(self.num_nodes, model_num_nodes)
                node_indices = torch.arange(actual_num_nodes, device=self.device)
                print(f"Using {actual_num_nodes} nodes for evaluation")

            # 检查边索引是否超出范围
            if self.edge_index.numel() > 0:
                max_edge_idx = self.edge_index.max().item()
                if max_edge_idx >= len(node_indices):
                    print(f"Warning: Edge index {max_edge_idx} >= node count {len(node_indices)}")
                    # 过滤边
                    valid_mask = (self.edge_index < len(node_indices)).all(dim=0)
                    self.edge_index = self.edge_index[:, valid_mask]
                    self.edge_type = self.edge_type[valid_mask]
                    print(f"Filtered edges, remaining: {self.edge_index.shape[1]}")

            node_embeddings = self.model.encode(node_indices, self.edge_index, self.edge_type)
            
            # 预测
            predictions = self.model.predict_triple_relations(
                node_embeddings,
                self.test_data['drug_indices'],
                self.test_data['protein_indices'],
                self.test_data['disease_indices']
            )
            
            # 转换预测结果
            existence_scores = torch.sigmoid(predictions['existence_prediction'].squeeze()).cpu().numpy()
            existence_pred = (existence_scores > 0.5).astype(int)
            
            protein_scores = torch.sigmoid(predictions['protein_importance'].squeeze()).cpu().numpy()
            protein_pred = (protein_scores > 0.5).astype(int)
            
            pathway_pred = torch.argmax(predictions['pathway_prediction'], dim=1).cpu().numpy()
            mechanism_pred = torch.argmax(predictions['mechanism_classification'], dim=1).cpu().numpy()
            
            # 获取注意力权重
            attention_weights = predictions['attention_weights'].cpu().numpy()
            
            # 转换真实标签
            existence_true = self.test_labels['existence_labels'].cpu().numpy()
            protein_true = self.test_labels['protein_importance'].cpu().numpy()
            pathway_true = self.test_labels['pathway_labels'].cpu().numpy()
            mechanism_true = self.test_labels['mechanism_labels'].cpu().numpy()
        
        # 计算指标
        metrics = self.calculate_metrics(
            existence_true, existence_pred, existence_scores,
            protein_true, protein_pred, protein_scores,
            pathway_true, pathway_pred,
            mechanism_true, mechanism_pred
        )
        
        return metrics, {
            'existence_scores': existence_scores,
            'existence_pred': existence_pred,
            'protein_scores': protein_scores,
            'pathway_pred': pathway_pred,
            'mechanism_pred': mechanism_pred,
            'attention_weights': attention_weights
        }
    
    def calculate_metrics(self, existence_true, existence_pred, existence_scores,
                         protein_true, protein_pred, protein_scores,
                         pathway_true, pathway_pred,
                         mechanism_true, mechanism_pred):
        """计算评估指标"""
        metrics = {}
        
        # 关系存在性指标
        metrics['existence_accuracy'] = (existence_true == existence_pred).mean()
        metrics['existence_auc'] = roc_auc_score(existence_true, existence_scores)
        metrics['existence_ap'] = average_precision_score(existence_true, existence_scores)
        
        # 蛋白质重要性指标
        metrics['protein_accuracy'] = (protein_true == protein_pred).mean()
        metrics['protein_auc'] = roc_auc_score(protein_true, protein_scores)
        
        # 通路预测指标
        metrics['pathway_accuracy'] = (pathway_true == pathway_pred).mean()
        
        # 机制预测指标
        metrics['mechanism_accuracy'] = (mechanism_true == mechanism_pred).mean()
        
        return metrics
    
    def analyze_attention_patterns(self, attention_weights):
        """分析注意力模式"""
        print("Analyzing attention patterns...")
        
        # 计算平均注意力权重
        # attention_weights shape: [batch_size, num_heads, 3, 3]
        # 3个实体: drug, protein, disease
        
        avg_attention = attention_weights.mean(axis=(0, 1))  # [3, 3]
        
        # 可视化注意力矩阵
        plt.figure(figsize=(8, 6))
        entity_names = ['Drug', 'Protein', 'Disease']
        
        sns.heatmap(avg_attention, 
                   xticklabels=entity_names,
                   yticklabels=entity_names,
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues')
        
        plt.title('Average Attention Weights Between Entities')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'visualizations' / 'attention_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Attention analysis completed")
        
        return avg_attention
    
    def create_prediction_report(self, metrics, predictions):
        """创建预测报告"""
        print("Creating prediction report...")
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'drug_id': self.test_data['drug_ids'],
            'protein_id': self.test_data['protein_ids'],
            'disease_id': self.test_data['disease_ids'],
            'drug_protein_relation': self.test_data['drug_protein_relations'],
            'protein_disease_relation': self.test_data['protein_disease_relations'],
            'existence_score': predictions['existence_scores'],
            'existence_prediction': predictions['existence_pred'],
            'protein_importance_score': predictions['protein_scores'],
            'pathway_prediction': predictions['pathway_pred'],
            'mechanism_prediction': predictions['mechanism_pred']
        })
        
        # 保存到CSV
        results_df.to_csv(self.results_dir / 'predictions' / 'triple_predictions.csv', index=False)
        
        # 创建性能可视化
        self.create_performance_plots(metrics, predictions)
        
        print(f"Prediction report saved to {self.results_dir / 'predictions'}")
    
    def create_performance_plots(self, metrics, predictions):
        """创建性能可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 存在性预测分数分布
        existence_scores = predictions['existence_scores']
        axes[0, 0].hist(existence_scores, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Existence Prediction Score Distribution')
        axes[0, 0].set_xlabel('Prediction Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 0].legend()
        
        # 2. 蛋白质重要性分数分布
        protein_scores = predictions['protein_scores']
        axes[0, 1].hist(protein_scores, bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].set_title('Protein Importance Score Distribution')
        axes[0, 1].set_xlabel('Importance Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 1].legend()
        
        # 3. 通路预测分布
        pathway_pred = predictions['pathway_pred']
        pathway_counts = np.bincount(pathway_pred)
        axes[1, 0].bar(range(len(pathway_counts)), pathway_counts)
        axes[1, 0].set_title('Pathway Prediction Distribution')
        axes[1, 0].set_xlabel('Pathway ID')
        axes[1, 0].set_ylabel('Count')
        
        # 4. 机制预测分布
        mechanism_pred = predictions['mechanism_pred']
        mechanism_counts = np.bincount(mechanism_pred)
        mechanism_names = ['Agonist', 'Antagonist', 'Inhibitor', 'Activator', 'Modulator']
        axes[1, 1].bar(range(len(mechanism_counts)), mechanism_counts)
        axes[1, 1].set_title('Mechanism Prediction Distribution')
        axes[1, 1].set_xlabel('Mechanism Type')
        axes[1, 1].set_ylabel('Count')
        if len(mechanism_names) >= len(mechanism_counts):
            axes[1, 1].set_xticks(range(len(mechanism_counts)))
            axes[1, 1].set_xticklabels(mechanism_names[:len(mechanism_counts)], rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'visualizations' / 'performance_plots.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_evaluation(self):
        """运行完整评估"""
        print("Starting Stage II evaluation...")
        
        # 评估模型
        metrics, predictions = self.evaluate_model()
        
        # 打印结果
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # 分析注意力模式
        attention_matrix = self.analyze_attention_patterns(predictions['attention_weights'])
        
        # 创建预测报告
        self.create_prediction_report(metrics, predictions)
        
        print("\nEvaluation completed!")
        print(f"Results saved to {self.results_dir}")
        
        return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description="Stage II Triple Relation Evaluation")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--model_path", required=True, help="Model checkpoint path")
    
    args = parser.parse_args()
    
    evaluator = TripleRelationEvaluator(args.config, args.model_path)
    metrics, predictions = evaluator.run_evaluation()
    
    print("\nFinal Results Summary:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
