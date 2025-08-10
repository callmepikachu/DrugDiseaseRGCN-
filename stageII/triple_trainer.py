#!/usr/bin/env python3
"""
Stage II 三元关系预测训练脚本
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from tqdm import tqdm

# 添加父目录以导入Stage I的模块
sys.path.append('../src')
from data_loader import PrimeKGDataLoader
from utils import set_seed, get_device, EarlyStopping

# 导入Stage II模块
from triple_model import TripleRelationRGCN


class TripleRelationTrainer:
    """三元关系预测训练器"""
    
    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置随机种子
        set_seed(self.config['seed'])
        
        # 设置设备
        self.device = get_device(self.config['device'])
        
        # 设置日志
        self.setup_logging()
        
        # 创建结果目录
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        
        self.logger.info("TripleRelationTrainer initialized")
    
    def setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'triple_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('triple_trainer')
    
    def load_data(self):
        """加载和准备数据"""
        self.logger.info("Loading data...")
        
        # 加载Stage I处理的数据
        data_loader = PrimeKGDataLoader(self.config['data']['data_dir'])
        hetero_data, mappings, drug_disease_df = data_loader.load_processed_data()
        
        self.mappings = mappings
        self.num_nodes = mappings['num_nodes']
        self.num_relations = mappings['num_relations']
        
        # 加载三元关系数据
        triple_file = Path("results/triple_relations.csv")
        if not triple_file.exists():
            self.logger.error("Triple relations file not found. Please run data_analyzer.py first.")
            raise FileNotFoundError("Triple relations data not found")
        
        self.triple_df = pd.read_csv(triple_file)
        self.logger.info(f"Loaded {len(self.triple_df)} triple relations")
        
        # 准备训练数据
        self.prepare_training_data()

        # 准备图数据（在训练数据准备后，因为需要节点映射）
        self.prepare_graph_data(hetero_data)
    
    def prepare_graph_data(self, hetero_data):
        """准备图数据"""
        # 基于三元关系构建图结构
        edges = []
        edge_types = []

        # 从三元关系中提取边
        for _, row in self.triple_df.iterrows():
            drug_idx = self.mappings['node_to_idx'][str(row['drug_id'])]
            protein_idx = self.mappings['node_to_idx'][str(row['protein_id'])]
            disease_idx = self.mappings['node_to_idx'][str(row['disease_id'])]

            # 添加药物-蛋白质边
            edges.append([drug_idx, protein_idx])
            edge_types.append(0)  # 药物-蛋白质关系类型

            # 添加蛋白质-疾病边
            edges.append([protein_idx, disease_idx])
            edge_types.append(1)  # 蛋白质-疾病关系类型

        # 转换为张量
        if len(edges) > 0:
            self.edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
            self.edge_type = torch.tensor(edge_types, dtype=torch.long, device=self.device)

            # 检查边索引是否超出节点范围
            max_edge_idx = self.edge_index.max().item()
            if max_edge_idx >= self.num_nodes:
                self.logger.error(f"Edge index {max_edge_idx} exceeds node count {self.num_nodes}")
                # 过滤掉超出范围的边
                valid_mask = (self.edge_index < self.num_nodes).all(dim=0)
                self.edge_index = self.edge_index[:, valid_mask]
                self.edge_type = self.edge_type[valid_mask]
                self.logger.info(f"Filtered edges, remaining: {self.edge_index.shape[1]}")
        else:
            # 如果没有边，创建最小的图结构
            self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=self.device).t()
            self.edge_type = torch.tensor([0, 0], dtype=torch.long, device=self.device)

        num_edges = self.edge_index.shape[1]
        self.logger.info(f"Graph: {self.num_nodes} nodes, {num_edges} edges, {self.num_relations} relations")
    
    def prepare_training_data(self):
        """准备训练数据"""
        self.logger.info("Preparing training data...")
        
        # 创建新的节点映射，包含三元关系中的所有节点
        all_nodes = set()
        for _, row in self.triple_df.iterrows():
            all_nodes.add(row['drug_id'])
            all_nodes.add(row['protein_id'])
            all_nodes.add(row['disease_id'])

        # 创建节点到索引的映射（将所有节点ID转换为字符串以避免排序问题）
        all_nodes_str = [str(node) for node in all_nodes]
        node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes_str))}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}

        self.logger.info(f"Created mapping for {len(node_to_idx)} unique nodes")

        # 更新映射信息
        self.mappings['node_to_idx'] = node_to_idx
        self.mappings['idx_to_node'] = idx_to_node
        self.num_nodes = len(node_to_idx)

        # 转换三元组（确保使用字符串形式的节点ID）
        valid_triples = []
        for _, row in self.triple_df.iterrows():
            valid_triples.append({
                'drug_idx': node_to_idx[str(row['drug_id'])],
                'protein_idx': node_to_idx[str(row['protein_id'])],
                'disease_idx': node_to_idx[str(row['disease_id'])],
                'drug_protein_relation': row['drug_protein_relation'],
                'protein_disease_relation': row['protein_disease_relation']
            })

        self.logger.info(f"Valid triples: {len(valid_triples)}")
        
        # 转换为张量
        valid_df = pd.DataFrame(valid_triples)
        
        drug_indices = torch.tensor(valid_df['drug_idx'].values, dtype=torch.long)
        protein_indices = torch.tensor(valid_df['protein_idx'].values, dtype=torch.long)
        disease_indices = torch.tensor(valid_df['disease_idx'].values, dtype=torch.long)
        
        # 创建正样本标签
        num_samples = len(valid_triples)
        existence_labels = torch.ones(num_samples, dtype=torch.float)
        protein_importance = torch.ones(num_samples, dtype=torch.float)  # 简化：所有蛋白质都重要
        pathway_labels = torch.randint(0, 10, (num_samples,), dtype=torch.long)  # 模拟通路标签
        mechanism_labels = torch.randint(0, 5, (num_samples,), dtype=torch.long)  # 模拟机制标签
        
        # 创建负样本
        neg_ratio = self.config['data']['negative_sampling_ratio']
        num_neg = int(num_samples * neg_ratio)
        
        neg_drug_indices = torch.randint(0, self.num_nodes, (num_neg,), dtype=torch.long)
        neg_protein_indices = torch.randint(0, self.num_nodes, (num_neg,), dtype=torch.long)
        neg_disease_indices = torch.randint(0, self.num_nodes, (num_neg,), dtype=torch.long)
        
        neg_existence_labels = torch.zeros(num_neg, dtype=torch.float)
        neg_protein_importance = torch.zeros(num_neg, dtype=torch.float)
        neg_pathway_labels = torch.full((num_neg,), -1, dtype=torch.long)  # 忽略
        neg_mechanism_labels = torch.full((num_neg,), -1, dtype=torch.long)  # 忽略
        
        # 合并正负样本
        all_drug_indices = torch.cat([drug_indices, neg_drug_indices])
        all_protein_indices = torch.cat([protein_indices, neg_protein_indices])
        all_disease_indices = torch.cat([disease_indices, neg_disease_indices])
        all_existence_labels = torch.cat([existence_labels, neg_existence_labels])
        all_protein_importance = torch.cat([protein_importance, neg_protein_importance])
        all_pathway_labels = torch.cat([pathway_labels, neg_pathway_labels])
        all_mechanism_labels = torch.cat([mechanism_labels, neg_mechanism_labels])
        
        # 打乱数据
        perm = torch.randperm(len(all_drug_indices))
        all_drug_indices = all_drug_indices[perm]
        all_protein_indices = all_protein_indices[perm]
        all_disease_indices = all_disease_indices[perm]
        all_existence_labels = all_existence_labels[perm]
        all_protein_importance = all_protein_importance[perm]
        all_pathway_labels = all_pathway_labels[perm]
        all_mechanism_labels = all_mechanism_labels[perm]
        
        # 数据分割
        total_samples = len(all_drug_indices)
        train_size = int(total_samples * (1 - self.config['data']['test_ratio'] - self.config['data']['val_ratio']))
        val_size = int(total_samples * self.config['data']['val_ratio'])
        
        # 训练集
        self.train_data = {
            'drug_indices': all_drug_indices[:train_size].to(self.device),
            'protein_indices': all_protein_indices[:train_size].to(self.device),
            'disease_indices': all_disease_indices[:train_size].to(self.device),
            'existence_labels': all_existence_labels[:train_size].to(self.device),
            'protein_importance': all_protein_importance[:train_size].to(self.device),
            'pathway_labels': all_pathway_labels[:train_size].to(self.device),
            'mechanism_labels': all_mechanism_labels[:train_size].to(self.device)
        }
        
        # 验证集
        self.val_data = {
            'drug_indices': all_drug_indices[train_size:train_size+val_size].to(self.device),
            'protein_indices': all_protein_indices[train_size:train_size+val_size].to(self.device),
            'disease_indices': all_disease_indices[train_size:train_size+val_size].to(self.device),
            'existence_labels': all_existence_labels[train_size:train_size+val_size].to(self.device),
            'protein_importance': all_protein_importance[train_size:train_size+val_size].to(self.device),
            'pathway_labels': all_pathway_labels[train_size:train_size+val_size].to(self.device),
            'mechanism_labels': all_mechanism_labels[train_size:train_size+val_size].to(self.device)
        }
        
        # 测试集
        self.test_data = {
            'drug_indices': all_drug_indices[train_size+val_size:].to(self.device),
            'protein_indices': all_protein_indices[train_size+val_size:].to(self.device),
            'disease_indices': all_disease_indices[train_size+val_size:].to(self.device),
            'existence_labels': all_existence_labels[train_size+val_size:].to(self.device),
            'protein_importance': all_protein_importance[train_size+val_size:].to(self.device),
            'pathway_labels': all_pathway_labels[train_size+val_size:].to(self.device),
            'mechanism_labels': all_mechanism_labels[train_size+val_size:].to(self.device)
        }
        
        self.logger.info(f"Data split - Train: {len(self.train_data['drug_indices'])}, "
                        f"Val: {len(self.val_data['drug_indices'])}, "
                        f"Test: {len(self.test_data['drug_indices'])}")
    
    def build_model(self):
        """构建模型"""
        self.logger.info("Building model...")
        
        # 确保模型的节点数量与实际数据匹配
        self.logger.info(f"Building model with {self.num_nodes} nodes and {self.num_relations} relations")

        self.model = TripleRelationRGCN(
            num_nodes=self.num_nodes,
            num_relations=max(self.num_relations, 2),  # 至少2个关系类型
            num_pathways=self.config['model'].get('num_pathways', 100),
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            fusion_dim=self.config['model']['triple_fusion_dim'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=self.config['training']['scheduler']['factor'],
            patience=self.config['training']['scheduler']['patience'],
            min_lr=self.config['training']['scheduler']['min_lr']
        )
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['patience'],
            min_delta=0.001
        )
        
        # 损失函数
        self.criteria = {
            'existence': nn.BCEWithLogitsLoss(),
            'protein_importance': nn.BCEWithLogitsLoss(),
            'pathway': nn.CrossEntropyLoss(ignore_index=-1),
            'mechanism': nn.CrossEntropyLoss(ignore_index=-1)
        }
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parameters: {total_params:,}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 创建数据加载器
        dataset = TensorDataset(
            self.train_data['drug_indices'],
            self.train_data['protein_indices'],
            self.train_data['disease_indices'],
            self.train_data['existence_labels'],
            self.train_data['protein_importance'],
            self.train_data['pathway_labels'],
            self.train_data['mechanism_labels']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        # 编码所有节点（一次性）
        node_indices = torch.arange(self.num_nodes, device=self.device)
        # 确保节点索引不超出模型的节点数量
        max_node_idx = node_indices.max().item()
        if max_node_idx >= self.model.num_nodes:
            self.logger.error(f"Node index {max_node_idx} exceeds model node count {self.model.num_nodes}")
            raise ValueError(f"Node index out of range: {max_node_idx} >= {self.model.num_nodes}")

        with torch.no_grad():
            node_embeddings = self.model.encode(node_indices, self.edge_index, self.edge_type)
        
        for batch_idx, (drug_idx, protein_idx, disease_idx, exist_labels, 
                       protein_imp, pathway_labels, mechanism_labels) in enumerate(dataloader):
            
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model.predict_triple_relations(
                node_embeddings, drug_idx, protein_idx, disease_idx
            )
            
            # 计算多任务损失
            task_weights = self.config['model']['task_weights']
            
            existence_loss = self.criteria['existence'](
                predictions['existence_prediction'].squeeze(), exist_labels
            )
            
            protein_loss = self.criteria['protein_importance'](
                predictions['protein_importance'].squeeze(), protein_imp
            )
            
            pathway_loss = self.criteria['pathway'](
                predictions['pathway_prediction'], pathway_labels
            )
            
            mechanism_loss = self.criteria['mechanism'](
                predictions['mechanism_classification'], mechanism_labels
            )
            
            # 总损失
            total_batch_loss = (
                existence_loss * task_weights['existence_prediction'] +
                protein_loss * task_weights['protein_ranking'] +
                pathway_loss * task_weights['pathway_prediction'] +
                mechanism_loss * task_weights['mechanism_classification']
            )
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            if self.config['training'].get('gradient_clip'):
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
                    f'Batch {batch_idx}/{len(dataloader)}, '
                    f'Total Loss: {total_batch_loss.item():.4f}, '
                    f'Existence: {existence_loss.item():.4f}, '
                    f'Protein: {protein_loss.item():.4f}, '
                    f'Pathway: {pathway_loss.item():.4f}, '
                    f'Mechanism: {mechanism_loss.item():.4f}'
                )
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def evaluate(self, data):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct_existence = 0
        correct_pathway = 0
        total_samples = 0
        
        with torch.no_grad():
            # 编码所有节点
            node_indices = torch.arange(self.num_nodes, device=self.device)
            node_embeddings = self.model.encode(node_indices, self.edge_index, self.edge_type)
            
            # 预测
            predictions = self.model.predict_triple_relations(
                node_embeddings,
                data['drug_indices'],
                data['protein_indices'],
                data['disease_indices']
            )
            
            # 计算损失
            task_weights = self.config['model']['task_weights']
            
            existence_loss = self.criteria['existence'](
                predictions['existence_prediction'].squeeze(), data['existence_labels']
            )
            
            protein_loss = self.criteria['protein_importance'](
                predictions['protein_importance'].squeeze(), data['protein_importance']
            )
            
            pathway_loss = self.criteria['pathway'](
                predictions['pathway_prediction'], data['pathway_labels']
            )
            
            mechanism_loss = self.criteria['mechanism'](
                predictions['mechanism_classification'], data['mechanism_labels']
            )
            
            total_loss = (
                existence_loss * task_weights['existence_prediction'] +
                protein_loss * task_weights['protein_ranking'] +
                pathway_loss * task_weights['pathway_prediction'] +
                mechanism_loss * task_weights['mechanism_classification']
            )
            
            # 计算准确率
            existence_pred = torch.sigmoid(predictions['existence_prediction'].squeeze()) > 0.5
            correct_existence = (existence_pred == data['existence_labels']).sum().item()
            
            # 通路预测准确率（忽略-1标签）
            pathway_mask = data['pathway_labels'] != -1
            if pathway_mask.sum() > 0:
                pathway_pred = torch.argmax(predictions['pathway_prediction'][pathway_mask], dim=1)
                correct_pathway = (pathway_pred == data['pathway_labels'][pathway_mask]).sum().item()
                pathway_total = pathway_mask.sum().item()
            else:
                pathway_total = 1  # 避免除零
            
            total_samples = len(data['existence_labels'])
        
        metrics = {
            'total_loss': total_loss.item(),
            'existence_accuracy': correct_existence / total_samples,
            'pathway_accuracy': correct_pathway / pathway_total if pathway_total > 0 else 0.0
        }
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        self.logger.info("Starting training...")
        
        best_val_score = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_metrics = self.evaluate(self.val_data)
            
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}, "
                f"Val Existence Acc: {val_metrics['existence_accuracy']:.4f}, "
                f"Val Pathway Acc: {val_metrics['pathway_accuracy']:.4f}"
            )
            
            # 学习率调度
            self.scheduler.step(val_metrics['existence_accuracy'])
            
            # 保存最佳模型
            val_score = val_metrics['existence_accuracy']
            if val_score > best_val_score:
                best_val_score = val_score
                
                if self.config['logging']['save_model']:
                    save_path = self.results_dir / "models" / "triple_model_best.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'config': self.config
                    }, save_path)
                    self.logger.info(f"Best model saved to: {save_path}")
            
            # 早停检查
            if self.early_stopping(val_score, self.model):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 最终测试
        test_metrics = self.evaluate(self.test_data)
        self.logger.info("Final test results:")
        for metric, value in test_metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Stage II Triple Relation Training")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    trainer = TripleRelationTrainer(args.config)
    trainer.load_data()
    trainer.build_model()
    
    test_metrics = trainer.train()
    
    print("Training completed!")
    print("Final test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
