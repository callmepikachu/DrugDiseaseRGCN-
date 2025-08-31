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
    split_edges, split_edges_cross_disease, prepare_multitask_data, calculate_metrics,
    save_model, print_model_info, EarlyStopping, clip_loss, NegativeSampler
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
        
        # 读取损失函数配置
        self.loss_config = config.get('training', {}).get('loss', {'type': 'bce_ce'})
        self.loss_type = self.loss_config.get('type', 'bce_ce')
        self.clip_temperature = self.loss_config.get('clip_temperature', 0.07)
        
        # 多任务损失函数 (仅在使用bce_ce时使用)
        if self.loss_type == 'bce_ce':
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
        
        # 根据配置选择分割方式
        evaluation_mode = self.config.get('evaluation', {}).get('mode', 'standard')
        if evaluation_mode == 'cross_disease':
            self.logger.info("使用Cross-Disease Split模式")
            train_data, val_data, test_data = split_edges_cross_disease(
                edge_index, edge_type, drug_disease_df, mappings,
                test_ratio=self.config['data']['test_ratio'],
                val_ratio=self.config['data']['val_ratio'],
                seed=self.config['seed']
            )
        else:
            self.logger.info("使用标准分割模式")
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

            # 计算损失
            if self.loss_type == 'clip':
                # 使用CLIP损失
                drug_emb = node_embeddings[head_indices]
                disease_emb = node_embeddings[tail_indices]
                total_batch_loss = clip_loss(drug_emb, disease_emb, self.clip_temperature)
            else:
                # 使用原有的BCE+CE损失
                existence_loss = self.existence_criterion(existence_scores, existence_labels.float())
                relation_loss = self.relation_criterion(relation_logits, relation_labels)
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
                if self.loss_type == 'clip':
                    self.logger.info(
                        f'Batch {batch_idx}/{len(dataloader)}, Total Loss: {total_batch_loss.item():.4f}'
                    )
                else:
                    existence_loss_val = existence_loss.item() if 'existence_loss' in locals() else 0
                    relation_loss_val = relation_loss.item() if 'relation_loss' in locals() else 0
                    self.logger.info(
                        f'Batch {batch_idx}/{len(dataloader)}, Total Loss: {total_batch_loss.item():.4f}, '
                        f'Existence Loss: {existence_loss_val:.4f}, Relation Loss: {relation_loss_val:.4f}'
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
            if self.loss_type == 'clip':
                # 使用CLIP损失
                drug_emb = node_embeddings[data['edge_index'][0]]
                disease_emb = node_embeddings[data['edge_index'][1]]
                total_loss = clip_loss(drug_emb, disease_emb, self.clip_temperature)
                existence_loss = torch.tensor(0.0)  # CLIP模式下不计算existence_loss
                relation_loss = torch.tensor(0.0)  # CLIP模式下不计算relation_loss
            else:
                # 使用原有的BCE+CE损失
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