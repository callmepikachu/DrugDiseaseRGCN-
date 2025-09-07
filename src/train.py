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
    save_model, print_model_info, EarlyStopping, clip_loss, NegativeSampler, calculate_mrr
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
            # self.relation_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 关系类型，忽略负样本
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config['training']['patience']
        )
        
        # 创建保存目录
        os.makedirs(config['logging']['model_dir'], exist_ok=True)
        
        # 初始化负样本数据
        self.negative_df = None
        # 初始化正样本对属性
        self.train_positive_pairs = None

    def load_data(self):
        """加载数据"""
        self.logger.info("正在加载数据...")
        
        # 加载处理后的数据
        hetero_data, mappings, drug_disease_df = self.data_loader.load_processed_data()
        
        self.mappings = mappings
        self.num_nodes = mappings['num_nodes']
        self.num_relations = mappings['num_relations']
        
        # 加载负样本数据
        try:
            self.negative_df = self.data_loader.load_negative_samples()
            self.logger.info(f"成功加载负样本数据，共 {len(self.negative_df)} 条记录")
        except Exception as e:
            self.logger.warning(f"加载负样本数据失败: {e}")
            self.negative_df = None
        
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
        # --- 新增代码：为CLIP训练准备正样本对 ---
        if self.loss_type == 'clip':
            # 从训练数据中筛选出正样本 (existence_labels == 1)
            # 注意：self.train_data['existence_labels'] 包含了正负样本，1为正，0为负
            train_positive_mask = self.train_data['existence_labels'] == 1
            # 提取对应的头节点（药物）和尾节点（疾病）索引
            train_positive_head_indices = self.train_data['edge_index'][0][train_positive_mask]
            train_positive_tail_indices = self.train_data['edge_index'][1][train_positive_mask]

            # 将其存储为一个元组，供 train_epoch 使用
            self.train_positive_pairs = (train_positive_head_indices, train_positive_tail_indices)

            self.logger.info(f"为CLIP训练准备的正样本对数量: {len(train_positive_head_indices)}")
        else:
            # 如果不是CLIP模式，可以置为None或不做处理
            self.train_positive_pairs = None
        # --- 新增代码结束 ---
        
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
            all_positive_edges,
            self.negative_df,
            self.mappings
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
        self.model.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(self.device)
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
        self.model.train()
        total_loss = 0
        num_batches = 0

        # --- 关键修改：为多标签CLIP准备数据 ---
        # 获取当前批次中所有的药物和疾病索引（去重）
        # 注意：这里我们直接使用 self.train_data，它包含了正负样本，但我们只关心正样本构建 mask
        all_head_indices = self.train_data['edge_index'][0]
        all_tail_indices = self.train_data['edge_index'][1]
        all_existence_labels = self.train_data['existence_labels']

        # 筛选出所有正样本
        positive_mask_all = (all_existence_labels == 1)
        positive_head_indices_all = all_head_indices[positive_mask_all]
        positive_tail_indices_all = all_tail_indices[positive_mask_all]

        # 获取唯一的药物和疾病节点
        unique_drug_indices = torch.unique(positive_head_indices_all)
        unique_disease_indices = torch.unique(positive_tail_indices_all)

        # 创建从全局索引到局部索引的映射
        drug_map = {idx.item(): i for i, idx in enumerate(unique_drug_indices)}
        disease_map = {idx.item(): i for i, idx in enumerate(unique_disease_indices)}

        # 构建完整的 positive_mask 矩阵 [N_unique_drugs, N_unique_diseases]
        positive_mask = torch.zeros(len(unique_drug_indices), len(unique_disease_indices), dtype=torch.bool,
                                    device=self.device)

        # 填充 positive_mask
        for i in range(positive_head_indices_all.size(0)):
            drug_global_idx = positive_head_indices_all[i].item()
            disease_global_idx = positive_tail_indices_all[i].item()
            drug_local_idx = drug_map[drug_global_idx]
            disease_local_idx = disease_map[disease_global_idx]
            positive_mask[drug_local_idx, disease_local_idx] = True

        # 创建数据加载器，用于分批处理
        # 我们按药物或疾病分批，这里按药物分批
        drug_dataset = TensorDataset(unique_drug_indices)
        drug_dataloader = DataLoader(drug_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)

        for batch_idx, (batch_drug_indices,) in enumerate(drug_dataloader):
            self.optimizer.zero_grad()

            # 编码所有节点
            node_indices = torch.arange(self.num_nodes, device=self.device)
            node_embeddings = self.model.encoder(
                node_indices, self.full_edge_index, self.full_edge_type
            )

            # 提取当前批次药物的嵌入
            batch_drug_emb = node_embeddings[batch_drug_indices]  # [batch_size, D]

            # 提取所有疾病的嵌入 (也可以考虑分批处理疾病以节省内存)
            all_disease_emb = node_embeddings[unique_disease_indices]  # [M, D]

            # 构建当前批次的 positive_mask [batch_size, M]
            batch_drug_global_indices = batch_drug_indices.cpu().numpy()
            batch_positive_mask = torch.zeros(batch_drug_emb.size(0), all_disease_emb.size(0), dtype=torch.bool,
                                              device=self.device)
            for i, drug_global_idx in enumerate(batch_drug_global_indices):
                drug_local_idx = drug_map[drug_global_idx]
                batch_positive_mask[i, :] = positive_mask[drug_local_idx, :]

            # 计算多标签 CLIP 损失
            total_batch_loss = clip_loss(
                batch_drug_emb,
                all_disease_emb,
                batch_positive_mask,
                model=self.model  # 传入模型以获取可学习温度
            )

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
                    f'Batch {batch_idx}/{len(drug_dataloader)}, MultiLabel CLIP Loss: {total_batch_loss.item():.4f}'
                )

        return total_loss / num_batches if num_batches > 0 else float('inf')

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
            existence_scores = self.model.predict_links(
                node_embeddings,
                data['edge_index'][0],
                data['edge_index'][1]
            )
            # 由于新模型没有 relation_logits，我们创建一个占位符
            # 这样可以避免后续代码报错，因为后续代码会检查 positive_mask
            relation_logits = torch.zeros((existence_scores.size(0), self.num_relations), device=self.device)

            # 计算损失
            if self.loss_type == 'clip':
                # 使用多标签CLIP损失进行评估
                # 获取数据中所有的药物和疾病索引
                all_head_indices = data['edge_index'][0]
                all_tail_indices = data['edge_index'][1]
                all_existence_labels = data['existence_labels']

                # 筛选出所有正样本
                positive_mask_all = (all_existence_labels == 1)
                positive_head_indices_all = all_head_indices[positive_mask_all]
                positive_tail_indices_all = all_tail_indices[positive_mask_all]

                # 获取唯一的药物和疾病节点
                unique_drug_indices = torch.unique(positive_head_indices_all)
                unique_disease_indices = torch.unique(positive_tail_indices_all)

                if len(unique_drug_indices) == 0 or len(unique_disease_indices) == 0:
                    total_loss = torch.tensor(0.0, device=self.device)
                else:
                    # 创建映射
                    drug_map = {idx.item(): i for i, idx in enumerate(unique_drug_indices)}
                    disease_map = {idx.item(): i for i, idx in enumerate(unique_disease_indices)}

                    # 构建 positive_mask 矩阵
                    positive_mask = torch.zeros(len(unique_drug_indices), len(unique_disease_indices), dtype=torch.bool,
                                                device=self.device)
                    for i in range(positive_head_indices_all.size(0)):
                        drug_global_idx = positive_head_indices_all[i].item()
                        disease_global_idx = positive_tail_indices_all[i].item()
                        if drug_global_idx in drug_map and disease_global_idx in disease_map:  # 防御性编程
                            drug_local_idx = drug_map[drug_global_idx]
                            disease_local_idx = disease_map[disease_global_idx]
                            positive_mask[drug_local_idx, disease_local_idx] = True

                    # 获取嵌入
                    drug_emb = node_embeddings[unique_drug_indices]
                    disease_emb = node_embeddings[unique_disease_indices]

                    # 计算损失
                    total_loss = clip_loss(drug_emb, disease_emb, positive_mask,  model=self.model)

                existence_loss = torch.tensor(0.0, device=self.device)
                relation_loss = torch.tensor(0.0, device=self.device)
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

            # --- 新增代码：计算 MRR 指标 ---
            # 获取评估模式
            evaluation_mode = self.config.get('evaluation', {}).get('mode', 'standard')
            mrr_metrics = {}
            if evaluation_mode == 'cross_disease':
                try:
                    mrr_by_drug = calculate_mrr(
                        node_embeddings,
                        data['edge_index'],
                        data['existence_labels'],
                        self.mappings,
                        target_entity="drug"
                    )
                    mrr_by_disease = calculate_mrr(
                        node_embeddings,
                        data['edge_index'],
                        data['existence_labels'],
                        self.mappings,
                        target_entity="disease"
                    )
                    mrr_metrics['mrr_by_drug'] = mrr_by_drug
                    mrr_metrics['mrr_by_disease'] = mrr_by_disease
                except Exception as e:
                    self.logger.warning(f"计算 MRR 时出错: {e}")
                    mrr_metrics['mrr_by_drug'] = 0.0
                    mrr_metrics['mrr_by_disease'] = 0.0
            else:
                # 在标准模式下，也可以计算 MRR，但意义可能不如 cross_disease 模式大
                mrr_metrics['mrr_by_drug'] = 0.0
                mrr_metrics['mrr_by_disease'] = 0.0
            # --- 新增代码结束 ---

            # 合并指标
            metrics = {}
            for key, value in existence_metrics.items():
                metrics[f'existence_{key}'] = value
            # 将 MRR 指标加入 metrics 字典
            for key, value in mrr_metrics.items():
                metrics[key] = value

            metrics['total_loss'] = total_loss.item()
            metrics['existence_loss'] = existence_loss.item()
            metrics['relation_loss'] = relation_loss.item()

        return metrics

    def train(self):
        """训练模型"""
        self.logger.info("开始训练...")

        best_val_mrr = 0  # 使用 MRR 作为核心指标

        for epoch in range(self.config['training']['num_epochs']):
            self.logger.info(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_metrics = self.evaluate(self.val_data, "val")

            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Total Loss: {val_metrics['total_loss']:.4f}, "
                             f"Existence AUC: {val_metrics['existence_auc']:.4f}, ")
            self.logger.info(f"Val MRR by Drug: {val_metrics.get('mrr_by_drug', 0.0):.4f}, "
                             f"Val MRR by Disease: {val_metrics.get('mrr_by_disease', 0.0):.4f}")

            # 学习率调度 (保持不变，通常还是基于 AUC)
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['existence_auc'])
            else:
                self.scheduler.step()

            # --- 修改：保存最佳模型（基于 MRR）---
            # if val_metrics['existence_auc'] > best_val_auc:
            current_mrr = val_metrics.get('mrr_by_drug', 0.0)  # 或者使用 (mrr_by_drug + mrr_by_disease) / 2
            if current_mrr > best_val_mrr:
                best_val_mrr = current_mrr

                if self.config['logging']['save_model']:
                    save_path = os.path.join(
                        self.config['logging']['model_dir'],
                        f"{self.config['experiment']['name']}_best.pth"
                    )
                    save_model(
                        self.model, self.optimizer, epoch,
                        val_metrics['total_loss'], val_metrics, save_path
                    )
                    self.logger.info(f"保存最佳模型到: {save_path} (MRR: {best_val_mrr:.4f})")
            # --- 修改结束 ---

            # --- 修改：早停检查（基于 MRR）---
            # if self.early_stopping(val_metrics['existence_auc'], self.model):
            if self.early_stopping(current_mrr, self.model):
                self.logger.info(f"早停在epoch {epoch + 1} (基于 MRR)")
                break
            # --- 修改结束 ---

        # 最终测试
        test_metrics = self.evaluate(self.test_data, "test")
        self.logger.info("测试结果: ")
        for metric in ['existence_auc', 'existence_ap', 'mrr_by_drug', 'mrr_by_disease', 'total_loss']:
            if metric in test_metrics:
                self.logger.info(f"{metric}: {test_metrics[metric]:.4f}")
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