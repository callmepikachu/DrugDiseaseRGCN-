"""
工具函数
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def clip_loss(
        drug_embeddings: torch.Tensor,  # [N, D] - N个唯一药物的嵌入
        disease_embeddings: torch.Tensor,  # [M, D] - M个唯一疾病的嵌入
        positive_mask: torch.Tensor,  # [N, M] - 布尔矩阵，True 表示是正样本对
        model: torch.nn.Module = None,  # 新增参数，传入模型
        temperature: float = 1.0
) -> torch.Tensor:
    """
    多标签版本的 CLIP 损失。
    positive_mask[i][j] = True 表示 第i个药物 和 第j个疾病 是正样本关系。
    """
    device = drug_embeddings.device

    # # 1. L2 归一化嵌入
    drug_embeddings = F.normalize(drug_embeddings, p=2, dim=1)
    disease_embeddings = F.normalize(disease_embeddings, p=2, dim=1)

    # 使用模型内部的可学习温度
    if model is not None and hasattr(model, 'logit_scale'):
        temperature = model.logit_scale.exp() # 现在 temperature 是一个很大的值，但我们要用它做除法
        logits = torch.matmul(drug_embeddings, disease_embeddings.t()) / temperature
    else:
        # 兜底方案，使用默认值或从配置读取
        print("正在使用默认温度")
        logits = torch.matmul(drug_embeddings, disease_embeddings.t()) / temperature


    # --- 调试信息 (可选，用于监控) ---
    if model is not None and hasattr(model, 'logit_scale'):
        print(f"[Debug MultiLabel CLIP Loss] Logits shape: {logits.shape}, Learned Temperature = {temperature}")
    else:
        print(f"[Debug MultiLabel CLIP Loss] Logits shape: {logits.shape}, Temperature = {temperature}")
    # --- 调试信息结束 ---

    # 3. 创建标签 (直接使用 positive_mask)
    labels_drug_to_disease = positive_mask.float()  # [N, M]
    labels_disease_to_drug = positive_mask.t().float()  # [M, N]

    # 4. 计算损失 (使用二元交叉熵，支持多标签)
    bce_loss = nn.BCEWithLogitsLoss()

    loss_drug_to_disease = bce_loss(logits, labels_drug_to_disease)
    loss_disease_to_drug = bce_loss(logits.t(), labels_disease_to_drug)

    total_loss = (loss_drug_to_disease + loss_disease_to_drug) / 2.0
    return total_loss

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


class NegativeSampler:
    """负采样器，从预生成的负样本中采样"""
    
    def __init__(self, negative_file: str, node_to_idx: Dict):
        self.node_to_idx = node_to_idx
        self.idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        
        # 加载负样本数据
        print("正在加载负样本数据...")
        # 由于文件可能很大，我们只加载部分数据用于初始化
        self.negative_df = pd.read_csv(negative_file, nrows=100000)
        print(f"负样本数据形状: {self.negative_df.shape}")
        
        # 创建药物和疾病的映射
        self.drug_to_negatives = {}
        self.disease_to_negatives = {}
        
        # 初始化映射
        for _, row in self.negative_df.iterrows():
            drug_id = row['drug_id']
            disease_id = row['disease_id']
            
            # 添加到药物的负样本列表
            if drug_id not in self.drug_to_negatives:
                self.drug_to_negatives[drug_id] = []
            self.drug_to_negatives[drug_id].append(disease_id)
            
            # 添加到疾病的负样本列表
            if disease_id not in self.disease_to_negatives:
                self.disease_to_negatives[disease_id] = []
            self.disease_to_negatives[disease_id].append(drug_id)
    
    def get_negatives_by_drug(self, drug_id: str) -> List[str]:
        """根据药物ID获取负样本疾病列表"""
        return self.drug_to_negatives.get(drug_id, [])
    
    def get_negatives_by_disease(self, disease_id: str) -> List[str]:
        """根据疾病ID获取负样本药物列表"""
        return self.disease_to_negatives.get(disease_id, [])
    
    def sample_negative_edges(self, positive_edges: torch.Tensor, num_negative: int) -> torch.Tensor:
        """从预生成的负样本中采样负边"""
        negative_edges = []
        sampled = set()
        
        # 转换正样本为集合以便快速查找
        positive_set = set()
        for i in range(positive_edges.shape[1]):
            head, tail = positive_edges[0, i].item(), positive_edges[1, i].item()
            positive_set.add((head, tail))
        
        # 尝试从预生成的负样本中采样
        attempts = 0
        max_attempts = num_negative * 10  # 最大尝试次数
        
        while len(negative_edges) < num_negative and attempts < max_attempts:
            attempts += 1
            
            # 随机选择一个负样本
            idx = random.randint(0, len(self.negative_df) - 1)
            row = self.negative_df.iloc[idx]
            
            drug_id = row['drug_id']
            disease_id = row['disease_id']
            
            # 检查药物和疾病是否在节点映射中
            if drug_id in self.node_to_idx and disease_id in self.node_to_idx:
                head = self.node_to_idx[drug_id]
                tail = self.node_to_idx[disease_id]
                
                # 检查是否已经采样过或者在正样本中
                if (head, tail) not in sampled and (head, tail) not in positive_set:
                    negative_edges.append([head, tail])
                    sampled.add((head, tail))
        
        # 如果采样不足，使用随机采样补充
        if len(negative_edges) < num_negative:
            print(f"从预生成负样本中只采样到 {len(negative_edges)} 个，使用随机采样补充...")
            while len(negative_edges) < num_negative:
                head = random.randint(0, len(self.node_to_idx) - 1)
                tail = random.randint(0, len(self.node_to_idx) - 1)
                
                if head != tail and (head, tail) not in sampled and (head, tail) not in positive_set:
                    negative_edges.append([head, tail])
                    sampled.add((head, tail))
        
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


def split_edges_cross_disease(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    drug_disease_df: pd.DataFrame,
    mappings: Dict,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """实现Cross-Disease Splits"""
    # 获取所有唯一的疾病 ID
    unique_diseases = drug_disease_df['y_id'].unique()  # 假设疾病是y_id
    
    # 随机划分疾病为训练、验证、测试集
    train_diseases, temp_diseases = train_test_split(
        unique_diseases, test_size=(test_ratio + val_ratio), random_state=seed
    )
    val_diseases, test_diseases = train_test_split(
        temp_diseases, test_size=test_ratio/(test_ratio + val_ratio), random_state=seed
    )
    
    # 创建疾病到索引的映射
    node_to_idx = mappings['node_to_idx']
    
    # 根据划分的疾病集筛选边
    def filter_edges_by_diseases(diseases):
        mask = np.isin(edge_index[1].cpu().numpy(), [node_to_idx[d] for d in diseases if d in node_to_idx])
        return edge_index[:, mask], edge_type[mask]
    
    train_edge_index, train_edge_type = filter_edges_by_diseases(train_diseases)
    val_edge_index, val_edge_type = filter_edges_by_diseases(val_diseases)
    test_edge_index, test_edge_type = filter_edges_by_diseases(test_diseases)
    
    # 创建数据集
    train_data = {
        'edge_index': train_edge_index,
        'edge_type': train_edge_type
    }
    
    val_data = {
        'edge_index': val_edge_index,
        'edge_type': val_edge_type
    }
    
    test_data = {
        'edge_index': test_edge_index,
        'edge_type': test_edge_type
    }
    
    return train_data, val_data, test_data


def prepare_multitask_data(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    negative_sampling_ratio: float = 1.0,
    all_positive_edges: set = None,
    negative_df: pd.DataFrame = None,
    mappings: Dict = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """准备多任务学习数据

    Args:
        all_positive_edges: 所有分割中的正样本边集合，用于避免数据泄露
        negative_df: 预生成的负样本数据框
        mappings: 节点映射字典

    Returns:
        tuple: (all_edge_index, existence_labels, relation_type_labels, perm)
    """
    num_positive = edge_index.shape[1]
    num_negative = int(num_positive * negative_sampling_ratio)

    # 正样本：存在关系=1，关系类型=实际类型
    positive_existence_labels = torch.ones(num_positive)
    positive_relation_labels = edge_type.clone()

    # 创建负样本
    if negative_df is not None and mappings is not None:
        # 使用预生成的负样本
        node_to_idx = mappings['node_to_idx']
        
        # 获取正样本中涉及的唯一药物和疾病
        unique_drugs = set()
        unique_diseases = set()
        for i in range(num_positive):
            head, tail = edge_index[0, i].item(), edge_index[1, i].item()
            drug_id = mappings['idx_to_node'][head]
            disease_id = mappings['idx_to_node'][tail]
            unique_drugs.add(drug_id)
            unique_diseases.add(disease_id)
        
        # 筛选负样本数据框中相关的负样本
        filtered_negative_df = negative_df[
            (negative_df['drug_id'].isin(unique_drugs)) & 
            (negative_df['disease_id'].isin(unique_diseases))
        ].copy()
        
        # 如果筛选后的负样本足够，从中随机采样
        if len(filtered_negative_df) >= num_negative:
            sampled_negative_df = filtered_negative_df.sample(n=num_negative, random_state=42)
        else:
            # 如果不够，使用所有筛选后的负样本
            sampled_negative_df = filtered_negative_df
            print(f"负样本不足，需要 {num_negative} 个，但只有 {len(filtered_negative_df)} 个")
        
        # 转换为节点索引
        negative_edges = []
        for _, row in sampled_negative_df.iterrows():
            if row['drug_id'] in node_to_idx and row['disease_id'] in node_to_idx:
                head_idx = node_to_idx[row['drug_id']]
                tail_idx = node_to_idx[row['disease_id']]
                negative_edges.append([head_idx, tail_idx])
        
        # 如果转换后的负样本不足，补充随机生成的负样本
        if len(negative_edges) < num_negative:
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
            
            # 补充随机生成的负样本
            additional_negative_edges = create_negative_samples(
                edge_index, num_nodes, num_negative - len(negative_edges), existing_edges
            )
            # 将补充的负样本添加到negative_edges中
            for i in range(additional_negative_edges.shape[1]):
                negative_edges.append([additional_negative_edges[0, i].item(), additional_negative_edges[1, i].item()])
        
        negative_edge_index = torch.tensor(negative_edges, dtype=torch.long).t()
    else:
        # 使用原有的负样本生成方法
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
    negative_existence_labels = torch.zeros(negative_edge_index.shape[1])
    negative_relation_labels = torch.full((negative_edge_index.shape[1],), -1, dtype=torch.long)  # -1表示忽略

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


def calculate_mrr(
    node_embeddings: torch.Tensor,
    test_edge_index: torch.Tensor,
    test_existence_labels: torch.Tensor,
    mappings: Dict,
    target_entity: str = "drug",
    k_values: List[int] = [10, 50]
) -> float:
    """
    计算MRR指标
    
    Args:
        node_embeddings: 节点嵌入 [num_nodes, embedding_dim]
        test_edge_index: 测试边索引 [2, num_test_edges]
        test_existence_labels: 测试标签 [num_test_edges]
        mappings: 节点映射字典
        target_entity: 目标实体类型 "drug" 或 "disease"
        k_values: 用于扩展计算 MRR@k
        
    Returns:
        float: MRR值
    """
    # 筛选正样本
    positive_mask = test_existence_labels == 1
    positive_test_edges = test_edge_index[:, positive_mask]
    
    if target_entity == "drug":
        # 查询实体是药物
        query_indices = torch.unique(positive_test_edges[0, :])  # 所有要查询的药物索引
        candidate_indices = torch.unique(test_edge_index[1, :])  # 测试集中所有出现的疾病索引作为候选
    else:
        # 查询实体是疾病
        query_indices = torch.unique(positive_test_edges[1, :])  # 所有要查询的疾病索引
        candidate_indices = torch.unique(test_edge_index[0, :])  # 测试集中所有出现的药物索引作为候选
    
    # 获取候选实体嵌入
    candidate_embeddings = node_embeddings[candidate_indices]
    
    # 计算MRR
    reciprocal_ranks = []
    
    for q_idx in query_indices:
        # 获取查询实体嵌入
        q_emb = node_embeddings[q_idx].unsqueeze(0)  # [1, D]
        
        # 计算相似度得分
        scores = torch.matmul(
            F.normalize(q_emb, p=2, dim=1), 
            F.normalize(candidate_embeddings, p=2, dim=1).t()
        ).squeeze()  # [num_candidates]
        
        # 降序排序得分
        sorted_candidate_indices = candidate_indices[torch.argsort(scores, descending=True)]
        
        # 找到与 q_idx 相连的真实正样本候选索引
        if target_entity == "drug":
            # 查找与药物 q_idx 相连的疾病
            true_pos_mask = (positive_test_edges[0, :] == q_idx)
            true_pos_candidates = positive_test_edges[1, true_pos_mask]
        else:
            # 查找与疾病 q_idx 相连的药物
            true_pos_mask = (positive_test_edges[1, :] == q_idx)
            true_pos_candidates = positive_test_edges[0, true_pos_mask]
        
        # 对每个真实正样本计算排名
        for tp_cand_idx in true_pos_candidates:
            # 找到其在排序中的位置
            rank_positions = (sorted_candidate_indices == tp_cand_idx).nonzero(as_tuple=True)[0]
            if len(rank_positions) > 0:
                rank = rank_positions[0].item() + 1  # 排名从1开始
                reciprocal_ranks.append(1.0 / rank)
    
    # 返回MRR值
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def calculate_mrr_old(
    node_embeddings: torch.Tensor,
    test_edge_index: torch.Tensor,
    mappings: Dict
) -> Dict[str, float]:
    """计算MRR指标 (旧版本)"""
    # 获取测试集中的唯一药物和疾病
    unique_drugs = torch.unique(test_edge_index[0])
    unique_diseases = torch.unique(test_edge_index[1])
    
    reciprocal_ranks_drug = []
    reciprocal_ranks_disease = []
    
    # 计算药物到疾病的MRR
    for drug_idx in unique_drugs:
        # 获取与该药物相连的所有疾病
        drug_mask = test_edge_index[0] == drug_idx
        true_diseases = test_edge_index[1][drug_mask]
        
        if len(true_diseases) == 0:
            continue
            
        # 计算该药物与所有疾病候选的相似度
        drug_emb = node_embeddings[drug_idx].unsqueeze(0)  # [1, D]
        disease_embs = node_embeddings[unique_diseases]    # [num_diseases, D]
        
        # L2归一化
        drug_emb = F.normalize(drug_emb, p=2, dim=1)
        disease_embs = F.normalize(disease_embs, p=2, dim=1)
        
        # 计算相似度得分
        scores = torch.matmul(drug_emb, disease_embs.t()).squeeze()  # [num_diseases]
        
        # 获取排序索引（降序）
        sorted_indices = torch.argsort(scores, descending=True)
        
        # 找到真实疾病的排名
        for true_disease in true_diseases:
            # 找到真实疾病在候选疾病中的索引
            try:
                disease_candidate_idx = (unique_diseases == true_disease).nonzero(as_tuple=True)[0]
                if len(disease_candidate_idx) > 0:
                    disease_candidate_idx = disease_candidate_idx[0].item()
                    # 找到该疾病在排序中的位置
                    rank = (sorted_indices == disease_candidate_idx).nonzero(as_tuple=True)[0]
                    if len(rank) > 0:
                        rank = rank[0].item() + 1  # 排名从1开始
                        reciprocal_ranks_drug.append(1.0 / rank)
            except Exception:
                continue
    
    # 计算疾病到药物的MRR
    for disease_idx in unique_diseases:
        # 获取与该疾病相连的所有药物
        disease_mask = test_edge_index[1] == disease_idx
        true_drugs = test_edge_index[0][disease_mask]
        
        if len(true_drugs) == 0:
            continue
            
        # 计算该疾病与所有药物候选的相似度
        disease_emb = node_embeddings[disease_idx].unsqueeze(0)  # [1, D]
        drug_embs = node_embeddings[unique_drugs]               # [num_drugs, D]
        
        # L2归一化
        disease_emb = F.normalize(disease_emb, p=2, dim=1)
        drug_embs = F.normalize(drug_embs, p=2, dim=1)
        
        # 计算相似度得分
        scores = torch.matmul(disease_emb, drug_embs.t()).squeeze()  # [num_drugs]
        
        # 获取排序索引（降序）
        sorted_indices = torch.argsort(scores, descending=True)
        
        # 找到真实药物的排名
        for true_drug in true_drugs:
            # 找到真实药物在候选药物中的索引
            try:
                drug_candidate_idx = (unique_drugs == true_drug).nonzero(as_tuple=True)[0]
                if len(drug_candidate_idx) > 0:
                    drug_candidate_idx = drug_candidate_idx[0].item()
                    # 找到该药物在排序中的位置
                    rank = (sorted_indices == drug_candidate_idx).nonzero(as_tuple=True)[0]
                    if len(rank) > 0:
                        rank = rank[0].item() + 1  # 排名从1开始
                        reciprocal_ranks_disease.append(1.0 / rank)
            except Exception:
                continue
    
    # 计算MRR
    mrr_by_drug = np.mean(reciprocal_ranks_drug) if reciprocal_ranks_drug else 0.0
    mrr_by_disease = np.mean(reciprocal_ranks_disease) if reciprocal_ranks_disease else 0.0
    
    return {
        'mrr_by_drug': mrr_by_drug,
        'mrr_by_disease': mrr_by_disease
    }


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