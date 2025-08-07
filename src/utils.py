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


def prepare_link_prediction_data(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    negative_sampling_ratio: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """准备链接预测数据"""
    num_positive = edge_index.shape[1]
    num_negative = int(num_positive * negative_sampling_ratio)
    
    # 创建正样本标签
    positive_labels = torch.ones(num_positive)
    
    # 创建负样本
    existing_edges = set()
    for i in range(num_positive):
        head, tail = edge_index[0, i].item(), edge_index[1, i].item()
        existing_edges.add((head, tail))
    
    negative_edge_index = create_negative_samples(
        edge_index, num_nodes, num_negative, existing_edges
    )
    
    # 为负样本随机分配关系类型
    negative_edge_type = torch.randint(
        0, edge_type.max().item() + 1, (num_negative,)
    )
    
    # 创建负样本标签
    negative_labels = torch.zeros(num_negative)
    
    # 合并正负样本
    all_edge_index = torch.cat([edge_index, negative_edge_index], dim=1)
    all_edge_type = torch.cat([edge_type, negative_edge_type], dim=0)
    all_labels = torch.cat([positive_labels, negative_labels], dim=0)
    
    # 打乱顺序
    perm = torch.randperm(all_edge_index.shape[1])
    all_edge_index = all_edge_index[:, perm]
    all_edge_type = all_edge_type[perm]
    all_labels = all_labels[perm]
    
    return all_edge_index, all_edge_type, all_labels, perm


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
