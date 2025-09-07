"""
RGCN模型定义
"""
import numpy as np
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


class LinkPredictor(nn.Module):
    """简化版链接预测器：只预测关系存在性"""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super(LinkPredictor, self).__init__()

        self.hidden_dim = hidden_dim

        # 特征提取层
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 最后一层直接输出分数
        )

        self._init_parameters()
    
    def _init_parameters(self):
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        head_indices: torch.Tensor,
        tail_indices: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Returns:
            torch.Tensor: existence_scores [batch_size]
        """
        head_emb = node_embeddings[head_indices]
        tail_emb = node_embeddings[tail_indices]
        combined = torch.cat([head_emb, tail_emb], dim=-1)
        existence_scores = self.feature_layers(combined).squeeze(-1)

        return existence_scores



class DrugDiseaseRGCN(nn.Module):
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

        # 简化版链接预测器
        self.link_predictor = LinkPredictor(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # --- 关键修复：在这里定义可学习的温度参数 ---
        self.logit_scale = nn.Parameter(torch.tensor([np.log(0.5)]))  # 初始化为 ln(1.0) = 0.0
        # --- 关键修复结束 ---

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        head_indices: torch.Tensor,
        tail_indices: torch.Tensor
    ) -> torch.Tensor:
        node_embeddings = self.encoder(x, edge_index, edge_type)
        existence_scores = self.link_predictor(
            node_embeddings,
            head_indices,
            tail_indices
        )
        return existence_scores
    
    def predict_links(
        self,
        node_embeddings: torch.Tensor,
        head_indices: torch.Tensor,
        tail_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.link_predictor(
            node_embeddings,
            head_indices,
            tail_indices
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_type)
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
