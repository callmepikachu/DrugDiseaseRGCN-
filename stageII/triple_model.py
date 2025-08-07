#!/usr/bin/env python3
"""
Stage II 三元关系预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import sys
import math

# 添加父目录以导入Stage I的模块
sys.path.append('../src')


class EnhancedRGCNEncoder(nn.Module):
    """增强的RGCN编码器，支持更复杂的关系建模"""
    
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.15
    ):
        super(EnhancedRGCNEncoder, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 节点嵌入
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        
        # RGCN层
        self.rgcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.rgcn_layers.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations)
            )
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.node_embedding.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        # 获取节点嵌入
        h = self.node_embedding(x)
        
        # 通过RGCN层
        for i, (rgcn_layer, layer_norm) in enumerate(zip(self.rgcn_layers, self.layer_norms)):
            h_new = rgcn_layer(h, edge_index, edge_type)
            h_new = layer_norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            # 残差连接
            if i > 0:
                h = h + h_new
            else:
                h = h_new
        
        return h


class AttentionFusion(nn.Module):
    """注意力融合模块，用于融合三个实体的表示"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(AttentionFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # 多头注意力
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, drug_emb, protein_emb, disease_emb):
        """
        Args:
            drug_emb: [batch_size, hidden_dim]
            protein_emb: [batch_size, hidden_dim]
            disease_emb: [batch_size, hidden_dim]
        """
        batch_size = drug_emb.size(0)
        
        # 堆叠三个实体的嵌入 [batch_size, 3, hidden_dim]
        entities = torch.stack([drug_emb, protein_emb, disease_emb], dim=1)
        
        # 计算注意力
        Q = self.query(entities).view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(entities).view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(entities).view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, 3, self.hidden_dim
        )
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        
        # 返回融合后的表示
        fused_repr = attn_output.mean(dim=1)  # [batch_size, hidden_dim]
        
        return fused_repr, attn_weights.mean(dim=1)  # 返回注意力权重用于可视化


class TripleRelationPredictor(nn.Module):
    """三元关系预测器"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        num_pathways: int,
        fusion_dim: int = 256,
        dropout: float = 0.15
    ):
        super(TripleRelationPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        
        # 注意力融合模块
        self.attention_fusion = AttentionFusion(hidden_dim, num_heads=4)
        
        # 特征融合层
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多任务预测头
        self.task_heads = nn.ModuleDict({
            # 任务1: 药物-疾病关系存在性预测
            'existence_prediction': nn.Linear(fusion_dim // 2, 1),
            
            # 任务2: 蛋白质重要性评分
            'protein_importance': nn.Linear(fusion_dim // 2, 1),
            
            # 任务3: 通路预测
            'pathway_prediction': nn.Linear(fusion_dim // 2, num_pathways),
            
            # 任务4: 作用机制分类
            'mechanism_classification': nn.Linear(fusion_dim // 2, 5)  # 5种机制类型
        })
        
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, drug_emb, protein_emb, disease_emb):
        """
        前向传播
        
        Args:
            drug_emb: [batch_size, hidden_dim]
            protein_emb: [batch_size, hidden_dim] 
            disease_emb: [batch_size, hidden_dim]
            
        Returns:
            dict: 各任务的预测结果
        """
        # 注意力融合
        fused_repr, attention_weights = self.attention_fusion(drug_emb, protein_emb, disease_emb)
        
        # 特征融合
        fused_features = self.fusion_layers(fused_repr)
        
        # 多任务预测
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(fused_features)
        
        # 添加注意力权重用于解释性
        predictions['attention_weights'] = attention_weights
        
        return predictions


class TripleRelationRGCN(nn.Module):
    """完整的三元关系预测模型"""
    
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        num_pathways: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 3,
        fusion_dim: int = 256,
        dropout: float = 0.15
    ):
        super(TripleRelationRGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        # RGCN编码器
        self.encoder = EnhancedRGCNEncoder(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 三元关系预测器
        self.predictor = TripleRelationPredictor(
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            num_pathways=num_pathways,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
    
    def encode(self, x, edge_index, edge_type):
        """编码节点"""
        return self.encoder(x, edge_index, edge_type)
    
    def predict_triple_relations(
        self,
        node_embeddings,
        drug_indices,
        protein_indices,
        disease_indices
    ):
        """预测三元关系"""
        # 获取实体嵌入
        drug_emb = node_embeddings[drug_indices]
        protein_emb = node_embeddings[protein_indices]
        disease_emb = node_embeddings[disease_indices]
        
        # 预测
        predictions = self.predictor(drug_emb, protein_emb, disease_emb)
        
        return predictions
    
    def forward(
        self,
        x,
        edge_index,
        edge_type,
        drug_indices,
        protein_indices,
        disease_indices
    ):
        """完整的前向传播"""
        # 编码节点
        node_embeddings = self.encode(x, edge_index, edge_type)
        
        # 预测三元关系
        predictions = self.predict_triple_relations(
            node_embeddings, drug_indices, protein_indices, disease_indices
        )
        
        return predictions


def print_model_info(model):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"\n模型结构:")
    print(model)


if __name__ == "__main__":
    # 测试模型
    model = TripleRelationRGCN(
        num_nodes=1000,
        num_relations=20,
        num_pathways=100,
        hidden_dim=128
    )
    
    print_model_info(model)
    
    # 测试前向传播
    batch_size = 32
    x = torch.arange(1000)
    edge_index = torch.randint(0, 1000, (2, 2000))
    edge_type = torch.randint(0, 20, (2000,))
    
    drug_indices = torch.randint(0, 1000, (batch_size,))
    protein_indices = torch.randint(0, 1000, (batch_size,))
    disease_indices = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        predictions = model(x, edge_index, edge_type, drug_indices, protein_indices, disease_indices)
        
        print(f"\n预测结果形状:")
        for task, pred in predictions.items():
            if task != 'attention_weights':
                print(f"  {task}: {pred.shape}")
            else:
                print(f"  {task}: {pred.shape} (attention weights)")
    
    print("\n模型测试完成！")
