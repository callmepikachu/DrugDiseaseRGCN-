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
        
        # 正负样本文件路径
        self.positive_file = os.path.join(self.processed_dir, "positive.csv")
        self.negative_file = os.path.join(self.processed_dir, "negative.csv")
        
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
    
    def load_positive_samples(self) -> pd.DataFrame:
        """加载正样本数据"""
        if not os.path.exists(self.positive_file):
            raise FileNotFoundError(f"正样本文件不存在: {self.positive_file}")
        
        print("正在加载正样本数据...")
        df = pd.read_csv(self.positive_file)
        print(f"正样本数据形状: {df.shape}")
        return df
    
    def load_negative_samples(self) -> pd.DataFrame:
        """加载负样本数据"""
        if not os.path.exists(self.negative_file):
            raise FileNotFoundError(f"负样本文件不存在: {self.negative_file}")
        
        print("正在加载负样本数据...")
        # 由于负样本文件可能很大，我们只加载部分数据
        # 在实际使用中，可能需要根据需要加载全部或部分数据
        # df = pd.read_csv(self.negative_file)  # 限制加载10万行用于测试
        df = None
        # print(f"负样本数据形状: {df.shape}")
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