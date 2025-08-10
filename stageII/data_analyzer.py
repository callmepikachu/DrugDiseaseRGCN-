#!/usr/bin/env python3
"""
Stage II 数据分析器: 分析三元关系数据
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import networkx as nx
import argparse
from pathlib import Path

# 添加父目录到路径以导入Stage I的模块
sys.path.append('../src')
from data_loader import PrimeKGDataLoader


class TripleRelationAnalyzer:
    """三元关系数据分析器"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.data_loader = PrimeKGDataLoader(data_dir)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_and_filter_data(self):
        """加载并筛选相关数据"""
        print("Loading PrimeKG data...")
        df = self.data_loader.load_raw_data()
        
        # 首先检查实际存在的实体类型
        actual_x_types = set(df['x_type'].unique())
        actual_y_types = set(df['y_type'].unique())
        all_actual_types = actual_x_types | actual_y_types

        print(f"Actual entity types in data: {sorted(all_actual_types)}")

        # 筛选实际存在的目标实体类型
        target_entities = {'drug', 'disease', 'gene/protein', 'pathway'}  # 使用正确的实体类型
        available_targets = target_entities & all_actual_types

        print(f"Available target entities: {sorted(available_targets)}")

        mask = (df['x_type'].isin(available_targets)) & (df['y_type'].isin(available_targets))
        filtered_df = df[mask].copy()
        
        print(f"Original data: {len(df):,} edges")
        print(f"Filtered data: {len(filtered_df):,} edges")
        
        return filtered_df
    
    def analyze_entity_distribution(self, df):
        """分析实体分布"""
        print("\n=== Entity Distribution Analysis ===")
        
        # 统计各类型实体数量
        all_x_entities = set(zip(df['x_id'], df['x_type']))
        all_y_entities = set(zip(df['y_id'], df['y_type']))
        all_entities = all_x_entities | all_y_entities
        
        entity_counts = defaultdict(int)
        for entity_id, entity_type in all_entities:
            entity_counts[entity_type] += 1
        
        print("Entity type counts:")
        for entity_type, count in sorted(entity_counts.items()):
            print(f"  {entity_type}: {count:,}")
        
        # 可视化
        plt.figure(figsize=(10, 6))
        types = list(entity_counts.keys())
        counts = list(entity_counts.values())

        plt.bar(types, counts)
        plt.title('Entity Type Distribution')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'entity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return entity_counts
    
    def analyze_relation_patterns(self, df):
        """分析关系模式"""
        print("\n=== Relation Pattern Analysis ===")
        
        # 统计不同实体类型间的关系
        relation_patterns = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            x_type, y_type = row['x_type'], row['y_type']
            relation = row['display_relation']
            
            # 标准化关系方向
            if x_type <= y_type:
                pattern = f"{x_type}-{y_type}"
            else:
                pattern = f"{y_type}-{x_type}"
            
            relation_patterns[pattern][relation] += 1
        
        # 打印关系模式
        for pattern, relations in relation_patterns.items():
            print(f"\n{pattern} relations:")
            total = sum(relations.values())
            for relation, count in sorted(relations.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {relation}: {count:,} ({count/total*100:.1f}%)")
        
        return relation_patterns
    
    def build_triple_relations(self, df):
        """构建三元关系：药物-蛋白质-疾病"""
        print("\n=== Building Triple Relations ===")

        # 提取药物-蛋白质关系
        drug_protein_df = df[
            ((df['x_type'] == 'drug') & (df['y_type'] == 'gene/protein')) |
            ((df['x_type'] == 'gene/protein') & (df['y_type'] == 'drug'))
        ].copy()

        # 标准化方向：药物 -> 蛋白质
        drug_protein_relations = []
        for _, row in drug_protein_df.iterrows():
            if row['x_type'] == 'drug':
                drug_protein_relations.append({
                    'drug_id': row['x_id'],
                    'protein_id': row['y_id'],
                    'relation': row['display_relation']
                })
            else:
                drug_protein_relations.append({
                    'drug_id': row['y_id'],
                    'protein_id': row['x_id'],
                    'relation': row['display_relation']
                })

        drug_protein_df = pd.DataFrame(drug_protein_relations)
        print(f"Drug-Protein relations: {len(drug_protein_df):,}")

        # 提取蛋白质-疾病关系
        protein_disease_df = df[
            ((df['x_type'] == 'gene/protein') & (df['y_type'] == 'disease')) |
            ((df['x_type'] == 'disease') & (df['y_type'] == 'gene/protein'))
        ].copy()

        # 标准化方向：蛋白质 -> 疾病
        protein_disease_relations = []
        for _, row in protein_disease_df.iterrows():
            if row['x_type'] == 'gene/protein':
                protein_disease_relations.append({
                    'protein_id': row['x_id'],
                    'disease_id': row['y_id'],
                    'relation': row['display_relation']
                })
            else:
                protein_disease_relations.append({
                    'protein_id': row['y_id'],
                    'disease_id': row['x_id'],
                    'relation': row['display_relation']
                })

        protein_disease_df = pd.DataFrame(protein_disease_relations)
        print(f"Protein-Disease relations: {len(protein_disease_df):,}")

        # 构建三元组：药物-蛋白质-疾病
        triple_relations = []

        print("Building drug-protein-disease triples...")

        # 为了提高效率，我们只处理前1000个药物-蛋白质关系
        sample_drug_protein = drug_protein_df.head(1000)

        for _, dp_row in sample_drug_protein.iterrows():
            # 找到与该蛋白质相关的疾病
            related_diseases = protein_disease_df[
                protein_disease_df['protein_id'] == dp_row['protein_id']
            ]

            for _, pd_row in related_diseases.iterrows():
                triple_relations.append({
                    'drug_id': dp_row['drug_id'],
                    'protein_id': dp_row['protein_id'],
                    'disease_id': pd_row['disease_id'],
                    'drug_protein_relation': dp_row['relation'],
                    'protein_disease_relation': pd_row['relation']
                })

        triple_df = pd.DataFrame(triple_relations)
        print(f"Drug-Protein-Disease triples: {len(triple_df):,}")

        # 保存三元关系数据
        triple_df.to_csv(self.results_dir / 'triple_relations.csv', index=False)

        return triple_df
    
    def analyze_triple_statistics(self, triple_df):
        """分析三元关系统计"""
        print("\n=== Triple Relation Statistics ===")

        if len(triple_df) == 0:
            print("No triple relations to analyze")
            return {}

        # 统计每个药物关联的蛋白质数量
        drug_protein_counts = triple_df.groupby('drug_id')['protein_id'].nunique()
        print(f"Proteins per drug - Mean: {drug_protein_counts.mean():.2f}, "
              f"Median: {drug_protein_counts.median():.2f}, "
              f"Max: {drug_protein_counts.max()}")

        # 统计每个疾病关联的蛋白质数量
        disease_protein_counts = triple_df.groupby('disease_id')['protein_id'].nunique()
        print(f"Proteins per disease - Mean: {disease_protein_counts.mean():.2f}, "
              f"Median: {disease_protein_counts.median():.2f}, "
              f"Max: {disease_protein_counts.max()}")

        # 统计每个蛋白质关联的药物-疾病对数量
        protein_pair_counts = triple_df.groupby('protein_id').apply(
            lambda x: len(x[['drug_id', 'disease_id']].drop_duplicates())
        )
        print(f"Drug-disease pairs per protein - Mean: {protein_pair_counts.mean():.2f}, "
              f"Median: {protein_pair_counts.median():.2f}, "
              f"Max: {protein_pair_counts.max()}")

        # 关系类型组合分析
        relation_combinations = triple_df.groupby([
            'drug_protein_relation', 'protein_disease_relation'
        ]).size().sort_values(ascending=False)

        print(f"\nTop 10 relation combinations:")
        for (dp_rel, pd_rel), count in relation_combinations.head(10).items():
            print(f"  {dp_rel} -> {pd_rel}: {count:,}")

        return {
            'drug_protein_counts': drug_protein_counts,
            'disease_protein_counts': disease_protein_counts,
            'protein_pair_counts': protein_pair_counts,
            'relation_combinations': relation_combinations
        }
    
    def create_network_visualization(self, triple_df, max_nodes=100):
        """创建网络可视化"""
        print(f"\n=== Creating Network Visualization (max {max_nodes} nodes) ===")
        
        if len(triple_df) == 0:
            print("No triple relations to visualize")
            return

        # 选择最活跃的节点进行可视化
        top_drugs = triple_df['drug_id'].value_counts().head(10).index
        top_diseases = triple_df['disease_id'].value_counts().head(10).index
        top_proteins = triple_df['protein_id'].value_counts().head(15).index

        # 筛选子图
        subset_df = triple_df[
            (triple_df['drug_id'].isin(top_drugs)) &
            (triple_df['disease_id'].isin(top_diseases)) &
            (triple_df['protein_id'].isin(top_proteins))
        ]

        # 创建网络图
        G = nx.Graph()

        # 添加节点
        for drug_id in subset_df['drug_id'].unique():
            G.add_node(drug_id, type='drug')
        for protein_id in subset_df['protein_id'].unique():
            G.add_node(protein_id, type='protein')
        for disease_id in subset_df['disease_id'].unique():
            G.add_node(disease_id, type='disease')

        # 添加边
        for _, row in subset_df.iterrows():
            G.add_edge(row['drug_id'], row['protein_id'], type='drug-protein')
            G.add_edge(row['protein_id'], row['disease_id'], type='protein-disease')

        # 可视化
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # 绘制不同类型的节点
        drug_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'drug']
        protein_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'protein']
        disease_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'disease']

        nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color='red',
                              node_size=100, alpha=0.7, label='Drug')
        nx.draw_networkx_nodes(G, pos, nodelist=protein_nodes, node_color='green',
                              node_size=80, alpha=0.7, label='Protein')
        nx.draw_networkx_nodes(G, pos, nodelist=disease_nodes, node_color='blue',
                              node_size=120, alpha=0.7, label='Disease')

        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

        plt.title('Drug-Protein-Disease Network')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'triple_network.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Network visualization saved with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("Starting Stage II Data Analysis...")
        
        # 加载数据
        df = self.load_and_filter_data()
        
        # 分析实体分布
        entity_counts = self.analyze_entity_distribution(df)
        
        # 分析关系模式
        relation_patterns = self.analyze_relation_patterns(df)
        
        # 构建三元关系
        triple_df = self.build_triple_relations(df)
        
        # 分析三元关系统计
        triple_stats = self.analyze_triple_statistics(triple_df)
        
        # 创建网络可视化
        self.create_network_visualization(triple_df)
        
        print(f"\nAnalysis complete! Results saved to {self.results_dir}")
        
        return {
            'entity_counts': entity_counts,
            'relation_patterns': relation_patterns,
            'triple_relations': triple_df,
            'triple_stats': triple_stats
        }


def main():
    parser = argparse.ArgumentParser(description="Stage II Data Analysis")
    parser.add_argument("--data_dir", default="../data", help="Data directory")
    parser.add_argument("--analyze_triple_relations", action="store_true", 
                       help="Run triple relation analysis")
    
    args = parser.parse_args()
    
    analyzer = TripleRelationAnalyzer(args.data_dir)
    
    if args.analyze_triple_relations:
        results = analyzer.run_full_analysis()
        print("Triple relation analysis completed!")
    else:
        print("Use --analyze_triple_relations to run the analysis")


if __name__ == "__main__":
    main()
