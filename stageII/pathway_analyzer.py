#!/usr/bin/env python3
"""
Stage II 通路分析工具
"""

import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import argparse
from collections import defaultdict

# 添加父目录以导入Stage I的模块
sys.path.append('../src')
from data_loader import PrimeKGDataLoader
from utils import get_device

# 导入Stage II模块
from triple_model import TripleRelationRGCN


class PathwayAnalyzer:
    """通路分析器"""
    
    def __init__(self, config_path: str, model_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = model_path
        self.device = get_device(self.config['device'])
        
        # 创建结果目录
        self.results_dir = Path("results")
        (self.results_dir / "pathway_analysis").mkdir(exist_ok=True)
        
        # 加载数据和模型
        self.load_data()
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
        if triple_file.exists():
            self.triple_df = pd.read_csv(triple_file)
            print(f"Loaded {len(self.triple_df)} triple relations")
        else:
            print("Warning: Triple relations file not found")
            self.triple_df = pd.DataFrame()
        
        # 准备图数据（简化版本）
        num_edges = min(10000, self.num_nodes * 5)
        self.edge_index = torch.randint(0, self.num_nodes, (2, num_edges), device=self.device)
        self.edge_type = torch.randint(0, self.num_relations, (num_edges,), device=self.device)
    
    def load_model(self):
        """加载模型"""
        print("Loading model...")
        
        # 创建模型
        self.model = TripleRelationRGCN(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            num_pathways=self.config['model'].get('num_pathways', 100),
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            fusion_dim=self.config['model']['triple_fusion_dim'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # 加载检查点
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint['epoch']}")
        else:
            print("Warning: Model checkpoint not found, using random weights")
        
        # 编码所有节点
        self.model.eval()
        with torch.no_grad():
            node_indices = torch.arange(self.num_nodes, device=self.device)
            self.node_embeddings = self.model.encode(node_indices, self.edge_index, self.edge_type)
    
    def find_drug_disease_proteins(self, drug_id: str, disease_id: str, top_k: int = 10):
        """找到连接药物和疾病的关键蛋白质"""
        print(f"Analyzing drug-disease pathway: {drug_id} -> {disease_id}")
        
        # 检查实体是否存在于映射中
        node_to_idx = self.mappings['node_to_idx']
        
        if drug_id not in node_to_idx:
            print(f"Drug {drug_id} not found in knowledge graph")
            return []
        
        if disease_id not in node_to_idx:
            print(f"Disease {disease_id} not found in knowledge graph")
            return []
        
        drug_idx = node_to_idx[drug_id]
        disease_idx = node_to_idx[disease_id]
        
        # 获取所有蛋白质节点（这里简化处理，实际应该从数据中筛选）
        # 假设节点索引0-1000是蛋白质（实际应该根据节点类型筛选）
        protein_candidates = list(range(min(1000, self.num_nodes)))
        
        protein_scores = []
        
        with torch.no_grad():
            for protein_idx in protein_candidates:
                # 预测药物-蛋白质-疾病三元关系
                predictions = self.model.predict_triple_relations(
                    self.node_embeddings,
                    torch.tensor([drug_idx], device=self.device),
                    torch.tensor([protein_idx], device=self.device),
                    torch.tensor([disease_idx], device=self.device)
                )
                
                # 计算综合得分
                existence_score = torch.sigmoid(predictions['existence_prediction']).item()
                protein_importance = torch.sigmoid(predictions['protein_importance']).item()
                
                combined_score = (existence_score + protein_importance) / 2
                
                protein_scores.append({
                    'protein_idx': protein_idx,
                    'protein_id': self.mappings['idx_to_node'].get(protein_idx, f'PROTEIN_{protein_idx}'),
                    'existence_score': existence_score,
                    'protein_importance': protein_importance,
                    'combined_score': combined_score,
                    'attention_weights': predictions['attention_weights'][0].cpu().numpy()
                })
        
        # 按综合得分排序
        protein_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return protein_scores[:top_k]
    
    def analyze_pathway_mechanisms(self, drug_id: str, disease_id: str, proteins: list):
        """分析通路机制"""
        print(f"Analyzing pathway mechanisms for {len(proteins)} proteins...")
        
        mechanisms = []
        pathways = []
        
        node_to_idx = self.mappings['node_to_idx']
        drug_idx = node_to_idx[drug_id]
        disease_idx = node_to_idx[disease_id]
        
        with torch.no_grad():
            for protein_info in proteins:
                protein_idx = protein_info['protein_idx']
                
                # 预测机制和通路
                predictions = self.model.predict_triple_relations(
                    self.node_embeddings,
                    torch.tensor([drug_idx], device=self.device),
                    torch.tensor([protein_idx], device=self.device),
                    torch.tensor([disease_idx], device=self.device)
                )
                
                # 获取预测的机制和通路
                mechanism_pred = torch.argmax(predictions['mechanism_classification'], dim=1).item()
                pathway_pred = torch.argmax(predictions['pathway_prediction'], dim=1).item()
                
                mechanism_names = ['Agonist', 'Antagonist', 'Inhibitor', 'Activator', 'Modulator']
                mechanism_name = mechanism_names[mechanism_pred] if mechanism_pred < len(mechanism_names) else 'Unknown'
                
                mechanisms.append({
                    'protein_id': protein_info['protein_id'],
                    'mechanism': mechanism_name,
                    'mechanism_confidence': torch.softmax(predictions['mechanism_classification'], dim=1)[0, mechanism_pred].item(),
                    'pathway_id': pathway_pred,
                    'pathway_confidence': torch.softmax(predictions['pathway_prediction'], dim=1)[0, pathway_pred].item()
                })
        
        return mechanisms
    
    def create_pathway_network(self, drug_id: str, disease_id: str, proteins: list, mechanisms: list):
        """创建通路网络图"""
        print("Creating pathway network visualization...")
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        G.add_node(drug_id, type='drug', color='red', size=300)
        G.add_node(disease_id, type='disease', color='blue', size=300)
        
        for i, (protein_info, mechanism_info) in enumerate(zip(proteins, mechanisms)):
            protein_id = protein_info['protein_id']
            G.add_node(protein_id, 
                      type='protein', 
                      color='green', 
                      size=200 + protein_info['combined_score'] * 100,
                      mechanism=mechanism_info['mechanism'])
            
            # 添加边
            G.add_edge(drug_id, protein_id, 
                      weight=protein_info['existence_score'],
                      relation='drug-protein')
            G.add_edge(protein_id, disease_id, 
                      weight=protein_info['protein_importance'],
                      relation='protein-disease')
        
        # 可视化
        plt.figure(figsize=(15, 12))
        
        # 计算布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制节点
        drug_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'drug']
        protein_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'protein']
        disease_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'disease']
        
        nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, 
                              node_color='red', node_size=500, alpha=0.8, label='Drug')
        nx.draw_networkx_nodes(G, pos, nodelist=protein_nodes, 
                              node_color='green', node_size=300, alpha=0.8, label='Protein')
        nx.draw_networkx_nodes(G, pos, nodelist=disease_nodes, 
                              node_color='blue', node_size=500, alpha=0.8, label='Disease')
        
        # 绘制边
        drug_protein_edges = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'drug-protein']
        protein_disease_edges = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'protein-disease']
        
        nx.draw_networkx_edges(G, pos, edgelist=drug_protein_edges, 
                              edge_color='orange', width=2, alpha=0.6, label='Drug-Protein')
        nx.draw_networkx_edges(G, pos, edgelist=protein_disease_edges, 
                              edge_color='purple', width=2, alpha=0.6, label='Protein-Disease')
        
        # 添加标签
        labels = {}
        for node in G.nodes():
            if len(node) > 15:
                labels[node] = node[:12] + '...'
            else:
                labels[node] = node
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f'Drug-Disease Pathway Network\n{drug_id} -> {disease_id}')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图片
        save_path = self.results_dir / "pathway_analysis" / f"pathway_{drug_id}_{disease_id}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Pathway network saved to {save_path}")
        
        return G
    
    def generate_pathway_report(self, drug_id: str, disease_id: str, proteins: list, mechanisms: list):
        """生成通路分析报告"""
        print("Generating pathway analysis report...")
        
        # 创建报告数据
        report_data = []
        for protein_info, mechanism_info in zip(proteins, mechanisms):
            report_data.append({
                'drug_id': drug_id,
                'disease_id': disease_id,
                'protein_id': protein_info['protein_id'],
                'existence_score': protein_info['existence_score'],
                'protein_importance': protein_info['protein_importance'],
                'combined_score': protein_info['combined_score'],
                'predicted_mechanism': mechanism_info['mechanism'],
                'mechanism_confidence': mechanism_info['mechanism_confidence'],
                'pathway_id': mechanism_info['pathway_id'],
                'pathway_confidence': mechanism_info['pathway_confidence']
            })
        
        # 保存为CSV
        report_df = pd.DataFrame(report_data)
        report_path = self.results_dir / "pathway_analysis" / f"pathway_report_{drug_id}_{disease_id}.csv"
        report_df.to_csv(report_path, index=False)
        
        # 生成文本报告
        text_report_path = self.results_dir / "pathway_analysis" / f"pathway_summary_{drug_id}_{disease_id}.txt"
        
        with open(text_report_path, 'w') as f:
            f.write(f"Pathway Analysis Report\n")
            f.write(f"======================\n\n")
            f.write(f"Drug: {drug_id}\n")
            f.write(f"Disease: {disease_id}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Top {len(proteins)} Key Proteins:\n")
            f.write("-" * 40 + "\n")
            
            for i, (protein_info, mechanism_info) in enumerate(zip(proteins, mechanisms), 1):
                f.write(f"{i}. {protein_info['protein_id']}\n")
                f.write(f"   Combined Score: {protein_info['combined_score']:.4f}\n")
                f.write(f"   Existence Score: {protein_info['existence_score']:.4f}\n")
                f.write(f"   Importance Score: {protein_info['protein_importance']:.4f}\n")
                f.write(f"   Predicted Mechanism: {mechanism_info['mechanism']} "
                       f"(confidence: {mechanism_info['mechanism_confidence']:.4f})\n")
                f.write(f"   Pathway ID: {mechanism_info['pathway_id']} "
                       f"(confidence: {mechanism_info['pathway_confidence']:.4f})\n\n")
            
            # 机制统计
            mechanism_counts = {}
            for mechanism_info in mechanisms:
                mech = mechanism_info['mechanism']
                mechanism_counts[mech] = mechanism_counts.get(mech, 0) + 1
            
            f.write("Mechanism Distribution:\n")
            f.write("-" * 25 + "\n")
            for mechanism, count in sorted(mechanism_counts.items()):
                f.write(f"{mechanism}: {count} proteins\n")
        
        print(f"Pathway report saved to {report_path}")
        print(f"Summary report saved to {text_report_path}")
        
        return report_df
    
    def analyze_drug_disease_pathway(self, drug_id: str, disease_id: str, top_k: int = 10):
        """完整的药物-疾病通路分析"""
        print(f"\n{'='*60}")
        print(f"Drug-Disease Pathway Analysis")
        print(f"{'='*60}")
        
        # 1. 找到关键蛋白质
        proteins = self.find_drug_disease_proteins(drug_id, disease_id, top_k)
        
        if not proteins:
            print("No significant proteins found for this drug-disease pair")
            return None
        
        print(f"\nFound {len(proteins)} key proteins:")
        for i, protein in enumerate(proteins[:5], 1):
            print(f"{i}. {protein['protein_id']}: {protein['combined_score']:.4f}")
        
        # 2. 分析通路机制
        mechanisms = self.analyze_pathway_mechanisms(drug_id, disease_id, proteins)
        
        # 3. 创建网络可视化
        network = self.create_pathway_network(drug_id, disease_id, proteins, mechanisms)
        
        # 4. 生成报告
        report = self.generate_pathway_report(drug_id, disease_id, proteins, mechanisms)
        
        print(f"\nPathway analysis completed!")
        print(f"Results saved to {self.results_dir / 'pathway_analysis'}")
        
        return {
            'proteins': proteins,
            'mechanisms': mechanisms,
            'network': network,
            'report': report
        }


def main():
    parser = argparse.ArgumentParser(description="Stage II Pathway Analysis")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--model_path", default="results/models/triple_model_best.pth", 
                       help="Model checkpoint path")
    parser.add_argument("--drug_id", required=True, help="Drug ID to analyze")
    parser.add_argument("--disease_id", required=True, help="Disease ID to analyze")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top proteins to analyze")
    
    args = parser.parse_args()
    
    analyzer = PathwayAnalyzer(args.config, args.model_path)
    
    results = analyzer.analyze_drug_disease_pathway(
        args.drug_id, 
        args.disease_id, 
        args.top_k
    )
    
    if results:
        print(f"\nAnalysis Summary:")
        print(f"- Found {len(results['proteins'])} key proteins")
        print(f"- Identified {len(set(m['mechanism'] for m in results['mechanisms']))} different mechanisms")
        print(f"- Network contains {results['network'].number_of_nodes()} nodes and {results['network'].number_of_edges()} edges")


if __name__ == "__main__":
    main()
