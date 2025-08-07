"""
模型评估脚本
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import PrimeKGDataLoader
from model import DrugDiseaseRGCN
from utils import (
    load_config, get_device, load_model, calculate_metrics,
    prepare_multitask_data, split_edges
)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config_path: str, model_path: str):
        self.config = load_config(config_path)
        self.model_path = model_path
        self.device = get_device(self.config['device'])
        
        # 加载数据
        self.data_loader = PrimeKGDataLoader(self.config['data']['data_dir'])
        self.load_data()
        
        # 加载模型
        self.load_model()
    
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        hetero_data, mappings, drug_disease_df = self.data_loader.load_processed_data()
        
        self.mappings = mappings
        self.num_nodes = mappings['num_nodes']
        self.num_relations = mappings['num_relations']
        
        # 筛选目标关系
        target_relations = self.config['data']['target_relations']
        if target_relations:
            mask = drug_disease_df['display_relation'].isin(target_relations)
            drug_disease_df = drug_disease_df[mask]
        
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
        
        # 分割数据
        train_data, val_data, test_data = split_edges(
            edge_index, edge_type,
            test_ratio=self.config['data']['test_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            seed=self.config['seed']
        )
        
        # 准备测试数据
        self.test_data = self._prepare_data_for_evaluation(test_data)
        
        # 创建完整图用于编码
        self.full_edge_index = edge_index.to(self.device)
        self.full_edge_type = edge_type.to(self.device)
        
        print(f"测试集大小: {len(self.test_data['existence_labels'])}")
    
    def _prepare_data_for_evaluation(self, data):
        """准备多任务评估数据"""
        edge_index, existence_labels, relation_labels, _ = prepare_multitask_data(
            data['edge_index'],
            data['edge_type'],
            self.num_nodes,
            self.config['data']['negative_sampling_ratio']
        )

        return {
            'edge_index': edge_index.to(self.device),
            'existence_labels': existence_labels.to(self.device),
            'relation_labels': relation_labels.to(self.device)
        }
    
    def load_model(self):
        """加载模型"""
        print("正在加载模型...")
        
        self.model = DrugDiseaseRGCN(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            **self.config['model']
        ).to(self.device)
        
        # 创建优化器（加载检查点需要）
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # 加载检查点
        epoch, loss, metrics = load_model(self.model, optimizer, self.model_path, self.device)
        
        print(f"已加载模型 (epoch {epoch}, loss: {loss:.4f})")
        if metrics:
            print("训练时的最佳指标:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    def evaluate(self):
        """评估模型"""
        print("正在评估模型...")
        
        self.model.eval()
        
        with torch.no_grad():
            # 编码所有节点
            node_indices = torch.arange(self.num_nodes, device=self.device)
            node_embeddings = self.model.encode(
                node_indices, self.full_edge_index, self.full_edge_type
            )
            
            # 多任务预测
            existence_scores, relation_logits = self.model.predict_links(
                node_embeddings,
                self.test_data['edge_index'][0],
                self.test_data['edge_index'][1]
            )

            # 关系存在性评估
            existence_y_true = self.test_data['existence_labels'].cpu().numpy()
            existence_y_score = torch.sigmoid(existence_scores).cpu().numpy()
            existence_y_pred = (existence_y_score > 0.5).astype(int)

            # 计算关系存在性指标
            existence_metrics = calculate_metrics(
                existence_y_true, existence_y_pred, existence_y_score,
                self.config['evaluation']['k_values']
            )

            # 关系类型评估（只对正样本）
            positive_mask = self.test_data['existence_labels'] == 1
            if positive_mask.sum() > 0:
                relation_y_true = self.test_data['relation_labels'][positive_mask].cpu().numpy()
                relation_y_pred = torch.argmax(relation_logits[positive_mask], dim=1).cpu().numpy()
                relation_accuracy = (relation_y_true == relation_y_pred).mean()
            else:
                relation_accuracy = 0.0

            # 合并指标
            metrics = {}
            for key, value in existence_metrics.items():
                metrics[f'existence_{key}'] = value
            metrics['relation_accuracy'] = relation_accuracy
        
        return metrics, existence_y_true, existence_y_pred, existence_y_score
    
    def detailed_analysis(self, y_true, y_pred, y_score):
        """详细分析"""
        print("\n" + "="*50)
        print("详细分析结果")
        print("="*50)
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=['负样本', '正样本']))
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n混淆矩阵:")
        print(f"真负例: {cm[0,0]}, 假正例: {cm[0,1]}")
        print(f"假负例: {cm[1,0]}, 真正例: {cm[1,1]}")
        
        # 按关系类型分析
        self._analyze_by_relation_type(y_true, y_pred, y_score)
        
        # 可视化
        self._create_visualizations(y_true, y_pred, y_score, cm)
    
    def _analyze_by_relation_type(self, y_true, y_pred, y_score):
        """按关系类型分析"""
        print("\n按关系类型分析:")

        # 获取关系类型（只分析正样本）
        positive_mask = self.test_data['existence_labels'] == 1
        if positive_mask.sum() == 0:
            print("  没有正样本进行关系类型分析")
            return

        relation_types = self.test_data['relation_labels'][positive_mask].cpu().numpy()
        relation_names = self.mappings['relation_encoder'].classes_

        # 只分析存在关系的样本
        pos_y_true = y_true[positive_mask]
        pos_y_pred = y_pred[positive_mask]
        pos_y_score = y_score[positive_mask]

        for rel_idx, rel_name in enumerate(relation_names):
            if rel_name in self.config['data']['target_relations']:
                mask = relation_types == rel_idx
                if mask.sum() > 0:
                    rel_y_true = pos_y_true[mask]
                    rel_y_pred = pos_y_pred[mask]
                    rel_y_score = pos_y_score[mask]

                    rel_metrics = calculate_metrics(rel_y_true, rel_y_pred, rel_y_score)

                    print(f"\n{rel_name}:")
                    print(f"  样本数: {mask.sum()}")
                    print(f"  AUC: {rel_metrics['auc']:.4f}")
                    print(f"  AP: {rel_metrics['ap']:.4f}")
    
    def _create_visualizations(self, y_true, y_pred, y_score, cm):
        """创建可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 混淆矩阵热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('混淆矩阵')
        axes[0,0].set_xlabel('预测标签')
        axes[0,0].set_ylabel('真实标签')
        
        # 预测分数分布
        axes[0,1].hist(y_score[y_true==0], bins=50, alpha=0.7, label='负样本', density=True)
        axes[0,1].hist(y_score[y_true==1], bins=50, alpha=0.7, label='正样本', density=True)
        axes[0,1].set_title('预测分数分布')
        axes[0,1].set_xlabel('预测分数')
        axes[0,1].set_ylabel('密度')
        axes[0,1].legend()
        
        # ROC曲线
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[1,0].plot(fpr, tpr, label=f'ROC (AUC = {calculate_metrics(y_true, y_pred, y_score)["auc"]:.3f})')
        axes[1,0].plot([0, 1], [0, 1], 'k--', label='随机')
        axes[1,0].set_title('ROC曲线')
        axes[1,0].set_xlabel('假正例率')
        axes[1,0].set_ylabel('真正例率')
        axes[1,0].legend()
        
        # Precision-Recall曲线
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        axes[1,1].plot(recall, precision, label=f'PR (AP = {calculate_metrics(y_true, y_pred, y_score)["ap"]:.3f})')
        axes[1,1].set_title('Precision-Recall曲线')
        axes[1,1].set_xlabel('召回率')
        axes[1,1].set_ylabel('精确率')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.config['logging']['log_dir'], 'evaluation_plots.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化结果已保存到: {save_path}")
        
        plt.show()
    
    def save_predictions(self, existence_y_true, existence_y_pred, existence_y_score):
        """保存多任务预测结果"""
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'head_node': self.test_data['edge_index'][0].cpu().numpy(),
            'tail_node': self.test_data['edge_index'][1].cpu().numpy(),
            'true_existence': existence_y_true,
            'predicted_existence': existence_y_pred,
            'existence_score': existence_y_score,
            'true_relation_type': self.test_data['relation_labels'].cpu().numpy()
        })
        
        # 添加节点和关系名称
        idx_to_node = self.mappings['idx_to_node']
        relation_names = self.mappings['relation_encoder'].classes_
        
        results_df['head_node_id'] = results_df['head_node'].map(idx_to_node)
        results_df['tail_node_id'] = results_df['tail_node'].map(idx_to_node)
        results_df['true_relation_name'] = results_df['true_relation_type'].map(
            lambda x: relation_names[x] if x >= 0 and x < len(relation_names) else 'none'
        )
        
        # 保存结果
        save_path = os.path.join(self.config['logging']['log_dir'], 'predictions.csv')
        results_df.to_csv(save_path, index=False)
        print(f"预测结果已保存到: {save_path}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description="评估药物-疾病关系预测模型")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--model_path", required=True, help="模型检查点路径")
    parser.add_argument("--save_predictions", action="store_true", help="保存预测结果")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(args.config, args.model_path)
    
    # 评估模型
    metrics, y_true, y_pred, y_score = evaluator.evaluate()
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 详细分析
    evaluator.detailed_analysis(y_true, y_pred, y_score)
    
    # 保存预测结果
    if args.save_predictions:
        evaluator.save_predictions(y_true, y_pred, y_score)
    
    print("\n评估完成!")


if __name__ == "__main__":
    main()
