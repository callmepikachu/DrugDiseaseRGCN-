import pandas as pd
import os
from tqdm import tqdm


def build_dataset(kg_file, positive_out, negative_out, overwrite=False):
    if os.path.exists(negative_out) and not overwrite:
        print(f"{negative_out} already exists. Set overwrite=True to rebuild.")
        return

    # 读取知识图谱
    df = pd.read_csv(kg_file, sep=",", dtype={"x_id": str, "y_id": str})

    # 1. 找到所有药物和疾病节点
    drugs = set(df.loc[df["x_type"] == "drug", "x_id"]).union(
        df.loc[df["y_type"] == "drug", "y_id"]
    )
    diseases = set(df.loc[df["x_type"] == "disease", "x_id"]).union(
        df.loc[df["y_type"] == "disease", "y_id"]
    )

    # 2. 找出所有 drug–disease 的边（不论关系类型）
    all_drug_disease_edges = set()
    drug_disease_relation_count = {"indication": 0, "off-label use": 0, "contraindication": 0, "other": 0}

    print("Scanning drug–disease edges...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Edges processed"):
        if row["x_type"] == "drug" and row["y_type"] == "disease":
            pair = (row["x_id"], row["y_id"])
        elif row["x_type"] == "disease" and row["y_type"] == "drug":
            pair = (row["y_id"], row["x_id"])
        else:
            continue

        all_drug_disease_edges.add(pair)

        # 分类统计
        rel = row["relation"].lower()
        if "indication" in rel and "off" not in rel:  # 只算 approved indication
            drug_disease_relation_count["indication"] += 1
        elif "off-label" in rel:
            drug_disease_relation_count["off-label use"] += 1
        elif "contraindication" in rel:
            drug_disease_relation_count["contraindication"] += 1
        else:
            drug_disease_relation_count["other"] += 1

    # 3. 筛选正样本：indication
    print("Extracting positive samples...")
    pos_records = []
    for _, row in tqdm(df[df["relation"] == "indication"].iterrows(),
                       total=(df["relation"] == "indication").sum(),
                       desc="Positive samples"):
        if row["x_type"] == "drug" and row["y_type"] == "disease":
            pair = (row["x_id"], row["y_id"])
        elif row["x_type"] == "disease" and row["y_type"] == "drug":
            pair = (row["y_id"], row["x_id"])
        else:
            continue
        pos_records.append(pair)

    pd.DataFrame(pos_records, columns=["drug_id", "disease_id"]).to_csv(
        positive_out, index=False
    )
    print(f"✅ Saved {len(pos_records)} positive samples to {positive_out}")

    # 4. 负样本全集 = 所有可能对 - 所有已有边
    print("Generating negative samples (this may take a while)...")
    all_pairs = set()
    for d in tqdm(drugs, desc="Drugs"):
        for dis in diseases:
            all_pairs.add((d, dis))

    neg_records = list(all_pairs - all_drug_disease_edges)

    pd.DataFrame(neg_records, columns=["drug_id", "disease_id"]).to_csv(
        negative_out, index=False
    )
    print(f"✅ Saved {len(neg_records)} negative samples to {negative_out}")

    # ------------------- 统计信息 -------------------
    print("\n===== Dataset Statistics =====")
    print(f"Drugs: {len(drugs)}")
    print(f"Diseases: {len(diseases)}")
    print(f"Drug–disease edges (total): {len(all_drug_disease_edges)}")
    print(f"  - Indication:       {drug_disease_relation_count['indication']}")
    print(f"  - Off-label use:    {drug_disease_relation_count['off-label use']}")
    print(f"  - Contraindication: {drug_disease_relation_count['contraindication']}")
    print(f"  - Other:            {drug_disease_relation_count['other']}")
    print(f"Positive (indication): {len(pos_records)}")
    print(f"Negative (no edge):    {len(neg_records)}")
    print("==============================\n")


# ------------------- 查询功能 -------------------

class NegativeSampler:
    def __init__(self, negative_csv):
        self.neg_df = pd.read_csv(negative_csv)

    def get_negatives_by_drug(self, drug_id):
        """给定 drug_id，返回所有没有关系的 disease_id 列表"""
        return self.neg_df[self.neg_df["drug_id"] == drug_id]["disease_id"].tolist()

    def get_negatives_by_disease(self, disease_id):
        """给定 disease_id，返回所有没有关系的 drug_id 列表"""
        return self.neg_df[self.neg_df["disease_id"] == disease_id]["drug_id"].tolist()


if __name__ == "__main__":
    # 生成数据集（只需跑一次）
    # build_dataset(
    #     "data/raw/kg.csv",
    #     "data/processed/positive.csv",
    #     "data/processed/negative.csv",
    #     overwrite=True  # 若要重新生成负样本全集，设为 True
    # )

    # 示例：查询
    sampler = NegativeSampler("data/processed/negative.csv")
    print("Example negatives for drug DB00492:", sampler.get_negatives_by_drug("DB00492")[:10])
    print("Example negatives for disease 5044:", sampler.get_negatives_by_disease("5044")[:10])
