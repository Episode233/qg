import os
import json
import argparse
from datasets import load_from_disk

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")


def build_relation_vocab(dataset_name):
    dataset_path = os.path.join(DATA_DIR, dataset_name)

    # 动态生成输出文件名: 例如 mix_all_kb_relations.json
    output_filename = f"{dataset_name}_relations.json"
    output_path = os.path.join(DATA_DIR, output_filename)

    print(f"Target Dataset: {dataset_name}")
    print(f"Checking path: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset not found at {dataset_path}")
        return

    unique_relations = set()

    try:
        print("Loading dataset dict...")
        ds_dict = load_from_disk(dataset_path)

        # 扫描 Train, Validation, Test 所有的 split
        # 确保不会遗漏任何稀有关系
        for split_name, dataset in ds_dict.items():
            print(f"  Scanning split: {split_name} ({len(dataset)} samples)...")
            for sample in dataset:
                rels = sample['edge_attr']
                unique_relations.update(rels)

    except Exception as e:
        print(f"[Error] Failed to process {dataset_name}: {e}")
        return

    # 排序并建立映射
    sorted_relations = sorted(list(unique_relations))
    rel2id = {rel: i for i, rel in enumerate(sorted_relations)}

    print(f"\nFound {len(rel2id)} unique relations in {dataset_name}.")

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f, indent=2, ensure_ascii=False)

    print(f"Relation vocabulary saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build relation vocab for a specific dataset")
    # 添加命令行参数 -d
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help="Name of the dataset folder (e.g., mix_all_kb)")

    args = parser.parse_args()

    build_relation_vocab(args.dataset)

# 为特定数据集生成词表
# python utils/build_vocab.py -d mix_all_kb