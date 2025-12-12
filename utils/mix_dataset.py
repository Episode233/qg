import os
import sys
import shutil
from collections import defaultdict
from datasets import load_from_disk, concatenate_datasets, DatasetDict

# --- 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
processed_dir = os.path.join(project_root, "datasets", "processed")
mixed_dir = os.path.join(project_root, "datasets", "mixed")


def get_dataset_groups(source_dir):
    """扫描目录，按前缀分组"""
    groups = defaultdict(list)
    if not os.path.exists(source_dir):
        return {}

    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if not os.path.isdir(folder_path): continue
        if "dataset_dict.json" not in os.listdir(folder_path): continue

        if folder_name.endswith("_2h"):
            base_name = folder_name[:-3]
        elif folder_name.endswith("_3h"):
            base_name = folder_name[:-3]
        else:
            continue
        groups[base_name].append(folder_name)
    return groups


def mix_datasets_auto():
    if not os.path.exists(mixed_dir):
        os.makedirs(mixed_dir)
        print(f">>> Created output directory: {mixed_dir}")

    print(f">>> Scanning {processed_dir} ...")
    groups = get_dataset_groups(processed_dir)

    if not groups:
        print("No valid datasets found.")
        return

    print(f">>> Found {len(groups)} groups: {list(groups.keys())}\n")

    for base_name, dataset_folders in groups.items():
        dataset_folders.sort()
        target_name = f"{base_name}_mix"
        save_path = os.path.join(mixed_dir, target_name)

        print(f"=== Processing: {base_name} ===")
        print(f"  Sources: {dataset_folders}")

        train_list, val_list, test_list = [], [], []

        for name in dataset_folders:
            path = os.path.join(processed_dir, name)
            try:
                ds_dict = load_from_disk(path)
                train_list.append(ds_dict['train'])
                val_list.append(ds_dict['validation'])
                test_list.append(ds_dict['test'])
            except Exception as e:
                print(f"  [Error] {name}: {e}")

        if not train_list: continue

        # 合并
        full_train = concatenate_datasets(train_list).shuffle(seed=42)
        full_val = concatenate_datasets(val_list).shuffle(seed=42)
        full_test = concatenate_datasets(test_list)

        mixed_dataset = DatasetDict({
            'train': full_train,
            'validation': full_val,
            'test': full_test
        })

        # 1. 保存为 HF Dataset (二进制 Arrow)
        mixed_dataset.save_to_disk(save_path)
        print(f"  Saved Binary to: {save_path}")

        # 2. 【修改】保存为 JSON (保持与 build_dataset.py 一致)
        # 使用 datasets 库自带的 to_json 方法
        print("  Exporting JSON for inspection...")
        for split in ['train', 'validation', 'test']:
            json_filename = f"{target_name}_{split}.json"
            json_path = os.path.join(save_path, json_filename)

            # 直接调用 HF 的 API，这会生成 JSON Lines 格式
            mixed_dataset[split].to_json(json_path)
            print(f"    JSON {split} saved to: {json_path}")

        print(f"  Stats -> Train: {len(full_train)}, Val: {len(full_val)}, Test: {len(full_test)}")
        print("-" * 40)

    print("\n>>> All Done!")


if __name__ == "__main__":
    mix_datasets_auto()