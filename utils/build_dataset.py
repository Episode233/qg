import argparse
import os
import random
import networkx as nx
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from llm import generate_question

# ==========================================
# 1. 全局配置参数 (Path & Config)
# ==========================================

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "background_kbs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")


# ==========================================
# 2. 图处理工具函数 (Graph Utils)
# ==========================================

def load_kb(file_path):
    """加载 TXT Knowledge Base 到 NetworkX 有向图"""
    G = nx.DiGraph()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                h, r, t = parts
                G.add_edge(h, t, relation=r)
    return G


def get_k_hop_path_no_loop(G, start_node, k):
    """寻找一条严格无环的 k-hop 路径"""
    path_nodes = [start_node]
    path_edges = []

    curr = start_node
    visited = {start_node}

    for _ in range(k):
        neighbors = list(G.neighbors(curr))
        valid_neighbors = [n for n in neighbors if n not in visited]

        if not valid_neighbors:
            return None, None

        next_node = random.choice(valid_neighbors)
        r = G[curr][next_node]['relation']

        path_edges.append((curr, r, next_node))
        path_nodes.append(next_node)
        visited.add(next_node)
        curr = next_node

    return path_nodes, path_edges


def build_subgraph_data(G, path_nodes, path_edges, noise_m):
    """基于黄金路径构建加噪子图"""
    # A. 收集节点
    subgraph_nodes = set(path_nodes)
    for node in path_nodes:
        neighbors = list(G.neighbors(node))
        candidates = [n for n in neighbors if n not in subgraph_nodes]
        num_to_take = min(len(candidates), noise_m)
        if num_to_take > 0:
            subgraph_nodes.update(random.sample(candidates, num_to_take))

    subgraph_nodes_list = list(subgraph_nodes)

    # B. 局部索引
    node_to_idx = {node: i for i, node in enumerate(subgraph_nodes_list)}

    # C. 边处理 (双向)
    sub_G = G.subgraph(subgraph_nodes_list)
    edge_index_src = []
    edge_index_tgt = []
    edge_attr_list = []

    for u, v, data in sub_G.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        r = data['relation']

        # 正向
        edge_index_src.append(u_idx)
        edge_index_tgt.append(v_idx)
        edge_attr_list.append(r)

        # 反向
        edge_index_src.append(v_idx)
        edge_index_tgt.append(u_idx)
        edge_attr_list.append(r + "_inv")

    # D. 文本标记
    start_node = path_nodes[0]
    end_node = path_nodes[-1]

    labeled_nodes_text = []
    for node in subgraph_nodes_list:
        if node == start_node:
            labeled_nodes_text.append(f"[TOPIC] {node}")
        elif node == end_node:
            labeled_nodes_text.append(f"[ANS] {node}")
        else:
            labeled_nodes_text.append(node)

    return {
        "nodes": labeled_nodes_text,
        "edge_index": [edge_index_src, edge_index_tgt],
        "edge_attr": edge_attr_list,
        "label_ids": [node_to_idx[start_node], node_to_idx[end_node]]
    }


# ==========================================
# 3. 主逻辑
# ==========================================

def process_kb_file(kb_filename, num_hops, samples_per_node_attempts, noise_neighbors, target_total_samples):
    print(f"\n>>> Processing {kb_filename} | Hops: {num_hops}")

    # 确保文件名带后缀（如果用户没输的话）
    if not kb_filename.endswith(".txt"):
        kb_filename += ".txt"

    kb_path = os.path.join(DATA_DIR, kb_filename)

    # 检查文件是否存在
    if not os.path.exists(kb_path):
        print(f"[ERROR] File NOT found: {kb_path}")
        print(f"        Expected DATA_DIR: {DATA_DIR}")
        return

    G = load_kb(kb_path)
    all_nodes = list(G.nodes())
    print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    dataset_list = []
    global_seen_paths = set()

    # 只保留那些出度(out_degree)大于0的节点作为起点
    valid_start_nodes = [n for n in G.nodes() if G.out_degree(n) > 0]

    if len(valid_start_nodes) > target_total_samples:
        valid_start_nodes = random.sample(valid_start_nodes, target_total_samples)

    print(f"    Valid start nodes: {len(valid_start_nodes)} / {len(all_nodes)}")

    pbar = tqdm(valid_start_nodes, desc="    Sampling", unit="node")

    for start_node in pbar:
        # if len(dataset_list) >= 10:
        #     break
        for _ in range(samples_per_node_attempts):
            path_nodes, path_edges = get_k_hop_path_no_loop(G, start_node, num_hops)

            if not path_nodes: continue

            # 路径去重签名
            signature_parts = [path_nodes[0]]
            for _, r, v in path_edges:
                signature_parts.append(f"|{r}|{v}")
            path_signature = "".join(signature_parts)

            if path_signature in global_seen_paths: continue
            global_seen_paths.add(path_signature)

            # LLM 生成
            llm_path_str = path_nodes[0]
            for _, r, v in path_edges:
                llm_path_str += f" --({r})--> {v}"

            try:
                question = generate_question(llm_path_str, path_nodes[0], path_nodes[-1])
                if not question or "INVALID" in question: continue

                sample_data = build_subgraph_data(G, path_nodes, path_edges, noise_neighbors)
                sample_data['question'] = question
                dataset_list.append(sample_data)

            except Exception:
                continue

    print(f"    Generated {len(dataset_list)} samples.")

    if not dataset_list: return

    # 保存
    full_ds = Dataset.from_list(dataset_list).shuffle(seed=42)
    splits_1 = full_ds.train_test_split(test_size=0.2, seed=42)
    splits_2 = splits_1['test'].train_test_split(test_size=0.5, seed=42)

    final_dict = DatasetDict({
        'train': splits_1['train'],
        'validation': splits_2['train'],
        'test': splits_2['test']
    })

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    base_name = os.path.splitext(kb_filename)[0]
    save_folder_name = f"{base_name}_{num_hops}h"
    save_path = os.path.join(OUTPUT_DIR, save_folder_name)
    final_dict.save_to_disk(save_path)
    print(f"    Saved to: {save_path}")

    # ========== 新增：保存为JSON ==========
    # 保存为JSON格式
    for split_name, dataset in final_dict.items():
        json_filename = f"{save_folder_name}_{split_name}.json"
        json_save_path = os.path.join(save_path, json_filename)
        dataset.to_json(json_save_path)
        print(f"    JSON {split_name} saved to: {json_save_path}")


# ==========================================
# 4. 命令行入口 (Argparse Entry)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K-Hop QA dataset from Knowledge Base")

    # 定义命令行参数
    parser.add_argument("-k", "--kb", type=str, required=True, help="Knowledge Base filename (e.g., PQ.txt)")
    parser.add_argument("--hops", type=int, default=2, help="Number of hops for the reasoning path (default: 2)")
    parser.add_argument("--samples", type=int, default=3000, help="Target total number of starting nodes to sample (default: 3000)")
    parser.add_argument("--noise", type=int, default=5, help="Number of noise neighbors to add per node (default: 5)")
    parser.add_argument("--attempts", type=int, default=3, help="Number of attempts to generate paths per start node (default: 3)")
    args = parser.parse_args()

    # 调用主函数
    process_kb_file(
        kb_filename=args.kb,
        num_hops=args.hops,
        samples_per_node_attempts=args.attempts,
        noise_neighbors=args.noise,
        target_total_samples=args.samples
    )