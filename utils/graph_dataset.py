import torch
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class ExpDataset(Dataset):
    def __init__(self, hf_dataset, relation_vocab_path, tokenizer, max_node_len=32, max_tgt_len=64):
        """
        参数:
            hf_dataset: HuggingFace Dataset 对象 (Arrow格式，已经 load_from_disk 好的)
            relation_vocab_path: 关系字典(relations.json)的路径
            tokenizer: BART Tokenizer
        """
        # 1. 持有数据引用 (Arrow 格式，高效内存映射)
        # 程序只认这个，不看那些 debug 用的 json
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_node_len = max_node_len
        self.max_tgt_len = max_tgt_len

        # 2. 加载关系映射表 (这是唯一读取的 JSON)
        # 作用: 把 "spouse" 变成 0
        with open(relation_vocab_path, 'r', encoding='utf-8') as f:
            self.rel2id = json.load(f)

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        核心方法：只在需要时（DataLoader取数据时）才被调用。
        返回: 一个 PyG Data 对象
        """
        # A. 从 Arrow 中取出一条原始数据
        # 格式: {'nodes': [...], 'edge_index': [[...],[...]], 'edge_attr': [...], 'question': "..."}
        item = self.data[idx]

        # B. 处理节点 (Nodes) -> Token IDs
        # 输入是字符串列表: ["[TOPIC] Obama", "Hawaii", ...]
        node_encodings = self.tokenizer(
            item['nodes'],
            padding='max_length',  # 强制填充到 32
            truncation=True,  # 超长截断
            max_length=self.max_node_len,
            return_tensors='pt',  # 返回 PyTorch Tensor
            add_special_tokens=True
        )
        x_ids = node_encodings['input_ids']  # Shape: [num_nodes, 32]
        x_mask = node_encodings['attention_mask']  # Shape: [num_nodes, 32]

        # C. 处理边 (Edges) -> Relation IDs
        # 边索引直接转 Tensor
        edge_index = torch.tensor(item['edge_index'], dtype=torch.long)  # Shape: [2, num_edges]

        # 边属性查表: "spouse" -> 0
        # 如果遇到字典里没有的关系(极少见)，默认给 0，防止报错
        rel_ids = [self.rel2id.get(r, 0) for r in item['edge_attr']]
        edge_attr = torch.tensor(rel_ids, dtype=torch.long)  # Shape: [num_edges]

        # D. 处理目标问题 (Target Question) -> Label IDs
        target_encoding = self.tokenizer(
            item['question'],
            padding='max_length',
            truncation=True,
            max_length=self.max_tgt_len,
            return_tensors='pt'
        )
        labels = target_encoding['input_ids'].view(1, -1)

        # 将 Padding 部分的 ID (通常是1) 设为 -100
        # 这样计算 Loss 时会自动忽略这些位置
        labels[labels == self.tokenizer.pad_token_id] = -100

        # E. 组装成 PyG Data
        data = Data(
            x=x_ids,  # 节点 Token IDs
            node_mask=x_mask,  # 节点 Mask
            edge_index=edge_index,  # 拓扑结构
            edge_attr=edge_attr,  # 关系 IDs
            y=labels  # 目标问题 IDs
        )

        return data


class ExpCollator:
    """
    DataLoader 的胶水函数。
    作用: 把 DataLoader 取出的 [Data1, Data2, Data3, Data4] 拼成一个 Batch 对象。
    """

    def __call__(self, batch_list):
        # PyG 的 Batch.from_data_list 会自动处理图的拼接
        # 例如: 自动修正 edge_index 的偏移量
        return Batch.from_data_list(batch_list)