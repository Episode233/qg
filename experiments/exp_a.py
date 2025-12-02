import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch
from transformers.modeling_outputs import BaseModelOutput

# 引用基础组件
# 注意：train.py 会在根目录运行，所以这里使用 models.bart 是没问题的
from models.bart import get_bart_with_lora


class ExpaModel(nn.Module):
    def __init__(self, num_relations, gnn_layers=3, dropout=0.1):
        """
        EXPA 模型 (Experiment A)
        参数:
            num_relations: 关系种类的数量 (用于初始化 Embedding)
            gnn_layers: GNN 的层数 (建议 3 层以覆盖 3-hop)
            dropout: Dropout 比率
        """
        super().__init__()

        # 1. 加载 BART + LoRA
        # 这步会自动下载/加载模型并注入 LoRA，同时冻结非 LoRA 参数
        print(">>> Initializing BART with LoRA...")
        self.tokenizer, self.bart = get_bart_with_lora(lora_rank=8)

        # 获取隐藏层维度 (BART-base 是 768)
        self.d_model = self.bart.config.d_model

        # 2. 边向量化 (Trainable Embedding Table)
        # 专门用来学习 "spouse", "born_in" 这种关系的向量表示
        # 这是一个全新的、从头训练的矩阵
        self.relation_embedding = nn.Embedding(num_relations, self.d_model)

        # 3. 图推理 (Trainable GNN)
        # 使用 GATv2Conv，它支持 edge_dim 参数，允许我们将边向量注入到注意力计算中
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=self.d_model,
                    out_channels=self.d_model,
                    heads=4,  # 多头注意力 (4头)
                    concat=False,  # False表示输出维度不拼接，保持 d_model
                    dropout=dropout,
                    edge_dim=self.d_model  # 关键：告诉 GAT 边也有 768 维的特征
                )
            )

        # 4. 适配层 (Trainable Projection)
        # 负责把 GNN "揉搓"过的特征重新整理，让 BART Decoder 更容易接受
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def encode_nodes(self, input_ids, attention_mask):
        """
        Step 1: 文本 -> BART Encoder -> 节点初始向量
        """
        # 我们需要访问 BART 内部的 encoder
        # self.bart 是一个 PeftModel，它会自动把调用转发给底层的 BartForConditionalGeneration
        # .model 是 BartModel, .encoder 是 BartEncoder

        # 获取 Encoder 输出
        # outputs.last_hidden_state: [total_nodes, seq_len(32), 768]
        # 使用 get_encoder() 接口，它能自动处理层级关系
        encoder = self.bart.get_encoder()

        # 获取 Encoder 输出
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 策略：取第一个 Token (<s>) 作为该节点的代表向量
        # shape: [total_nodes, 768]
        node_embeddings = outputs.last_hidden_state[:, 0, :]
        return node_embeddings

    def forward(self, data):
        """
        训练时的前向传播 (Forward Pass)
        输入: PyG 的 Batch 对象
        输出: Loss (标量)
        """
        # --- A. 节点向量化 (Node Encoding) ---
        # data.x: [total_nodes, 32] (Token IDs)
        # x: [total_nodes, 768] (Vectors)
        x = self.encode_nodes(data.x, data.node_mask)

        # --- B. 图推理 (Graph Reasoning) ---
        # 1. 把关系 ID 变成向量
        # data.edge_attr: [num_edges] -> [num_edges, 768]
        edge_attr_emb = self.relation_embedding(data.edge_attr)

        # 2. GNN 迭代
        for gnn in self.gnn_layers:
            residual = x
            # GAT 计算: 融合了 邻居信息 + 边信息
            x = gnn(x, data.edge_index, edge_attr=edge_attr_emb)
            x = self.activation(x)
            x = self.dropout(x) + residual  # 残差连接，防止梯度消失

        # --- C. 适配层 (Projection) ---
        x = self.projection(x)

        # --- D. 重构为 Batch (Restructure) ---
        # GNN 输出的是大长条 [total_nodes, 768]
        # BART 需要的是 [batch_size, max_nodes_in_batch, 768]
        # data.batch 记录了每个节点属于哪个图
        encoder_hidden_states, encoder_attention_mask = to_dense_batch(x, data.batch)

        # --- E. 解码与 Loss 计算 (Decoder) ---
        # data.y: [batch_size, 64] (Target IDs)
        # 调用 BART 计算 Loss
        outputs = self.bart(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=encoder_attention_mask,  # 告诉 Decoder 哪里是填充的
            labels=data.y  # 目标，有了它就会自动算 Loss
        )

        return outputs.loss

    def generate(self, data, num_beams=4):
        """
        推理时的生成方法 (Generation)
        输入: PyG 的 Batch 对象 (通常 batch_size=1)
        输出: 生成的 Token IDs
        """
        # 前面的步骤 A, B, C, D 和 forward 一模一样
        x = self.encode_nodes(data.x, data.node_mask)

        edge_attr_emb = self.relation_embedding(data.edge_attr)
        for gnn in self.gnn_layers:
            x = x + self.dropout(self.activation(gnn(x, data.edge_index, edge_attr=edge_attr_emb)))

        x = self.projection(x)

        encoder_hidden_states, encoder_attention_mask = to_dense_batch(x, data.batch)

        # --- E. Beam Search 生成 ---
        # 使用 BART 的 generate 方法
        generated_ids = self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=encoder_attention_mask,
            max_length=64,  # 生成问题的最大长度
            num_beams=num_beams,  # Beam Search 宽度
            early_stopping=True,
            no_repeat_ngram_size=3  # 防止生成重复的话
        )

        return generated_ids