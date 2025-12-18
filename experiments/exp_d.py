import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch
from transformers.modeling_outputs import BaseModelOutput

from models.bart import get_bart_with_lora


class ExpDModel(nn.Module):
    def __init__(self, num_relations, gnn_layers=3, dropout=0.1):
        """
        ExpD: 结构增强模型 (Structure Injection)
        继承 ExpC 的融合架构，但在 GNN 输入端注入位置和类型信息。
        """
        super().__init__()

        # 1. 基础组件 (BART + LoRA)
        print(">>> Initializing BART with LoRA (Structure Injection Mode)...")
        self.tokenizer, self.bart = get_bart_with_lora(lora_rank=8)
        self.d_model = self.bart.config.d_model

        # --- A. 结构 Embeddings (ExpD 新增核心) ---
        # 节点类型: 0=Topic, 1=Ans, 2=Other
        self.node_type_embedding = nn.Embedding(3, self.d_model)

        # 跳数距离: 0, 1, 2... (预设最大支持 9 跳，足够了)
        # 如果 dataset 算出 >9，我们会在 forward 里截断
        self.max_hop_id = 9
        self.hop_embedding = nn.Embedding(self.max_hop_id + 1, self.d_model)

        # --- B. 图逻辑流组件 ---
        self.relation_embedding = nn.Embedding(num_relations, self.d_model)
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            self.gnn_layers.append(
                GATv2Conv(
                    self.d_model,
                    self.d_model,
                    heads=4,
                    concat=False,
                    dropout=dropout,
                    edge_dim=self.d_model
                )
            )

        # --- C. 融合门控 (同 ExpC) ---
        self.gate_net = nn.Sequential(
            nn.Linear(self.d_model * 2, 1),
            nn.Sigmoid()
        )

        # --- D. 适配层 ---
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # 用于 WandB 记录 Alpha
        self.alpha_stats = {}

    def encode_nodes(self, input_ids, attention_mask):
        encoder = self.bart.get_encoder()
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def _inject_structure(self, x_text, data):
        """
        ExpD 核心逻辑: 将结构信息注入到语义向量中
        """
        # 1. 类型 Embedding
        # data.node_type: [N] -> [N, 768]
        type_emb = self.node_type_embedding(data.node_type)

        # 2. 跳数 Embedding
        # 防止 BFS 算出来的距离超过 Embedding 表的大小
        hops = data.hop_id.clamp(max=self.max_hop_id)
        hop_emb = self.hop_embedding(hops)

        # 3. 注入 (相加)
        # x_struct 包含了: "字面意思" + "我是起点吗" + "我离起点多远"
        x_struct = x_text + type_emb + hop_emb

        return x_struct

    def forward(self, data):
        # 1. 纯语义流 (Semantic Stream)
        x_text = self.encode_nodes(data.x, data.node_mask)

        # 2. 结构注入 (Structure Injection)
        # 注意: 我们只把注入后的向量喂给 GNN，x_text 保持纯净用于 Fusion
        x_struct = self._inject_structure(x_text, data)

        # 3. 逻辑流 (Logic Stream - GNN)
        x_graph = x_struct  # GNN 的起点是增强后的向量
        edge_attr_emb = self.relation_embedding(data.edge_attr)

        for gnn in self.gnn_layers:
            residual = x_graph
            x_graph = gnn(x_graph, data.edge_index, edge_attr=edge_attr_emb)
            x_graph = self.activation(x_graph)
            x_graph = self.dropout(x_graph) + residual

        # 4. 双流融合 (Fusion)
        # 输入: [纯语义; 增强后的图逻辑]
        combined = torch.cat([x_text, x_graph], dim=-1)
        alpha = self.gate_net(combined)

        if self.training:
            with torch.no_grad():
                self.alpha_stats = {
                    "alpha/mean": alpha.mean().item(),
                    "alpha/std": alpha.std().item(),  # 标准差 (看是不是所有节点都一样)
                    "alpha/min": alpha.min().item(),
                    "alpha/max": alpha.max().item()
                }

        x_fused = alpha * x_graph + (1 - alpha) * x_text

        # 5. 解码
        x_final = self.projection(x_fused)
        encoder_hidden_states, encoder_attention_mask = to_dense_batch(x_final, data.batch)
        encoder_outputs_obj = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        outputs = self.bart(
            encoder_outputs=encoder_outputs_obj,
            attention_mask=encoder_attention_mask,
            labels=data.y
        )

        return outputs.loss

    def generate(self, data, num_beams=4):
        # 1. Encode
        x_text = self.encode_nodes(data.x, data.node_mask)

        # 2. Inject & GNN
        x_struct = self._inject_structure(x_text, data)
        x_graph = x_struct

        edge_attr_emb = self.relation_embedding(data.edge_attr)
        for gnn in self.gnn_layers:
            x_graph = x_graph + self.dropout(self.activation(gnn(x_graph, data.edge_index, edge_attr=edge_attr_emb)))

        # 3. Fusion
        combined = torch.cat([x_text, x_graph], dim=-1)
        alpha = self.gate_net(combined)
        x_fused = alpha * x_graph + (1 - alpha) * x_text

        # 4. Decode
        x_final = self.projection(x_fused)
        encoder_hidden_states, encoder_attention_mask = to_dense_batch(x_final, data.batch)
        encoder_outputs_obj = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        generated_ids = self.bart.generate(
            encoder_outputs=encoder_outputs_obj,
            attention_mask=encoder_attention_mask,
            max_length=64,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        return generated_ids