import os
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'facebook/bart-base'
CACHE_DIR = os.path.join(CURRENT_DIR, 'bart-base')


def get_bart_with_lora(lora_rank=8, lora_alpha=32):
    """
    加载 BART 模型，添加特殊 Token，并注入 LoRA 适配器。
    返回: tokenizer, model (PeftModel)
    """
    print(f"Loading BART from: {CACHE_DIR}")

    # 1. 加载 Tokenizer
    # local_files_only=True 确保不联网，只用本地
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)

    # 2. 添加 Special Tokens (关键步骤！)
    # 这些是我们之前在预处理时决定好的标记
    special_tokens_dict = {'additional_special_tokens': ['[TOPIC]', '[ANS]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens: {special_tokens_dict['additional_special_tokens']}")

    # 3. 加载 BART 模型
    # 使用 BartForConditionalGeneration，因为它包含了生成文本所需的 Head
    bart_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)

    # 4. 调整 Embedding 大小
    # 因为加了新 Token，原来的 Embedding 矩阵只有 50265，现在变大了，必须 resize
    bart_model.resize_token_embeddings(len(tokenizer))

    # 5. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # 任务类型：序列到序列
        inference_mode=False,  # 训练模式
        r=lora_rank,  # LoRA 的秩 (Rank)，通常 8 或 16
        lora_alpha=lora_alpha,  # 缩放系数，通常是 rank 的 2 倍
        lora_dropout=0.1,  # LoRA 层的 Dropout
        target_modules=["q_proj", "v_proj"]  # 只对 Attention 的 Query 和 Value 注入 LoRA
    )

    # 6. 注入 LoRA (Magic Happens Here!)
    # 这步操作会自动冻结 BART 原有参数，只把 LoRA 参数设为 requires_grad=True
    lora_model = get_peft_model(bart_model, peft_config)

    # 打印一下可训练参数的情况，确认 LoRA 生效
    print(">>> LoRA Injection Complete. Trainable Parameters:")
    lora_model.print_trainable_parameters()

    # 将模型移动到 GPU (如果可用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_model.to(device)

    return tokenizer, lora_model


# 为了方便测试，可以加一个 main
# if __name__ == "__main__":
#     tokenizer, model = get_bart_with_lora()
#     # 测试一下 Tokenizer 是否认得 [TOPIC]
#     test_str = "[TOPIC] Obama [ANS] USA"
#     print(f"Test Tokenization: {test_str} -> {tokenizer.encode(test_str)}")