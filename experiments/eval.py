import os
import sys
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate  # HuggingFace 的评估库

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
metrics_dir = os.path.join(project_root, 'metrics')
sys.path.append(project_root)

from utils.graph_dataset import ExpaDataset, ExpaCollator
from models.bart import get_bart_with_lora
from datasets import load_from_disk


def compute_metrics(predictions, references):
    """
    计算 BLEU 和 ROUGE 分数
    """
    print(">>> Computing Metrics...")

    # 加载指标
    bleu = evaluate.load(os.path.join(metrics_dir, 'bleu'))
    rouge = evaluate.load(os.path.join(metrics_dir, 'rouge'))

    # 1. 计算 BLEU
    # BLEU 需要 references 是 list of list (因为可能有多参考答案，虽然这里只有1个)
    # refs_for_bleu = [[ref] for ref in references]
    # HuggingFace evaluate 的 bleu 实现比较简单，直接传 list 也行，
    # 但标准做法通常是 text.

    results = {}

    # 计算 BLEU-1, 2, 3, 4
    # max_order=4 默认就是 BLEU-4
    bleu_res = bleu.compute(predictions=predictions, references=references)
    results['BLEU'] = round(bleu_res['bleu'] * 100, 2)

    # 2. 计算 ROUGE (L)
    rouge_res = rouge.compute(predictions=predictions, references=references)
    results['ROUGE-1'] = round(rouge_res['rouge1'] * 100, 2)
    results['ROUGE-2'] = round(rouge_res['rouge2'] * 100, 2)
    results['ROUGE-L'] = round(rouge_res['rougeL'] * 100, 2)

    return results


def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Evaluating using: {device}")

    # 1. 路径准备
    processed_dir = os.path.join(project_root, "datasets", "processed")
    data_path = os.path.join(processed_dir, args.dataset)
    vocab_filename = f"{args.dataset}_relations.json"
    vocab_path = os.path.join(processed_dir, vocab_filename)

    # 结果保存路径
    result_save_dir = os.path.dirname(args.checkpoint)
    output_csv = os.path.join(result_save_dir, "test_predictions.csv")
    output_metrics = os.path.join(result_save_dir, "test_metrics.txt")

    # 2. 加载数据
    print(">>> Loading Data & Tokenizer...")
    tokenizer, _ = get_bart_with_lora()

    hf_ds_dict = load_from_disk(data_path)
    # 加载 Test 集
    test_dataset = ExpaDataset(hf_ds_dict['test'], vocab_path, tokenizer)
    collator = ExpaCollator()

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collator)

    num_relations = len(test_dataset.rel2id)

    # 3. 加载模型
    print(f">>> Loading Model from {args.checkpoint}...")

    if args.exp_name == 'a':
        from experiments.exp_a import ExpaModel
        model = ExpaModel(num_relations=num_relations, gnn_layers=args.gnn_layers)
    else:
        raise ValueError("Unknown experiment")

    # 加载权重
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. 推理循环
    print(">>> Starting Inference...")
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):
            batch = batch.to(device)

            # 使用 Beam Search 生成
            # num_beams=4 是标准配置，效果通常比 greedy search 好很多
            gen_ids = model.generate(batch, num_beams=4)

            # 解码 Prediction
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # 解码 Reference (Target)
            tgt_ids = batch.y.clone()
            tgt_ids[tgt_ids == -100] = tokenizer.pad_token_id
            golds = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)

            all_preds.extend(preds)
            all_golds.extend(golds)

    # 5. 保存生成的 Case
    df = pd.DataFrame({
        'Generated': all_preds,
        'Reference': all_golds
    })
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 6. 计算并打印分数
    scores = compute_metrics(all_preds, all_golds)

    print("\n" + "=" * 30)
    print("       EVALUATION REPORT       ")
    print("=" * 30)
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {os.path.basename(args.checkpoint)}")
    print("-" * 30)
    print(f"BLEU:    {scores['BLEU']}")
    print(f"ROUGE-1: {scores['ROUGE-1']}")
    print(f"ROUGE-2: {scores['ROUGE-2']}")
    print(f"ROUGE-L: {scores['ROUGE-L']}")
    print("=" * 30)

    # 把分数保存到文件
    with open(output_metrics, 'w') as f:
        for k, v in scores.items():
            f.write(f"{k}: {v}\n")

    # 7. 打印几个 Case 给用户看
    print("\n>>> Sample Predictions:")
    for i in range(min(5, len(all_preds))):
        print(f"Pred: {all_preds[i]}")
        print(f"Ref:  {all_golds[i]}")
        print("-" * 20)

    print(f"\n>>> Full results saved to: {result_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--exp_name', type=str, default='a')
    parser.add_argument('-d', '--dataset', type=str, required=True, help="dataset used for testing")
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help="path to best_model.pt")
    parser.add_argument('--batch_size', type=int, default=16)  # 推理不占显存，可以大点
    parser.add_argument('--gnn_layers', type=int, default=3)

    args = parser.parse_args()
    evaluate_model(args)

# 示例：测试实验 A，使用 mix_all_kb 数据集
# python experiments/eval.py -a a -d mix_all_kb -c results/a_mix_all_kb_20251202_071154/best_model.pt