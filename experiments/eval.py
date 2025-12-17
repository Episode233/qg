import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate  # HuggingFace 的评估库
import sacrebleu

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
metrics_dir = os.path.join(project_root, 'metrics')
sys.path.append(project_root)

from utils.graph_dataset import ExpaDataset, ExpaCollator
from models.bart import get_bart_with_lora
from datasets import load_from_disk
from utils.llm import evaluate_question

def compute_metrics(predictions, references):
    """
    计算 SacreBLEU, ROUGE, METEOR, BERTScore 分数
    """
    print(">>> Computing Metrics...")

    # 加载指标
    rouge = evaluate.load(path=os.path.join(metrics_dir, 'rouge'))
    meteor = evaluate.load(path=os.path.join(metrics_dir, 'meteor'))
    bertscore = evaluate.load(path=os.path.join(metrics_dir, "bertscore"))

    results = {}

    # --- SacreBLEU ---
    bleu_res = sacrebleu.corpus_bleu(predictions, [references])
    results['BLEU'] = round(bleu_res.score, 2)

    # --- ROUGE ---
    rouge_res = rouge.compute(predictions=predictions, references=references)
    results['ROUGE-1'] = round(rouge_res['rouge1'] * 100, 2)
    results['ROUGE-2'] = round(rouge_res['rouge2'] * 100, 2)
    results['ROUGE-L'] = round(rouge_res['rougeL'] * 100, 2)

    # --- METEOR ---
    meteor_res = meteor.compute(predictions=predictions, references=references)
    results['METEOR'] = round(meteor_res['meteor'] * 100, 2)

    # --- BERTScore ---
    # lang='en' 指定英语，rescale_with_baseline=True 可选
    bert_res = bertscore.compute(predictions=predictions, references=references, lang="en")
    # BERTScore 返回的是 list，需要求 mean
    results['BERTScore'] = round(np.mean(bert_res['f1']) * 100, 2)

    return results


def compute_llm_score(hf_test_dataset, all_preds, all_golds, limit=-1):
    """
    对整个测试集运行 LLM 评分
    注意: 这里会遍历原始 dataset 来获取 Graph Context
    limit: 限制评估的样本数，-1 表示不限制 (跑全量)
    """
    print(f"\n>>> Starting LLM Evaluation (Limit: {limit if limit > 0 else 'All'})...")

    scores_list = [-1] * len(hf_test_dataset)
    valid_scores = []

    # 遍历所有样本 (这里假设 all_preds 和 hf_test_dataset 是一一对应的，顺序没变)
    for i, item in enumerate(tqdm(hf_test_dataset, desc="LLM Judging")):
        if limit > 0 and i >= limit:
            break  # 达到限制，停止调用 LLM，后面的保持 -1

        # 1. 还原三元组上下文 (Graph Context)
        nodes = item['nodes']
        src_indices = item['edge_index'][0]
        tgt_indices = item['edge_index'][1]
        relations = item['edge_attr']

        triples_text = []
        for src, tgt, rel in zip(src_indices, tgt_indices, relations):
            # 去掉 Special Tokens
            s_text = nodes[src].replace("[TOPIC] ", "").replace("[ANS] ", "")
            t_text = nodes[tgt].replace("[TOPIC] ", "").replace("[ANS] ", "")
            triples_text.append(f"({s_text}, {rel}, {t_text})")

        context_str = "\n".join(triples_text)

        # 2. 获取其他信息
        start_node_idx = item['label_ids'][0]
        end_node_idx = item['label_ids'][1]
        start_node = nodes[start_node_idx].replace("[TOPIC] ", "")
        end_node = nodes[end_node_idx].replace("[ANS] ", "")

        ref_q = all_golds[i]
        gen_q = all_preds[i]

        # 3. 调用 LLM 打分
        score = evaluate_question(context_str, start_node, end_node, ref_q, gen_q)

        # 更新分数列表
        scores_list[i] = score
        valid_scores.append(score)

    # 计算平均分 (只计算实际评估过的样本)
    if len(valid_scores) > 0:
        avg_score = round(sum(valid_scores) / len(valid_scores), 2)
    else:
        avg_score = 0

    print(f">>> Evaluated {len(valid_scores)} samples. Avg Score: {avg_score}")
    return avg_score, scores_list


def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Evaluating using: {device}")

    # 1. 路径准备
    mixed_dir = os.path.join(project_root, "datasets", "mixed")
    data_path = os.path.join(mixed_dir, args.dataset)
    vocab_filename = f"{args.dataset}_relations.json"
    vocab_path = os.path.join(data_path, vocab_filename)

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
                             shuffle=False, collate_fn=collator, num_workers=8, pin_memory=True)

    num_relations = len(test_dataset.rel2id)

    # 3. 加载模型
    print(f">>> Loading Model from {args.checkpoint}...")

    if args.exp_name == 'a':
        from experiments.exp_a import ExpaModel
        model = ExpaModel(num_relations=num_relations, gnn_layers=args.gnn_layers)
    elif args.exp_name == 'b':
        from experiments.exp_b import ExpBModel
        model = ExpBModel(num_relations=num_relations, gnn_layers=args.gnn_layers)
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
            gen_ids = model.generate(batch, num_beams=args.num_beams)

            # 解码 Prediction
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # 解码 Reference (Target)
            tgt_ids = batch.y.clone()
            tgt_ids[tgt_ids == -100] = tokenizer.pad_token_id
            golds = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)

            all_preds.extend(preds)
            all_golds.extend(golds)

    # 我们需要传入原始的 hf_ds_dict['test']，因为它包含了原始的 edge_index 和 nodes
    llm_avg_score, llm_scores_list = compute_llm_score(hf_ds_dict['test'], all_preds, all_golds, limit=args.llm_limit)

    # 5. 保存生成的 Case
    df = pd.DataFrame({
        'Generated': all_preds,
        'Reference': all_golds,
        'LLM_Score': llm_scores_list  # 【新增】每条的分数
    })
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 6. 计算并打印分数
    scores = compute_metrics(all_preds, all_golds)

    # 【新增】加入 LLM 分数
    scores['LLM-Judge'] = llm_avg_score

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
    print(f"METEOR:  {scores['METEOR']}")
    print(f"BERTScore: {scores['BERTScore']}")
    print(f"LLM-Judge: {scores['LLM-Judge']}")
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
    parser.add_argument('-e', '--exp_name', type=str, default='a')
    parser.add_argument('-d', '--dataset', type=str, required=True, help="dataset used for testing")
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help="path to best_model.pt")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--num_beams', type=int, default=4, help="Beam size for generation")
    parser.add_argument('--llm_limit', type=int, default=100, help="Max samples for LLM eval (-1 for all)")

    args = parser.parse_args()
    evaluate_model(args)

# 示例：测试实验 A，使用 PQ_mix 数据集
# python experiments/eval.py -e a -d PQ_mix -c results/a_PQ_mix_20251212_090839/best_model.pt