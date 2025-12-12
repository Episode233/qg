import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 路径黑魔法 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入自定义模块
from utils.graph_dataset import ExpaDataset, ExpaCollator
from models.bart import get_bart_with_lora
from datasets import load_from_disk


def train(args):
    # ==========================================
    # 1. 初始化与配置
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using Device: {device}")

    # 创建结果目录 results/exp_name_timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.exp_name}_{args.dataset}_{timestamp}"
    save_dir = os.path.join(project_root, "results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f">>> Results will be saved to: {save_dir}")

    # ==========================================
    # 2. 准备数据
    # ==========================================
    print(">>> Loading Data...")
    processed_dir = os.path.join(project_root, "datasets", "processed")
    data_path = os.path.join(processed_dir, args.dataset)
    vocab_filename = f"{args.dataset}_relations.json"
    vocab_path = os.path.join(data_path, vocab_filename)

    if not os.path.exists(data_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Data not found at {data_path}. Run build_dataset.py first.")

    tokenizer, _ = get_bart_with_lora()
    hf_ds_dict = load_from_disk(data_path)

    train_dataset = ExpaDataset(hf_ds_dict['train'], vocab_path, tokenizer)
    val_dataset = ExpaDataset(hf_ds_dict['validation'], vocab_path, tokenizer)

    collator = ExpaCollator()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collator, num_workers=8, pin_memory=True)

    num_relations = len(train_dataset.rel2id)
    print(f">>> Data Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f">>> Num Relations: {num_relations}")

    # ==========================================
    # 3. 初始化模型
    # ==========================================
    print(">>> Initializing Model...")
    if args.exp_name == 'a':
        from experiments.exp_a import ExpaModel
        model = ExpaModel(num_relations=num_relations, gnn_layers=args.gnn_layers, dropout=args.dropout)
    elif args.exp_name == 'b':
        from experiments.exp_b import ExpBModel
        model = ExpBModel(num_relations=num_relations, gnn_layers=args.gnn_layers, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown experiment: {args.exp_name}")

    model.to(device)

    # ==========================================
    # 4. 优化器配置
    # ==========================================
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    print(f">>> Optimizer ready. Training {len(trainable_params)} tensors.")

    # ==========================================
    # 5. 训练循环
    # ==========================================
    best_val_loss = float('inf')
    patience_counter = 0  # 【新增】耐心计数器

    # 用于记录 Loss 的字典，方便最后画图
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        # 命令行进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            batch = batch.to(device)

            loss = model(batch)
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # 实时在命令行显示当前 Loss
                current_loss = loss.item() * args.grad_accum_steps
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})

            total_train_loss += loss.item() * args.grad_accum_steps

        avg_train_loss = total_train_loss / len(train_loader)

        # ==========================================
        # 6. 验证循环
        # ==========================================
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = model(batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # 命令行打印 Epoch 总结
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 记录数据
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # ==========================================
        # 7. 保存最佳模型
        # ==========================================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # 【重置】只要有进步，就重置计数器
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f">>> New Best Model Saved!")
        else:
            patience_counter += 1  # 【增加】没进步，计数器+1
            print(f">>> No improvement. Patience: {patience_counter}/{args.patience}")

        # 【新增】触发停止条件
        if patience_counter >= args.patience:
            print(f"\n>>> Early Stopping triggered! No improvement for {args.patience} epochs.")
            break  # 跳出 epoch 循环

    print(">>> Training Finished.")

    # ==========================================
    # 8. 训练结束：使用 Seaborn 画图并保存
    # ==========================================
    print(">>> Generating Loss Plots...")

    # 转换数据格式
    df = pd.DataFrame(history)

    # 保存 CSV 数据备份
    csv_path = os.path.join(save_dir, "training_log.csv")
    df.to_csv(csv_path, index=False)

    # 设置 Seaborn 风格
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))

    # 绘制 Train 和 Val Loss
    sns.lineplot(data=df, x='epoch', y='train_loss', label='Train Loss', marker='o')
    sns.lineplot(data=df, x='epoch', y='val_loss', label='Val Loss', marker='o')

    plt.title(f"Training Curve: {args.dataset} (Exp {args.exp_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 保存图片
    plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()  # 关闭画布，释放内存

    print(f">>> Plot saved to: {plot_path}")
    print(f">>> Log saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EXPA Model")
    parser.add_argument('-e', '--exp_name', type=str, default='a', help="Experiment Name")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--epochs', type=int, default=233, help="Max epochs (will stop early)")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")

    args = parser.parse_args()

    train(args)

# 示例：训练实验 A，使用 PQ_mix 数据集
# python experiments/train.py -e a -d PQ_mix --batch_size 32 --patience 10 --grad_accum_steps 1 --lr 5e-4
# python experiments/train.py -e a -d PQL_mix --batch_size 32 --patience 10 --grad_accum_steps 1 --lr 5e-4
# python experiments/train.py -e a -d WC2014_mix --batch_size 128 --patience 5 --grad_accum_steps 1 --lr 5e-4
# python experiments/train.py -e a -d FB15k-237_mix --batch_size 128 --patience 5 --grad_accum_steps 1 --lr 5e-4
# python experiments/train.py -e a -d YAGO3-10_mix --batch_size 128 --patience 5 --grad_accum_steps 1 --lr 5e-4
# python experiments/train.py -e a -d WN18RR_mix --batch_size 128 --patience 5 --grad_accum_steps 1 --lr 5e-4