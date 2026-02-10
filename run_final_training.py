#!/usr/bin/env python3
"""
run_final_training.py — 最良ハイパラでの最終学習 + fine-tuned重み保存

全trainデータで学習 (KFoldなし)、checkpoint保存、resume対応。

Usage:
    python run_final_training.py \
        --data-dir ./data/deep-past-initiative-machine-translation \
        --model nllb_600m_dict \
        --lr 1.5e-5 --batch-size 8 \
        --epochs 50 \
        --output-dir results/final_model

    # 途中再開
    python run_final_training.py \
        --data-dir ./data/deep-past-initiative-machine-translation \
        --model nllb_600m_dict \
        --epochs 50 \
        --output-dir results/final_model \
        --resume-from results/final_model/checkpoint-epoch-25
"""

import argparse
import json
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from config import (
    EXPERIMENT_CONFIGS,
    PREPROCESS_FN,
    TRAINING_DEFAULTS,
    apply_dictionary_prefix,
    load_dictionary,
)
from run_experiments import (
    TranslationDataset,
    generate_predictions,
    load_data,
    load_model_and_tokenizer,
    train_epoch,
)


def main():
    parser = argparse.ArgumentParser(description="Final training with best hyperparameters")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the competition data directory")
    parser.add_argument("--model", type=str, required=True,
                        help="Model config name (e.g. nllb_600m_dict)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Total number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config default)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides config default)")
    parser.add_argument("--max-source-length", type=int, default=None,
                        help="Max source length (overrides config default)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Weight decay (overrides default)")
    parser.add_argument("--warmup-ratio", type=float, default=None,
                        help="Warmup ratio (overrides default)")
    parser.add_argument("--lora-r", type=int, default=None,
                        help="LoRA rank (overrides config default)")
    parser.add_argument("--output-dir", type=str, default="results/final_model",
                        help="Output directory for model and predictions")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint directory")
    args = parser.parse_args()

    model_name = args.model
    if model_name not in EXPERIMENT_CONFIGS:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"Available: {list(EXPERIMENT_CONFIGS.keys())}")
        return

    set_seed(TRAINING_DEFAULTS["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Config構築 (CLIオーバーライドを適用)
    config = EXPERIMENT_CONFIGS[model_name].copy()
    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.max_source_length is not None:
        config["max_source_length"] = args.max_source_length
    if args.lora_r is not None:
        config["lora_r"] = args.lora_r

    weight_decay = args.weight_decay or TRAINING_DEFAULTS["weight_decay"]
    warmup_ratio = args.warmup_ratio or TRAINING_DEFAULTS["warmup_ratio"]

    # データ読み込み
    sources, targets, test_sources, test_ids, _, _ = load_data(args.data_dir)
    dictionary = load_dictionary(args.data_dir)

    # 前処理
    preprocess_fn = PREPROCESS_FN[config["preprocess"]]
    train_src = [preprocess_fn(s) for s in sources]
    train_tgt = [preprocess_fn(t) for t in targets]
    test_src = [preprocess_fn(s) for s in test_sources]

    # 辞書拡張
    if config.get("use_dict") and dictionary:
        train_src = [apply_dictionary_prefix(s, dictionary) for s in train_src]
        test_src = [apply_dictionary_prefix(s, dictionary) for s in test_src]

    # タスクprefix
    prefix = config.get("task_prefix", "")
    if prefix:
        train_src = [prefix + s for s in train_src]
        test_src = [prefix + s for s in test_src]

    print(f"\n{'='*60}")
    print(f"FINAL TRAINING: {model_name}")
    print(f"  Model: {config['model_name']}")
    print(f"  LR: {config['lr']}, Batch: {config['batch_size']}")
    print(f"  Epochs: {args.epochs}, Train samples: {len(train_src)}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    # モデルロード (resume or fresh)
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[INFO] Resuming from checkpoint: {args.resume_from}")
        # checkpoint configを読み込み
        ckpt_config = config.copy()
        ckpt_config["model_name"] = args.resume_from
        model, tokenizer = load_model_and_tokenizer(ckpt_config, device)
        # エポック番号を復元
        meta_path = os.path.join(args.resume_from, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            start_epoch = meta.get("epoch", 0)
            print(f"[INFO] Resuming from epoch {start_epoch}")
    else:
        model, tokenizer = load_model_and_tokenizer(config, device)

    # データセット
    train_dataset = TranslationDataset(
        train_src, train_tgt, tokenizer,
        config["max_source_length"], config["max_target_length"],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
    )

    test_dataset = TranslationDataset(
        test_src, [""] * len(test_src), tokenizer,
        config["max_source_length"], config["max_target_length"],
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False,
    )

    # Optimizer & Scheduler
    remaining_epochs = args.epochs - start_epoch
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=weight_decay,
    )
    total_steps = len(train_loader) * remaining_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)

    # 学習設定を保存
    training_config = {
        "model_name": model_name,
        "config": config,
        "epochs": args.epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2, default=str)

    # 学習ループ
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            fp16=TRAINING_DEFAULTS["fp16"] and device.type == "cuda",
        )
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"Elapsed: {elapsed/60:.1f}min")

        # Checkpoint保存
        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            with open(os.path.join(ckpt_dir, "training_meta.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "loss": avg_loss}, f)
            print(f"  [CHECKPOINT] Saved to {ckpt_dir}")

    total_time = time.time() - start_time
    print(f"\n[INFO] Training complete in {total_time/60:.1f} minutes")

    # 最終モデル保存
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump({"epoch": args.epochs, "total_time_sec": total_time}, f)
    print(f"[INFO] Final model saved to {args.output_dir}")

    # テスト予測
    print("\n[INFO] Generating test predictions...")
    predictions = generate_predictions(
        model, test_loader, tokenizer, device, config["max_target_length"],
    )

    submission = pd.DataFrame({"id": test_ids, "translation": predictions})
    sub_path = os.path.join(args.output_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"[INFO] Submission saved to {sub_path} ({len(submission)} rows)")

    # サンプル表示
    print("\n--- Sample predictions ---")
    for i in range(min(10, len(predictions))):
        print(f"  [{test_ids[i]}] {predictions[i][:100]}")


if __name__ == "__main__":
    main()
