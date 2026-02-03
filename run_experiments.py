#!/usr/bin/env python3
"""
run_experiments.py — ローカル実験メインスクリプト

3フェーズを自動実行:
  Phase 1: 高速スクリーニング (7設定 x 3エポック, fold 0のみ)
  Phase 2: 中深度評価 (上位3設定 x 15エポック + early stopping)
  Phase 3: 最終学習 (最良1設定 x 全データ x 20エポック)

Usage:
    python run_experiments.py --data-dir ./data/deep-past-initiative-machine-translation
"""

import argparse
import csv
import os
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from tqdm import tqdm

from config import (
    EXPERIMENT_CONFIGS,
    KFOLD_SEED,
    PREPROCESS_FN,
    TRAINING_DEFAULTS,
    apply_dictionary_prefix,
    compute_metrics,
    get_kfold_splits,
    load_dictionary,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_source_length, max_target_length):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = self.sources[idx]
        target = self.targets[idx]
        source_enc = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str):
    """trainデータを読み込み (source, target) のリストを返す"""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # カラム名を確認して自動検出
    print(f"[INFO] Train columns: {list(df_train.columns)}")
    print(f"[INFO] Test columns: {list(df_test.columns)}")
    print(f"[INFO] Train samples: {len(df_train)}, Test samples: {len(df_test)}")

    # 実データ: train.csv = oare_id, transliteration, translation
    #           test.csv  = id, text_id, line_start, line_end, transliteration
    src_col = None
    tgt_col = None
    for c in df_train.columns:
        cl = c.lower()
        if cl in ("input", "source", "akkadian", "src", "transliteration"):
            src_col = c
        elif cl in ("output", "target", "translation", "english", "tgt"):
            tgt_col = c

    if src_col is None or tgt_col is None:
        # フォールバック: 2番目と3番目のカラム
        cols = list(df_train.columns)
        src_col = cols[1] if len(cols) > 1 else cols[0]
        tgt_col = cols[2] if len(cols) > 2 else cols[-1]
        print(f"[WARN] Auto-detected columns: src={src_col}, tgt={tgt_col}")
    else:
        print(f"[INFO] Using columns: src={src_col}, tgt={tgt_col}")

    sources = df_train[src_col].astype(str).tolist()
    targets = df_train[tgt_col].astype(str).tolist()

    # テストデータ
    test_src_col = None
    for c in df_test.columns:
        cl = c.lower()
        if cl in ("input", "source", "akkadian", "src", "transliteration"):
            test_src_col = c
    if test_src_col is None:
        test_src_col = list(df_test.columns)[1] if len(df_test.columns) > 1 else list(df_test.columns)[0]

    test_sources = df_test[test_src_col].astype(str).tolist()
    test_ids = df_test.iloc[:, 0].tolist()  # ID column

    return sources, targets, test_sources, test_ids, src_col, tgt_col


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config: dict, device: torch.device):
    """設定に基づきモデルとトークナイザーをロード"""
    model_name = config["model_name"]
    print(f"[INFO] Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # NLLBの場合、ソース/ターゲット言語を設定
    if "src_lang" in config:
        try:
            tokenizer.src_lang = config["src_lang"]
        except Exception:
            print(f"[WARN] Could not set src_lang={config['src_lang']}, using default")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # LoRA適用
    if config.get("use_lora", False):
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=["q", "v"],  # 一般的なattentionモジュール
        )
        try:
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except Exception as e:
            print(f"[WARN] LoRA failed with default targets, trying auto: {e}")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    model = model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, device, fp16=True):
    """1エポック分の学習"""
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler("cuda") if fp16 and device.type == "cuda" else None

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, tokenizer, device, max_target_length=128):
    """評価: 予測テキストを生成してメトリクスを計算"""
    model.eval()
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True,
            )

            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            # labelsの-100をpad_token_idに戻す
            labels_cleaned = labels.clone()
            labels_cleaned[labels_cleaned == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels_cleaned, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

    metrics = compute_metrics(all_preds, all_refs)
    return metrics, all_preds, all_refs


def generate_predictions(model, dataloader, tokenizer, device, max_target_length=128):
    """テストデータの予測を生成"""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True,
            )

            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_preds.extend(preds)

    return all_preds


# ---------------------------------------------------------------------------
# 実験1回分の実行
# ---------------------------------------------------------------------------

def run_single_experiment(
    exp_name: str,
    config: dict,
    train_sources: list[str],
    train_targets: list[str],
    val_sources: list[str],
    val_targets: list[str],
    device: torch.device,
    num_epochs: int = 3,
    early_stopping_patience: int = 0,
    dictionary: dict | None = None,
):
    """1つの実験設定を実行し、メトリクスを返す"""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Config: {config['description']}")
    print(f"Epochs: {num_epochs}, Train: {len(train_sources)}, Val: {len(val_sources)}")
    print(f"{'='*60}")

    start_time = time.time()

    # 前処理適用
    preprocess_fn = PREPROCESS_FN[config["preprocess"]]
    train_src = [preprocess_fn(s) for s in train_sources]
    train_tgt = [preprocess_fn(t) for t in train_targets]
    val_src = [preprocess_fn(s) for s in val_sources]
    val_tgt = [preprocess_fn(t) for t in val_targets]

    # 辞書拡張
    if config.get("use_dict") and dictionary:
        train_src = [apply_dictionary_prefix(s, dictionary) for s in train_src]
        val_src = [apply_dictionary_prefix(s, dictionary) for s in val_src]

    # タスクprefix
    prefix = config.get("task_prefix", "")
    if prefix:
        train_src = [prefix + s for s in train_src]
        val_src = [prefix + s for s in val_src]

    # モデル・トークナイザーロード
    model, tokenizer = load_model_and_tokenizer(config, device)

    # データセット作成
    train_dataset = TranslationDataset(
        train_src, train_tgt, tokenizer,
        config["max_source_length"], config["max_target_length"],
    )
    val_dataset = TranslationDataset(
        val_src, val_tgt, tokenizer,
        config["max_source_length"], config["max_target_length"],
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=TRAINING_DEFAULTS["weight_decay"],
    )
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * TRAINING_DEFAULTS["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 学習ループ
    best_geo_mean = 0
    patience_counter = 0
    best_metrics = None

    for epoch in range(num_epochs):
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            fp16=TRAINING_DEFAULTS["fp16"] and device.type == "cuda",
        )
        metrics, preds, refs = evaluate(
            model, val_loader, tokenizer, device, config["max_target_length"],
        )
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"BLEU: {metrics['bleu']:.2f} | chrF++: {metrics['chrf']:.2f} | "
              f"GeoMean: {metrics['geo_mean']:.2f}")

        if metrics["geo_mean"] > best_geo_mean:
            best_geo_mean = metrics["geo_mean"]
            best_metrics = metrics.copy()
            patience_counter = 0
            # サンプル予測を表示
            for i in range(min(3, len(preds))):
                print(f"    [Sample {i}] Pred: {preds[i][:80]}")
                print(f"    [Sample {i}] Ref:  {refs[i][:80]}")
        else:
            patience_counter += 1
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time

    # メモリ解放
    del model, optimizer, scheduler
    torch.cuda.empty_cache() if device.type == "cuda" else None

    result = {
        "experiment": exp_name,
        "model_name": config["model_name"],
        "preprocess": config["preprocess"],
        "use_dict": config.get("use_dict", False),
        "use_lora": config.get("use_lora", False),
        "lr": config["lr"],
        "num_epochs_run": epoch + 1,
        "bleu": best_metrics["bleu"] if best_metrics else 0,
        "chrf": best_metrics["chrf"] if best_metrics else 0,
        "geo_mean": best_metrics["geo_mean"] if best_metrics else 0,
        "elapsed_sec": round(elapsed, 1),
    }
    return result


# ---------------------------------------------------------------------------
# Phase 3: 最終学習（全データ使用）→ モデル保存
# ---------------------------------------------------------------------------

def run_final_training(
    exp_name: str,
    config: dict,
    all_sources: list[str],
    all_targets: list[str],
    test_sources: list[str],
    test_ids: list,
    device: torch.device,
    num_epochs: int = 20,
    output_dir: str = "results/final_model",
    dictionary: dict | None = None,
):
    """全データで最終学習し、テスト予測を出力"""
    print(f"\n{'='*60}")
    print(f"PHASE 3: Final Training — {exp_name}")
    print(f"All data: {len(all_sources)} samples, Epochs: {num_epochs}")
    print(f"{'='*60}")

    # 前処理
    preprocess_fn = PREPROCESS_FN[config["preprocess"]]
    train_src = [preprocess_fn(s) for s in all_sources]
    train_tgt = [preprocess_fn(t) for t in all_targets]
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

    model, tokenizer = load_model_and_tokenizer(config, device)

    train_dataset = TranslationDataset(
        train_src, train_tgt, tokenizer,
        config["max_source_length"], config["max_target_length"],
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # テストデータ用（ダミーtargetで作成し、予測のみ使用）
    test_dataset = TranslationDataset(
        test_src, [""] * len(test_src), tokenizer,
        config["max_source_length"], config["max_target_length"],
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"] * 0.5,  # Phase 3はLR半減
        weight_decay=TRAINING_DEFAULTS["weight_decay"],
    )
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * TRAINING_DEFAULTS["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(num_epochs):
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            fp16=TRAINING_DEFAULTS["fp16"] and device.type == "cuda",
        )
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    # テスト予測
    predictions = generate_predictions(
        model, test_loader, tokenizer, device, config["max_target_length"],
    )

    # モデル保存
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Model saved to {output_dir}")

    # Submission CSV
    submission = pd.DataFrame({"id": test_ids, "translation": predictions})
    sub_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"[INFO] Submission saved to {sub_path}")

    # サンプル表示
    print("\n--- Sample predictions ---")
    for i in range(min(10, len(predictions))):
        print(f"  [{test_ids[i]}] {predictions[i][:100]}")

    return predictions


# ---------------------------------------------------------------------------
# メインパイプライン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Akkadian-to-English experiment pipeline")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the competition data directory")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1/2/3). 0=all phases")
    parser.add_argument("--experiments", type=str, default="all",
                        help="Comma-separated experiment names for Phase 1 (default: all)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for models and logs")
    args = parser.parse_args()

    set_seed(TRAINING_DEFAULTS["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # データ読み込み
    sources, targets, test_sources, test_ids, src_col, tgt_col = load_data(args.data_dir)

    # 辞書読み込み
    dictionary = load_dictionary(args.data_dir)

    # KFold分割（fold 0 を使用）
    fold_splits = list(get_kfold_splits(len(sources)))
    train_idx, val_idx = fold_splits[0]
    train_src = [sources[i] for i in train_idx]
    train_tgt = [targets[i] for i in train_idx]
    val_src = [sources[i] for i in val_idx]
    val_tgt = [targets[i] for i in val_idx]

    log_path = os.path.join(args.output_dir, "experiment_log.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    # CSVも作業ディレクトリに置く
    root_log_path = "experiment_log.csv"

    all_results = []

    # ------------------------------------------------------------------
    # Phase 1: 高速スクリーニング (3 epochs, fold 0)
    # ------------------------------------------------------------------
    if args.phase in (0, 1):
        print("\n" + "=" * 70)
        print("PHASE 1: Fast Screening (3 epochs, fold 0)")
        print("=" * 70)

        if args.experiments == "all":
            exp_names = list(EXPERIMENT_CONFIGS.keys())
        else:
            exp_names = [e.strip() for e in args.experiments.split(",")]

        for exp_name in exp_names:
            if exp_name not in EXPERIMENT_CONFIGS:
                print(f"[WARN] Unknown experiment: {exp_name}, skipping")
                continue
            config = EXPERIMENT_CONFIGS[exp_name]
            try:
                result = run_single_experiment(
                    exp_name, config,
                    train_src, train_tgt, val_src, val_tgt,
                    device, num_epochs=3,
                    dictionary=dictionary if config.get("use_dict") else None,
                )
                result["phase"] = 1
                all_results.append(result)
            except Exception as e:
                print(f"[ERROR] Experiment {exp_name} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "experiment": exp_name, "model_name": config["model_name"],
                    "preprocess": config["preprocess"], "use_dict": config.get("use_dict", False),
                    "use_lora": config.get("use_lora", False), "lr": config["lr"],
                    "num_epochs_run": 0, "bleu": 0, "chrf": 0, "geo_mean": 0,
                    "elapsed_sec": 0, "phase": 1,
                })

            # 随時ログ保存
            _save_log(all_results, root_log_path)

        # Phase 1 結果サマリ
        print("\n--- Phase 1 Results ---")
        df_results = pd.DataFrame(all_results)
        print(df_results[["experiment", "bleu", "chrf", "geo_mean", "elapsed_sec"]]
              .sort_values("geo_mean", ascending=False).to_string(index=False))

    # ------------------------------------------------------------------
    # Phase 2: 中深度評価 (上位3, 15 epochs, early stopping)
    # ------------------------------------------------------------------
    if args.phase in (0, 2):
        print("\n" + "=" * 70)
        print("PHASE 2: Mid-depth Evaluation (15 epochs, top 3, early stopping)")
        print("=" * 70)

        # Phase 1の結果から上位3を選択
        if args.phase == 2 and os.path.exists(root_log_path):
            df_prev = pd.read_csv(root_log_path)
            df_phase1 = df_prev[df_prev["phase"] == 1]
        else:
            df_phase1 = pd.DataFrame([r for r in all_results if r.get("phase") == 1])

        # geo_mean > 0 の実験のみ上位3に含める
        df_valid = df_phase1[df_phase1["geo_mean"] > 0]
        top3 = df_valid.nlargest(3, "geo_mean")["experiment"].tolist()
        print(f"[INFO] Top 3 from Phase 1: {top3}")

        for exp_name in top3:
            config = EXPERIMENT_CONFIGS[exp_name].copy()
            config["lr"] = config["lr"] * 0.5  # LR半減

            try:
                result = run_single_experiment(
                    f"{exp_name}_p2", config,
                    train_src, train_tgt, val_src, val_tgt,
                    device, num_epochs=15,
                    early_stopping_patience=3,
                    dictionary=dictionary if config.get("use_dict") else None,
                )
                result["phase"] = 2
                all_results.append(result)
            except Exception as e:
                print(f"[ERROR] Experiment {exp_name}_p2 failed: {e}")
            _save_log(all_results, root_log_path)

        # Phase 2 結果サマリ
        print("\n--- Phase 2 Results ---")
        df_p2 = pd.DataFrame([r for r in all_results if r.get("phase") == 2])
        if len(df_p2) > 0:
            print(df_p2[["experiment", "bleu", "chrf", "geo_mean", "elapsed_sec"]]
                  .sort_values("geo_mean", ascending=False).to_string(index=False))

    # ------------------------------------------------------------------
    # Phase 3: 最終学習 (最良1設定, 全データ, 20 epochs)
    # ------------------------------------------------------------------
    if args.phase in (0, 3):
        print("\n" + "=" * 70)
        print("PHASE 3: Final Training (best config, all data, 20 epochs)")
        print("=" * 70)

        # 最良設定を特定
        if args.phase == 3 and os.path.exists(root_log_path):
            df_all = pd.read_csv(root_log_path)
        else:
            df_all = pd.DataFrame(all_results)

        best_row = df_all.nlargest(1, "geo_mean").iloc[0]
        best_exp = best_row["experiment"].replace("_p2", "")
        print(f"[INFO] Best experiment: {best_exp} (geo_mean={best_row['geo_mean']:.2f})")

        config = EXPERIMENT_CONFIGS[best_exp]
        final_output = os.path.join(args.output_dir, "final_model")

        run_final_training(
            best_exp, config,
            sources, targets,
            test_sources, test_ids,
            device, num_epochs=20,
            output_dir=final_output,
            dictionary=dictionary if config.get("use_dict") else None,
        )

    # 最終ログ保存
    _save_log(all_results, root_log_path)
    if log_path != root_log_path:
        _save_log(all_results, log_path)
    print(f"\n[DONE] Experiment log saved to {root_log_path}")


def _save_log(results: list[dict], path: str):
    """結果をCSVに保存"""
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
