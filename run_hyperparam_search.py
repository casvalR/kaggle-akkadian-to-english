#!/usr/bin/env python3
"""
run_hyperparam_search.py — Optunaベイズ最適化によるハイパーパラメータ探索

Usage:
    python run_hyperparam_search.py \
        --data-dir ./data/deep-past-initiative-machine-translation \
        --model nllb_600m_dict \
        --n-trials 20 \
        --epochs 5 \
        --output-dir results/hyperparam
"""

import argparse
import os

import optuna
import pandas as pd
import torch
from transformers import set_seed

from config import (
    EXPERIMENT_CONFIGS,
    HYPERPARAM_SEARCH_SPACE,
    TRAINING_DEFAULTS,
    load_dictionary,
)
from run_experiments import load_data, get_kfold_splits, run_single_experiment


def create_objective(
    model_name: str,
    train_src: list[str],
    train_tgt: list[str],
    val_src: list[str],
    val_tgt: list[str],
    device: torch.device,
    epochs: int,
    dictionary: dict | None,
):
    """Optuna objective関数を生成するクロージャ"""
    search_space = HYPERPARAM_SEARCH_SPACE.get(model_name, {})
    base_config = EXPERIMENT_CONFIGS[model_name].copy()

    def objective(trial: optuna.Trial) -> float:
        config = base_config.copy()

        # ハイパーパラメータのサンプリング
        for param_name, spec in search_space.items():
            if isinstance(spec, list):
                # カテゴリカル
                value = trial.suggest_categorical(param_name, spec)
            elif isinstance(spec, tuple) and len(spec) == 3 and spec[2] == "log":
                # 対数スケール float
                value = trial.suggest_float(param_name, spec[0], spec[1], log=True)
            elif isinstance(spec, tuple) and len(spec) == 2:
                # 線形スケール float
                value = trial.suggest_float(param_name, spec[0], spec[1])
            else:
                continue
            config[param_name] = value

        # TRAINING_DEFAULTS上書き用の値を保持
        weight_decay = config.pop("weight_decay", TRAINING_DEFAULTS["weight_decay"])
        warmup_ratio = config.pop("warmup_ratio", TRAINING_DEFAULTS["warmup_ratio"])

        # epoch_callback: Optunaにエポックごとのスコアを報告 + pruning判定
        def epoch_callback(epoch, metrics):
            trial.report(metrics["geo_mean"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # TRAINING_DEFAULTSを一時的に書き換え
        original_wd = TRAINING_DEFAULTS["weight_decay"]
        original_wr = TRAINING_DEFAULTS["warmup_ratio"]
        TRAINING_DEFAULTS["weight_decay"] = weight_decay
        TRAINING_DEFAULTS["warmup_ratio"] = warmup_ratio

        try:
            result = run_single_experiment(
                exp_name=f"{model_name}_trial{trial.number}",
                config=config,
                train_sources=train_src,
                train_targets=train_tgt,
                val_sources=val_src,
                val_targets=val_tgt,
                device=device,
                num_epochs=epochs,
                dictionary=dictionary if config.get("use_dict") else None,
                epoch_callback=epoch_callback,
            )
        finally:
            # TRAINING_DEFAULTSを復元
            TRAINING_DEFAULTS["weight_decay"] = original_wd
            TRAINING_DEFAULTS["warmup_ratio"] = original_wr

        return result["geo_mean"]

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the competition data directory")
    parser.add_argument("--model", type=str, required=True,
                        help="Model config name (e.g. nllb_600m_dict, byt5_base_v2)")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epochs per trial (short for screening)")
    parser.add_argument("--output-dir", type=str, default="results/hyperparam",
                        help="Directory for Optuna DB and results")
    parser.add_argument("--fold", type=int, default=0,
                        help="KFold index to use for evaluation")
    args = parser.parse_args()

    model_name = args.model
    if model_name not in EXPERIMENT_CONFIGS:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"Available: {list(EXPERIMENT_CONFIGS.keys())}")
        return

    if model_name not in HYPERPARAM_SEARCH_SPACE:
        print(f"[WARN] No search space defined for {model_name}, using defaults")
        print(f"Available search spaces: {list(HYPERPARAM_SEARCH_SPACE.keys())}")

    set_seed(TRAINING_DEFAULTS["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # データ読み込み
    sources, targets, _, _, _, _ = load_data(args.data_dir)
    dictionary = load_dictionary(args.data_dir)

    # KFold分割
    fold_splits = list(get_kfold_splits(len(sources)))
    train_idx, val_idx = fold_splits[args.fold]
    train_src = [sources[i] for i in train_idx]
    train_tgt = [targets[i] for i in train_idx]
    val_src = [sources[i] for i in val_idx]
    val_tgt = [targets[i] for i in val_idx]

    print(f"[INFO] Fold {args.fold}: Train={len(train_src)}, Val={len(val_src)}")

    # Optuna study
    os.makedirs(args.output_dir, exist_ok=True)
    db_path = os.path.join(args.output_dir, f"{model_name}.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction="maximize",
        study_name=f"akkadian_{model_name}",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        ),
    )

    objective = create_objective(
        model_name, train_src, train_tgt, val_src, val_tgt,
        device, args.epochs, dictionary,
    )

    print(f"\n[INFO] Starting Optuna search: {args.n_trials} trials, {args.epochs} epochs/trial")
    print(f"[INFO] DB: {db_path} (resume supported)")
    study.optimize(objective, n_trials=args.n_trials)

    # 結果出力
    print("\n" + "=" * 60)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best GeoMean: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # best_params.csvに追記
    best_csv = os.path.join(args.output_dir, "best_params.csv")
    best_row = {
        "model": model_name,
        "geo_mean": study.best_value,
        "n_trials": len(study.trials),
        **study.best_params,
    }
    if os.path.exists(best_csv):
        df = pd.read_csv(best_csv)
        # 同じモデルの行を更新
        df = df[df["model"] != model_name]
        df = pd.concat([df, pd.DataFrame([best_row])], ignore_index=True)
    else:
        df = pd.DataFrame([best_row])
    df.to_csv(best_csv, index=False)
    print(f"\n[INFO] Best params saved to {best_csv}")

    # 可視化 (matplotlibが利用可能な場合)
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_optimization_history(study)
        fig_path = os.path.join(args.output_dir, f"{model_name}_optimization.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Optimization history saved to {fig_path}")

        if len(study.trials) >= 5:
            fig = plot_param_importances(study)
            fig_path = os.path.join(args.output_dir, f"{model_name}_importance.png")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Param importance saved to {fig_path}")
    except Exception as e:
        print(f"[WARN] Could not generate plots: {e}")


if __name__ == "__main__":
    main()
