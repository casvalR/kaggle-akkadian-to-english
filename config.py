"""
config.py — 実験設定・前処理関数・評価関数を集約
"""

import re
import math
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# 前処理関数
# ---------------------------------------------------------------------------

def preprocess_basic(text: str) -> str:
    """基本的な正規化: 余分な空白除去、小文字化"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_moderate(text: str) -> str:
    """中程度の正規化: basic + 数字・特殊記号の統一"""
    text = preprocess_basic(text)
    # 角括弧内の復元マーカーを除去（例: [x], [...] など）
    text = re.sub(r"\[\.{3,}\]", " ", text)
    text = re.sub(r"\[x\]", " ", text, flags=re.IGNORECASE)
    # 連続するハイフンを単一に
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_aggressive(text: str) -> str:
    """積極的な正規化: moderate + 句読点除去、記号簡略化"""
    text = preprocess_moderate(text)
    # 角括弧自体を除去
    text = re.sub(r"[\[\]]", "", text)
    # 丸括弧とその中身を除去
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


PREPROCESS_FN = {
    "basic": preprocess_basic,
    "moderate": preprocess_moderate,
    "aggressive": preprocess_aggressive,
}


# ---------------------------------------------------------------------------
# 辞書拡張（eBL_Dictionary活用）
# ---------------------------------------------------------------------------

def load_dictionary(data_dir: str) -> dict[str, str]:
    """eBL辞書CSVを読み込み、単語 → 定義 の辞書を返す"""
    import os
    # CSVとTSV両方を探す
    for fname, sep in [("eBL_Dictionary.csv", ","), ("eBL_Dictionary.tsv", "\t")]:
        dict_path = os.path.join(data_dir, fname)
        if os.path.exists(dict_path):
            df = pd.read_csv(dict_path, sep=sep, header=0, on_bad_lines="skip")
            break
    else:
        print(f"[WARN] Dictionary not found in {data_dir}")
        return {}
    print(f"[INFO] Dictionary columns: {list(df.columns)}")
    mapping = {}
    # 実データ: word, definition, derived_from
    for col_src, col_tgt in [("word", "definition"), ("lemma", "meaning"), ("cf", "gw")]:
        if col_src in df.columns and col_tgt in df.columns:
            for _, row in df.iterrows():
                src = str(row[col_src]).strip()
                tgt = str(row[col_tgt]).strip()
                if src and tgt and src != "nan" and tgt != "nan":
                    mapping[src] = tgt
            break
    print(f"[INFO] Loaded dictionary with {len(mapping)} entries")
    return mapping


def apply_dictionary_prefix(text: str, dictionary: dict[str, str], max_hints: int = 5) -> str:
    """入力テキストの単語を辞書で引き、prefixとしてヒントを付加する"""
    if not dictionary:
        return text
    words = text.split()
    hints = []
    for w in words:
        if w in dictionary and len(hints) < max_hints:
            hints.append(f"{w}={dictionary[w]}")
    if hints:
        prefix = "Dictionary: " + "; ".join(hints) + " | "
        return prefix + text
    return text


# ---------------------------------------------------------------------------
# 評価関数
# ---------------------------------------------------------------------------

_bleu = BLEU()
_chrf = CHRF(word_order=2)  # chrF++


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """BLEU, chrF++, 幾何平均（コンペ評価基準）を計算"""
    bleu_score = _bleu.corpus_score(predictions, [references]).score
    chrf_score = _chrf.corpus_score(predictions, [references]).score
    # 幾何平均（0除算対策）
    if bleu_score > 0 and chrf_score > 0:
        geo_mean = math.sqrt(bleu_score * chrf_score)
    else:
        geo_mean = 0.0
    return {
        "bleu": round(bleu_score, 4),
        "chrf": round(chrf_score, 4),
        "geo_mean": round(geo_mean, 4),
    }


# ---------------------------------------------------------------------------
# データ分割
# ---------------------------------------------------------------------------

KFOLD_SPLITS = 5
KFOLD_SEED = 42


def get_kfold_splits(n_samples: int):
    """5-fold KFoldのインデックスを返すジェネレータ"""
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=KFOLD_SEED)
    return kf.split(range(n_samples))


# ---------------------------------------------------------------------------
# 実験設定レジストリ（Phase 1: 7 設定）
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIGS = {
    "marian_mul_en": {
        "model_name": "Helsinki-NLP/opus-mt-mul-en",
        "model_type": "seq2seq",
        "preprocess": "basic",
        "use_dict": False,
        "lr": 5e-5,
        "batch_size": 16,
        "use_lora": False,
        "max_source_length": 128,
        "max_target_length": 128,
        "description": "MarianMT mul-en baseline (~300M)",
    },
    "nllb_600m": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "model_type": "seq2seq",
        "preprocess": "basic",
        "use_dict": False,
        "lr": 3e-5,
        "batch_size": 8,
        "use_lora": False,
        "max_source_length": 128,
        "max_target_length": 128,
        "src_lang": "akk_Xsux",  # Akkadian
        "tgt_lang": "eng_Latn",
        "description": "NLLB-200-distilled-600M, low-resource specialist",
    },
    "nllb_600m_dict": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "model_type": "seq2seq",
        "preprocess": "moderate",
        "use_dict": True,
        "lr": 3e-5,
        "batch_size": 8,
        "use_lora": False,
        "max_source_length": 196,
        "max_target_length": 128,
        "src_lang": "akk_Xsux",
        "tgt_lang": "eng_Latn",
        "description": "NLLB-200-600M + dictionary prefix",
    },
    "t5_small": {
        "model_name": "google-t5/t5-small",
        "model_type": "seq2seq",
        "preprocess": "basic",
        "use_dict": False,
        "lr": 3e-4,
        "batch_size": 32,
        "use_lora": False,
        "max_source_length": 128,
        "max_target_length": 128,
        "task_prefix": "translate Akkadian to English: ",
        "description": "T5-small fast screening (60M)",
    },
    "byt5_small": {
        "model_name": "google/byt5-small",
        "model_type": "seq2seq",
        "preprocess": "basic",
        "use_dict": False,
        "lr": 1e-4,
        "batch_size": 16,
        "use_lora": False,
        "max_source_length": 256,
        "max_target_length": 256,
        "task_prefix": "translate Akkadian to English: ",
        "description": "ByT5-small character-level (300M)",
    },
    "byt5_base": {
        "model_name": "google/byt5-base",
        "model_type": "seq2seq",
        "preprocess": "moderate",
        "use_dict": False,
        "lr": 5e-5,
        "batch_size": 8,
        "use_lora": False,
        "max_source_length": 256,
        "max_target_length": 256,
        "task_prefix": "translate Akkadian to English: ",
        "description": "ByT5-base character-level (580M), most used in competition",
    },
    "madlad_3b": {
        "model_name": "google/madlad400-3b-mt",
        "model_type": "seq2seq",
        "preprocess": "basic",
        "use_dict": False,
        "lr": 1e-4,
        "batch_size": 4,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_source_length": 128,
        "max_target_length": 128,
        "task_prefix": "<2en> ",
        "description": "MADLAD-400-3B with LoRA, 450 languages",
    },
    # --- 新規モデル (Phase 2+) ---
    "byt5_base_v2": {
        "model_name": "google/byt5-base",
        "model_type": "seq2seq",
        "preprocess": "basic",
        "use_dict": False,
        "lr": 5e-4,
        "batch_size": 4,
        "use_lora": False,
        "max_source_length": 512,
        "max_target_length": 256,
        "task_prefix": "translate Akkadian to English: ",
        "description": "ByT5-base v2 (高LR, 長シーケンス, Discussion推奨設定)",
    },
    "mbart_50": {
        "model_name": "facebook/mbart-large-50-many-to-one-mmt",
        "model_type": "seq2seq",
        "preprocess": "basic",
        "use_dict": False,
        "src_lang": "ar_AR",
        "tgt_lang": "en_XX",
        "lr": 3e-5,
        "batch_size": 4,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_source_length": 128,
        "max_target_length": 128,
        "description": "mBART-50 many-to-one with LoRA (ar_AR→en_XX proxy)",
    },
    "nllb_1_3b_lora": {
        "model_name": "facebook/nllb-200-distilled-1.3B",
        "model_type": "seq2seq",
        "preprocess": "moderate",
        "use_dict": True,
        "src_lang": "akk_Xsux",
        "tgt_lang": "eng_Latn",
        "lr": 2e-5,
        "batch_size": 4,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_source_length": 196,
        "max_target_length": 128,
        "description": "NLLB-1.3B distilled with LoRA + dictionary",
    },
}

# ---------------------------------------------------------------------------
# 学習ハイパーパラメータのデフォルト値
# ---------------------------------------------------------------------------

TRAINING_DEFAULTS = {
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "fp16": True,
    "gradient_accumulation_steps": 1,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "logging_steps": 50,
    "seed": 42,
}

# ---------------------------------------------------------------------------
# Optuna ハイパーパラメータ探索範囲
# ---------------------------------------------------------------------------

HYPERPARAM_SEARCH_SPACE = {
    "nllb_600m_dict": {
        "lr": (1e-5, 1e-4, "log"),
        "batch_size": [4, 8, 16],
        "max_source_length": [128, 196, 256],
        "warmup_ratio": (0.05, 0.2),
        "weight_decay": (0.001, 0.1, "log"),
    },
    "byt5_base_v2": {
        "lr": (1e-4, 1e-3, "log"),
        "batch_size": [4, 8],
        "max_source_length": [256, 512],
        "warmup_ratio": (0.05, 0.2),
        "weight_decay": (0.001, 0.1, "log"),
    },
    "mbart_50": {
        "lr": (1e-5, 5e-5, "log"),
        "batch_size": [4, 8],
        "warmup_ratio": (0.05, 0.2),
        "weight_decay": (0.001, 0.1, "log"),
    },
    "nllb_1_3b_lora": {
        "lr": (5e-6, 5e-5, "log"),
        "lora_r": [8, 16, 32],
        "batch_size": [2, 4, 8],
        "warmup_ratio": (0.05, 0.2),
        "weight_decay": (0.001, 0.1, "log"),
    },
}
