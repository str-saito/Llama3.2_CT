# LlamaCT: Llama-3.2 ファインチューニングと推論

このプロジェクトは、[unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B) をベースに、ダウンストリームタスク向けにファインチューニングおよび推論を行うための実装例です。DeepSpeed などを利用してメモリ最適化も実施しています。

## 概要

- **モデル:** unsloth/Llama-3.2-1B をベースモデルとして使用
- **ファインチューニング:** JSONL 形式のデータセットを用いて微調整
- **推論:** 学習済みモデルをロードしてテキスト生成を実施
- **メモリ最適化:** 混合精度(fp16)や DeepSpeedを利用
- **ログ管理:** Weights & Biases (W&B) を利用

## ディレクトリ構成

```bash
.
├── config/
│   └── train_config.yaml       # 学習設定ファイル（モデル、データセット、トレーニングパラメータなど）
├── data/
│   └── cleaned_llama_extracted_pages.jsonl  # 学習用データセット（JSONL形式）
├── output/                     # 学習済みモデルやチェックポイントの保存先
├── src/
│   └── train.py                # 学習スクリプト
├── .gitignore                  # Git管理から除外するファイル・ディレクトリ
└── README.md                   # このファイル
```

## 環境構築

```
# 仮想環境の作成（Anaconda/Minicondaの場合）
mkdir data
mkdir output
mkdir wandb
conda create -n llma3.2 python=3.10 -y
conda activate llma3.2

# 必要なライブラリのインストール
pip install -r requirements.txt

sudo apt update
sudo apt install libopenmpi-dev
pip install mpi4py --no-cache-dir
```

## config 
train_config.yaml の例
```
model:
  model: "unsloth/Llama-3.2-1B"
  tokenizer: "unsloth/Llama-3.2-1B"
  use_cache: false
  max_length: 128

dataset:
  path: data/cleaned_llama_extracted_pages.jsonl
  split: "train"

train:
  output_dir: output
  run_name: "llama3.2_run"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 3e-5
  max_grad_norm: 1.0 
  evaluation_strategy: "steps"
  save_strategy: "steps"
  save_steps: 2000
  logging_steps: 100
  fp16: true
  deepspeed: "config/ds_config.json"
```
DeepSpeed 設定ファイル（ds_config.json）の例
VRAMのメモリが溢れるためメモリにオフロード
```
{
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "fp16": { "enabled": true },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": { "device": "cpu", "pin_memory": true }
    }
}

```

## 実行
### 継続学習の実行
```
python src/train.py --train_config config/train_config.yaml --wandb_user <your_wandb_user> --wandb_project llama3.2
```

### 推論の実行
学習が完了すると、output ディレクトリにチェックポイントが保存されます。推論の例は以下の通りです。
```
python src/inference.py
```

## 参考資料
```
- [https://note.com/izai/n/nd6783bf4e4c6](https://note.com/izai/n/nd6783bf4e4c6)
```




