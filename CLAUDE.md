# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## コマンド

### 依存関係のインストール
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### 基本的なトレーニングコマンド
```bash
# シェイクスピアデータでの文字レベルGPTトレーニング（初心者向け）
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py

# GPT-2の再現（シングルGPU）
python data/openwebtext/prepare.py
python train.py config/train_gpt2.py

# 分散データ並列での大規模トレーニング（複数GPU）
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# 日本語データでの事前学習
python train.py config/pretrain_japanese.py

# インストラクションチューニング
python finetune_instruction.py config/train_dolly_ja.py
```

### ファインチューニング
```bash
# GPT-2からのファインチューニング
python train.py config/finetune_shakespeare.py

# カスタムデータでのファインチューニング
python train.py --init_from=gpt2 --dataset=your_dataset
```

### サンプリングと推論
```bash
# 学習済みモデルからのテキスト生成
python sample.py --out_dir=out-shakespeare-char

# GPT-2モデルからの生成
python sample.py --init_from=gpt2-xl --start="What is the answer to life?"

# 対話モード
python chat.py
```

### 評価とベースライン
```bash
# GPT-2ベースラインの評価
python train.py config/eval_gpt2.py
python train.py config/eval_gpt2_medium.py
python train.py config/eval_gpt2_large.py
python train.py config/eval_gpt2_xl.py
```

### ベンチマーク
```bash
# モデルベンチマーク
python bench.py
```

### CPUでの実行
```bash
# CPU環境での実行（MacBookなど）
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000
```

### Apple Silicon（MPS）での実行
```bash
# Apple Silicon MacBook（MPS）での実行
python train.py config/train_shakespeare_char.py --device=mps
```

## アーキテクチャ

### 主要コンポーネント

1. **model.py**: GPTモデルの完全な実装
   - GPTConfig: モデル設定クラス
   - CausalSelfAttention: 因果的自己注意機構
   - MLP: 多層パーセプトロン
   - Block: Transformerブロック
   - GPT: メインモデルクラス

2. **train.py**: 事前学習用のメインスクリプト
   - 分散データ並列（DDP）サポート
   - 日本語データセットでの事前学習に対応
   - W&Bログ統合

3. **finetune_instruction.py**: インストラクションチューニング専用スクリプト
   - 事前学習済みモデルからのファインチューニング
   - Dolly JAなどの指示応答データセット対応

4. **dataset_loader.py**: 外部データセット読み込み
   - Hugging Face Datasetsライブラリ統合
   - ストリーミングモード対応
   - 複数データセットの混合学習サポート

5. **sample.py**: テキスト生成スクリプト
   - Temperature、top-kサンプリング
   - 学習済みモデルまたはGPT-2からの生成

6. **chat.py**: 対話モード
   - リアルタイム対話機能
   - システムプロンプト設定可能

### データフロー

1. **事前学習パイプライン**:
   データ準備 → 事前学習(train.py) → チェックポイント保存(out_pretrain/)

2. **インストラクションチューニングパイプライン**:
   事前学習済みモデル → ファインチューニング(finetune_instruction.py) → 最終モデル(out_dolly_ja/)

3. **推論パイプライン**:
   学習済みモデル → サンプリング(sample.py/chat.py) → テキスト生成

### 設定ファイル構造

config/ディレクトリには各用途別の設定ファイル:
- train_shakespeare_char.py: 文字レベル学習設定
- train_gpt2.py: GPT-2再現設定
- pretrain_japanese.py: 日本語事前学習設定
- train_dolly_ja.py: 日本語インストラクションチューニング設定
- eval_*.py: 各種評価設定

### モデルサイズとハードウェア要件

- **文字レベルGPT（Shakespeare）**: 1GPU、3分程度
- **GPT-2 124M再現**: 8×A100 40GB、約4日
- **日本語事前学習**: 大規模データセットに応じて調整
- **ファインチューニング**: 比較的小さなリソースで実行可能

### 出力ディレクトリ

- out/: デフォルトの出力ディレクトリ
- out_pretrain/: 日本語事前学習モデル
- out_dolly_ja/: インストラクションチューニング済みモデル
- out-shakespeare-char/: シェイクスピアモデル

### エラー対処

- Windows環境: `--compile=False`を追加
- メモリ不足: `--batch_size`を減らすかモデルサイズを小さく
- PyTorch 2.0未満: `--compile=False`を追加してFlash Attention無効化