# 外部データセット統合機能

このプロジェクトは、Hugging Face Datasetsライブラリを使用して外部データセットでnanoGPTモデルを学習する機能を追加しています。

## 新しく追加されたファイル

1. **`dataset_loader.py`** - Hugging Face Datasetsのストリーミング機能を使用したデータローダー
2. **`config/train_external_dataset.py`** - 外部データセット学習用の設定ファイル
3. **`README_external_datasets.md`** - この説明ファイル

## 機能

- **ストリーミングデータ読み込み**: 大規模データセットを効率的に処理
- **自動トークナイゼーション**: GPT-2エンコーディングを使用
- **フォールバック機能**: 外部データセット取得失敗時のローカルデータセット使用
- **完全統合**: 既存のnanoGPTコードとの完全な互換性

## 使用方法

### 基本的な使用例

```bash
# WikiTextデータセットでの学習
python train.py config/train_external_dataset.py --dataset_name=wikitext

# OpenWebTextデータセットでの学習  
python train.py config/train_external_dataset.py --dataset_name=openwebtext

# C4データセットでの学習
python train.py config/train_external_dataset.py --dataset_name=c4 --dataset_config=en
```

### 詳細な設定例

```bash
# カスタム設定での学習
python train.py config/train_external_dataset.py \
    --use_external_dataset=True \
    --dataset_name=wikitext \
    --dataset_config=wikitext-103-raw-v1 \
    --batch_size=4 \
    --block_size=512 \
    --max_iters=50000
```

### 設定ファイルでの学習

カスタム設定ファイルを作成することも可能です：

```python
# my_config.py
use_external_dataset = True
dataset_name = "your_dataset_name"
dataset_config = "your_config"
text_column = "text"
batch_size = 8
block_size = 1024
max_iters = 100000
```

```bash
python train.py my_config.py
```

## サポートされるデータセット

以下のデータセットが事前設定されています：

- **wikitext**: Wikipedia テキストデータ
- **openwebtext**: オープンソース版WebText
- **c4**: Common Crawlクリーンアップ版
- **bookcorpus**: 書籍コーパス
- **pile**: 多様なテキストソースの集合

## 設定パラメータ

### 外部データセット設定

- `use_external_dataset`: 外部データセットを使用するか（True/False）
- `dataset_name`: Hugging Faceデータセット名
- `dataset_config`: データセット設定（オプション）
- `text_column`: テキストが含まれるカラム名（デフォルト: "text"）
- `streaming`: ストリーミングモードを使用するか（True/False）
- `cache_dir`: キャッシュディレクトリパス（オプション）
- `tokenizer_type`: トークナイザーの種類（現在は"gpt2"のみ）

### 学習設定（推奨値）

```python
# 外部データセット用の推奨設定
batch_size = 4              # メモリ効率化
gradient_accumulation_steps = 8  # 実効バッチサイズ32
block_size = 512           # 中程度のシーケンス長
learning_rate = 3e-4       # 外部データセット用
max_iters = 50000         # 長期学習
eval_interval = 1000      # 評価間隔
```

## メモリ使用量とパフォーマンス

### 推奨GPU設定

- **8GB VRAM**: batch_size=4, block_size=256
- **16GB VRAM**: batch_size=8, block_size=512  
- **24GB VRAM**: batch_size=16, block_size=1024

### ストリーミングの利点

- **メモリ効率**: 全データをメモリに読み込む必要がない
- **スケーラビリティ**: 数TB規模のデータセットも処理可能
- **リアルタイム**: データセットの更新に即座に対応

## トラブルシューティング

### よくある問題

1. **データセット読み込みエラー**
   ```
   解決: インターネット接続を確認し、Hugging Faceのデータセット名を正確に指定
   ```

2. **メモリ不足エラー**
   ```
   解決: batch_sizeやblock_sizeを削減
   ```

3. **トークナイゼーションエラー**
   ```
   解決: text_columnパラメータが正しいカラム名を指しているか確認
   ```

### フォールバック機能

外部データセットが利用できない場合、システムは自動的にローカルデータセット（shakespeare等）にフォールバックします。

## 例：WikiTextでの学習

```bash
# ステップ1: 設定ファイルで学習開始
python train.py config/train_external_dataset.py --dataset_name=wikitext

# ステップ2: 学習の進捗確認
# 出力例:
# データセット 'wikitext' を読み込んでいます...
# 外部データセット 'wikitext' の初期化が完了しました
# vocab_size = 50257
# ステップ 0: 学習損失 10.8234, 検証損失 10.8156
```

## 高度な使用例

### カスタムデータセットの追加

`dataset_loader.py`の`DATASET_CONFIGS`に新しい設定を追加：

```python
DATASET_CONFIGS["my_dataset"] = {
    "dataset_config": "my_config",
    "text_column": "content",
    "streaming": True
}
```

### マルチGPU学習

```bash
# 4GPU での分散学習
torchrun --standalone --nproc_per_node=4 train.py config/train_external_dataset.py --dataset_name=c4
```

この機能により、nanoGPTは様々な外部データセットで効率的に学習できるようになり、研究や実験の幅が大幅に広がります。
