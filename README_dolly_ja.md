# 日本語インストラクションチューニング方法

## 概要

このドキュメントでは、nanoGPTを使用して日本語の高品質なインストラクションモデルを作成する2段階の学習プロセスについて説明します。

### 2段階学習プロセス

1. **事前学習（train.py）**: 大規模な日本語テキストデータセットでGPTモデルを事前学習
2. **インストラクションチューニング（finetune_instruction.py）**: 事前学習済みモデルを指示応答データセットでファインチューニング

## 第1段階: 事前学習

まず、大規模なテキストデータセットでモデルを事前学習します：

```bash
# 日本語事前学習の実行
python train.py config/pretrain_japanese.py
```

事前学習済みモデルは `out_pretrain/` ディレクトリに保存されます。

## 第2段階: インストラクションチューニング

事前学習完了後、`kunishou/databricks-dolly-15k-ja`データセットでインストラクションチューニングを行います：

### データセット構造

`kunishou/databricks-dolly-15k-ja`データセットの構造：

- **instruction**: 指示・質問
- **input**: 入力文脈（空の場合もある）
- **output**: 期待される出力・回答
- **category**: カテゴリ（closed_qa、open_qa、brainstorming等）
- **index**: インデックス

## インストラクションチューニングの実行方法

### 基本的な実行

```bash
# インストラクションチューニングの実行
python finetune_instruction.py config/train_dolly_ja.py
```

### カスタマイズされた実行

```bash
# バッチサイズやイテレーション数を調整
python finetune_instruction.py config/train_dolly_ja.py \
    --batch_size=4 \
    --max_iters=3000 \
    --learning_rate=5e-6
```

### 短時間テスト

```bash
# 10イテレーションのテスト実行
python finetune_instruction.py config/train_dolly_ja.py \
    --max_iters=10 \
    --eval_interval=5 \
    --wandb_log=False
```

### 効率的なインストラクションチューニング（推奨）

```bash
# 2000イテレーションでインストラクションチューニング（約30分）
python finetune_instruction.py config/train_dolly_ja.py \
    --max_iters=2000 \
    --eval_interval=200
```

## 設定の詳細

### インストラクションチューニング固有の設定

```python
# 事前学習済みモデルの読み込み
init_from = 'out_pretrain/ckpt.pt'  # 事前学習済みモデルのパス

# 指示応答形式の設定
format_instruction = True  # 指示形式でフォーマット
instruction_template = "chat"  # チャット形式のテンプレートを使用

# データセット設定
dataset_name = "kunishou/databricks-dolly-15k-ja"
text_column = "text"  # フォーマット後のテキストカラム
streaming = False  # 小さなデータセットなのでストリーミング不要
```

### インストラクションチューニング用最適化設定

```python
# メモリ効率的な設定
batch_size = 8  # インストラクションチューニング用
block_size = 1024  # 指示応答ペアに対応するため長めのシーケンス
gradient_accumulation_steps = 4  # 実効バッチサイズ32

# ファインチューニング設定
learning_rate = 1e-5  # ファインチューニング用の低い学習率
max_iters = 5000  # インストラクションチューニングは短期間
weight_decay = 0.01  # 軽い正則化
dropout = 0.1  # ファインチューニングでは少し高め
```

## データフォーマット例

元のデータ：
```json
{
  "instruction": "ヴァージン・オーストラリア航空はいつから運航を開始したのですか？",
  "input": "ヴァージン・オーストラリア航空...(長い文脈)",
  "output": "2000年8月31日に運航を開始しました。"
}
```

チャットテンプレートでフォーマット後：
```
<|user|>
ヴァージン・オーストラリア航空はいつから運航を開始したのですか？

ヴァージン・オーストラリア航空...(長い文脈)
<|assistant|>
2000年8月31日に運航を開始しました。
```

## 推奨学習環境

### GPU要件
- **8GB VRAM**: batch_size=4, block_size=512
- **16GB VRAM**: batch_size=8, block_size=1024
- **24GB VRAM**: batch_size=16, block_size=1024

### インストラクションチューニング時間の目安
- **推奨終了**: 1,500-2,000イテレーション（約20-30分）
- **最大学習**: 5,000イテレーション（約1-1.5時間・RTX 3080/4080級）
- **実際の収束**: 多くの場合2,000イテレーション以下で十分
- **チェックポイント保存**: 200イテレーションごと
- **評価**: 200イテレーションごと

**効率的なインストラクションチューニング方法:**
1. 2,000イテレーションまで学習
2. 検証損失が学習損失より大幅に高くなったら早期停止
3. 過学習の兆候（検証損失の上昇）に注意

### インストラクションチューニング進行の目安
```
イテレーション 0: 損失 ~1.5-3.0（事前学習済みモデルの初期状態）
イテレーション 500: 損失 ~0.8-1.2（インストラクション形式に適応）
イテレーション 1,500: 損失 ~0.3-0.6（収束開始）
イテレーション 2,000: 損失 ~0.1-0.3（実用的収束）★推奨終了点
イテレーション 3,000+: 損失 ~0.05-0.2（収束済み・過学習に注意）
```

**実際の収束パターン:**
- 1500-2000イテレーション: 損失が0.5以下で安定
- 2000イテレーション以降: 検証損失が学習損失を上回り始める場合がある
- **推奨**: 2000-3000イテレーションで終了、過学習に注意

**注意**: 
- 検証損失が学習損失より大幅に高くなったら過学習の兆候
- インストラクションチューニングは過学習しやすいため早期停止が重要
- 事前学習済みモデルから開始するため、収束が早い

## 学習結果の確認

### チェックポイント
インストラクションチューニング済みモデルは`out_dolly_ja/`ディレクトリに保存されます：

```
out_dolly_ja/
├── ckpt.pt          # インストラクションチューニング済みモデル
└── meta.pkl         # トークナイザーメタ情報
```

### 損失の推移例
```
ステップ 0: 学習損失 2.1523, 検証損失 2.1847（事前学習済みモデルの初期状態）
ステップ 200: 学習損失 1.2340, 検証損失 1.2678（インストラクション形式適応中）
ステップ 1000: 学習損失 0.5234, 検証損失 0.5891（収束中）
ステップ 2000: 学習損失 0.1987, 検証損失 0.2456（実用的収束）
```

**判断基準**: 
- 損失が0.5以下: インストラクション形式に適応済み
- 検証損失≈学習損失: 過学習なし
- 検証損失 > 学習損失（大幅）: 過学習の兆候（早期停止推奨）

## モデルの使用

インストラクションチューニング済みモデルは`chat.py`で対話に使用できます：

```bash
# インストラクションチューニング済みモデルで対話
python chat.py --init_from=resume --out_dir=out_dolly_ja
```

## 完全な学習フロー例

```bash
# ステップ1: 事前学習（数時間-数日）
python train.py config/pretrain_japanese.py --max_iters=50000

# ステップ2: インストラクションチューニング（20-30分）
python finetune_instruction.py config/train_dolly_ja.py --max_iters=2000

# ステップ3: 対話テスト
python chat.py --init_from=resume --out_dir=out_dolly_ja
```

## トラブルシューティング

### メモリ不足の場合
```python
batch_size = 4  # バッチサイズを小さく
block_size = 512  # シーケンス長を短く
gradient_accumulation_steps = 8  # 勾配累積で実効バッチサイズを維持
```

### 学習が遅い場合
```python
eval_interval = 1000  # 評価間隔を長く
log_interval = 100  # ログ間隔を長く
compile = True  # PyTorchコンパイルを有効化（環境が対応している場合）
```

### 過学習を避けるために
- **早期停止**: 検証損失が上昇し始めたら学習停止
- **ドロップアウト調整**: `dropout=0.2`
- **学習率を下げる**: `learning_rate=5e-6`
- **正則化強化**: `weight_decay=0.1`

### 事前学習済みモデルが見つからない場合
```bash
# 事前学習を先に実行
python train.py config/pretrain_japanese.py --max_iters=10000

# その後インストラクションチューニング
python finetune_instruction.py config/train_dolly_ja.py
```

## 高度な使用例

### 異なる事前学習済みモデルからの開始
```bash
# GPT-2からのインストラクションチューニング
python finetune_instruction.py config/train_dolly_ja.py --init_from=gpt2

# カスタム事前学習済みモデルからの開始
python finetune_instruction.py config/train_dolly_ja.py --init_from=my_pretrained_model/ckpt.pt
```

### 学習率の段階的調整
```bash
# 第1段階: 高い学習率でクイック適応
python finetune_instruction.py config/train_dolly_ja.py \
    --learning_rate=1e-4 --max_iters=1000

# 第2段階: 低い学習率で細かい調整
python finetune_instruction.py config/train_dolly_ja.py \
    --init_from=resume --learning_rate=1e-6 --max_iters=2000
```

### 分散学習
```bash
# マルチGPUでのインストラクションチューニング
torchrun --standalone --nproc_per_node=4 finetune_instruction.py config/train_dolly_ja.py
```

この2段階プロセスにより、事前学習で言語理解能力を獲得し、インストラクションチューニングで指示追従能力を特化させた高品質な日本語GPTモデルを効率的に作成できます。
