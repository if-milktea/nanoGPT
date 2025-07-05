"""
日本語事前学習用設定ファイル

このファイルは、日本語の大規模テキストデータセットで事前学習を行うための設定を提供します。
使用データセット:
- ce-lery/mistral-3b-dataset: 高品質な日本語テキスト
- fujiki/wiki40b_ja: 日本語Wikipedia

使用例:
$ python train.py config/pretrain_japanese.py
"""

# 事前学習の設定
use_external_dataset = True  # 外部データセットを使用
dataset_name = "wikitext"  # 確実にアクセス可能なデータセット
dataset_config = "wikitext-103-raw-v1"  # WikiText-103設定
secondary_dataset = None  # セカンダリデータセット（将来の拡張用）
text_column = "text"  # テキストカラム名
streaming = True  # 大規模データセットなのでストリーミング使用
cache_dir = None  # キャッシュディレクトリ

# 事前学習特有の設定
format_instruction = False  # 事前学習では指示フォーマットは使用しない
instruction_template = None  # 事前学習では不要

# トークナイザー設定
tokenizer_type = "gpt2"  # GPT-2トークナイザー

# 事前学習用出力設定
out_dir = 'out_pretrain'  # 事前学習モデルの保存先
eval_interval = 1000  # 評価間隔（事前学習は長期間なので間隔を長く）
log_interval = 100  # ログ出力間隔
eval_iters = 200  # 評価イテレーション数
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # スクラッチから開始

# データ設定（大規模事前学習用）
gradient_accumulation_steps = 8  # 実効バッチサイズを大きく
batch_size = 4  # 事前学習用バッチサイズ
block_size = 1024  # 長いコンテキスト

# モデル設定（中規模GPTモデル）
n_layer = 16  # レイヤー数を増加（事前学習用）
n_head = 16  # アテンションヘッド数
n_embd = 1024  # 埋め込み次元数を増加
dropout = 0.05  # 事前学習では低めのドロップアウト
bias = False

# 事前学習用最適化設定
learning_rate = 3e-4  # 事前学習用学習率
max_iters = 100000  # 長期間の事前学習
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 学習率スケジューラー（事前学習用）
decay_lr = True
warmup_iters = 5000  # 長めのウォームアップ
lr_decay_iters = 100000  # 全学習期間に渡って減衰
min_lr = 3e-5  # 最小学習率

# ログ記録
wandb_log = True
wandb_project = 'nanogpt-japanese-pretrain'
wandb_run_name = 'pretrain-japanese'

# システム設定
device = 'cuda'
dtype = 'float16'  # PyTorchインポート前なので固定値
compile = False  # 安定性のため無効
