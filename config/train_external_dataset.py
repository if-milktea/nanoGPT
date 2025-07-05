"""
外部データセット使用時のトレーニング設定

このファイルは、Hugging Face Datasetsライブラリを使用して
外部データセットでモデルを学習するための設定を提供します。

使用例:
$ python train.py config/train_external_dataset.py --dataset_name=wikitext
"""

# 外部データセットの設定
use_external_dataset = True  # 外部データセットを使用するかどうか
dataset_name = "wikitext"  # Hugging Faceデータセット名
dataset_config = "wikitext-103-raw-v1"  # データセット設定（Noneの場合はデフォルト）
text_column = "text"  # テキストデータが含まれるカラム名
streaming = True  # ストリーミングモードを使用するか
cache_dir = None  # キャッシュディレクトリ（Noneの場合はデフォルト）

# トークナイザー設定
tokenizer_type = "gpt2"  # "gpt2"または"custom"

# トレーニング設定（外部データセット用に調整）
# 基本設定
out_dir = 'out_external'
eval_interval = 1000
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# データ設定
gradient_accumulation_steps = 8  # より大きな実効バッチサイズ
batch_size = 4  # メモリ効率のため小さく設定
block_size = 512  # 中程度のシーケンス長

# モデル設定（中規模モデル）
n_layer = 12
n_head = 12  
n_embd = 768
dropout = 0.1
bias = False

# 最適化設定
learning_rate = 3e-4  # 外部データセット用に調整
max_iters = 50000  # より長期間の学習
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 学習率スケジューラー
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 50000
min_lr = 3e-5

# ログ記録
wandb_log = True
wandb_project = 'nanogpt-external'
wandb_run_name = 'external-dataset'

# システム設定
device = 'cuda'
dtype = 'float16'  # PyTorchインポート前なので固定値を使用
compile = False
