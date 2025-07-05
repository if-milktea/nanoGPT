"""
事前学習用の設定ファイル

ce-lery/mistral-3b-datasetとfujiki/wiki40b_jaで事前学習を実行
"""

# 外部データセットの設定
use_external_dataset = True
dataset_name = "ce-lery/mistral-3b-dataset"  # メインデータセット
secondary_dataset = "fujiki/wiki40b_ja"  # 日本語Wiki追加データセット
dataset_config = None
text_column = "text"
streaming = True
cache_dir = None
tokenizer_type = "gpt2"

# 事前学習設定
format_instruction = False  # 事前学習では指示フォーマットは使わない
instruction_template = None

# 基本設定
out_dir = 'out_pretrain'
eval_interval = 2000  # 事前学習では評価間隔を長く
log_interval = 50
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# データ設定（事前学習用）
gradient_accumulation_steps = 32  # 大きなバッチサイズで安定学習
batch_size = 4
block_size = 1024

# モデル設定（事前学習用に大きめ）
n_layer = 16
n_head = 16  
n_embd = 1024
dropout = 0.1
bias = False

# 事前学習用最適化設定
learning_rate = 3e-4  # 事前学習用の学習率
max_iters = 100000  # 事前学習は長期間
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 学習率スケジューラー（事前学習用）
decay_lr = True
warmup_iters = 5000  # 長めのウォームアップ
lr_decay_iters = 100000
min_lr = 3e-5

# ログ記録
wandb_log = True
wandb_project = 'nanogpt-pretrain'
wandb_run_name = 'mistral-wiki-pretrain'

# システム設定
device = 'cuda'
dtype = 'float16'
compile = False
