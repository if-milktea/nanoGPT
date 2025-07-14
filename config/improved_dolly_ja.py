"""
改善されたDolly-JAインストラクションチューニング設定
対話品質向上のための設定変更
"""

# 外部データセットの設定
use_external_dataset = True
dataset_name = "kunishou/databricks-dolly-15k-ja"
dataset_config = None
text_column = "formatted_text"
streaming = False
cache_dir = None

# 日本語対応の設定
format_instruction = True
# より対話に適したテンプレート
instruction_template = """<|system|>
あなたは親切で知識豊富な日本語AIアシスタントです。ユーザーの質問に対して、自然で役立つ回答を提供してください。

<|user|>
{instruction}
{input}

<|assistant|>
{output}<|endoftext|>"""  # 明確な終了トークンを追加

# トークナイザー設定
tokenizer_type = "gpt2"

# 改善されたトレーニング設定
out_dir = 'out_dolly_ja_improved'
eval_interval = 250  # より頻繁な評価
log_interval = 25   # より頻繁なログ
eval_iters = 200    # より多くの評価イテレーション
eval_only = False
always_save_checkpoint = True
init_from = 'out_pretrain/ckpt.pt'

# データ設定を改善
gradient_accumulation_steps = 16  # より大きなeffective batch size
batch_size = 8   # バッチサイズを倍増
block_size = 1024  # より長いコンテキスト

# モデル設定
n_layer = 16
n_head = 16
n_embd = 1024
dropout = 0.05  # より低いドロップアウト
bias = False

# 最適化設定を改善
learning_rate = 1e-5  # より高い学習率
max_iters = 8000     # より多くの学習イテレーション
weight_decay = 0.05  # より弱い正則化
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 学習率スケジューラー
decay_lr = True
warmup_iters = 200   # より長いウォームアップ
lr_decay_iters = 8000
min_lr = 1e-7

# ログ記録
wandb_log = True
wandb_project = 'nanogpt-japanese-instruction-improved'
wandb_run_name = 'dolly-ja-improved'

# システム設定
device = 'cuda'
dtype = 'float16'
compile = False