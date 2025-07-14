"""
RTX3060 8GB用 Dolly-JAインストラクションチューニング設定
メモリ制約に配慮した設定
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
instruction_template = """<|system|>
あなたは親切で知識豊富な日本語AIアシスタントです。

<|user|>
{instruction}
{input}

<|assistant|>
{output}<|endoftext|>"""

# トークナイザー設定
tokenizer_type = "gpt2"

# RTX3060 8GB用設定
out_dir = 'out_dolly_ja_rtx3060'
eval_interval = 500
log_interval = 50
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'out_pretrain/ckpt.pt'

# メモリ制約に配慮したデータ設定
gradient_accumulation_steps = 32  # 大きな累積でeffective batch sizeを確保
batch_size = 1   # 最小バッチサイズ
block_size = 512  # 短いコンテキスト

# 小さめのモデル設定（8GBに収まるサイズ）
n_layer = 12     # レイヤー数を削減
n_head = 12      # ヘッド数を削減
n_embd = 768     # 埋め込み次元を削減
dropout = 0.1
bias = False

# 最適化設定
learning_rate = 8e-6  # やや高めの学習率
max_iters = 5000     # 適度な学習回数
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 学習率スケジューラー
decay_lr = True
warmup_iters = 150
lr_decay_iters = 5000
min_lr = 1e-6

# ログ記録
wandb_log = True
wandb_project = 'nanogpt-japanese-rtx3060'
wandb_run_name = 'dolly-ja-rtx3060'

# システム設定
device = 'cuda'
dtype = 'float16'  # メモリ節約のためfloat16を使用
compile = False

# RTX3060向けの追加最適化
# これらの設定はtrainスクリプト内で使用される
use_mixed_precision = True  # 混合精度を使用
dataloader_num_workers = 0  # CPUワーカー数を0に（メモリ節約）