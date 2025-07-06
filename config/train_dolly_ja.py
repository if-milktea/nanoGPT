"""
日本語指示応答データセット（kunishou/databricks-dolly-15k-ja）用のインストラクションチューニング設定

このファイルは、事前学習済みモデルを日本語の指示応答データセットでファインチューニングするための設定を提供します。

使用例:
$ python finetune_instruction.py config/train_dolly_ja.py

前提条件:
- 事前にtrain.pyで事前学習を実行し、out_pretrainにモデルを保存しておく
"""

# 外部データセットの設定
use_external_dataset = True  # 外部データセットを使用するかどうか
dataset_name = "kunishou/databricks-dolly-15k-ja"  # Hugging Faceデータセット名
dataset_config = None  # データセット設定（Noneの場合はデフォルト）
text_column = "formatted_text"  # 後でformat_instructionで作成するカラム名
streaming = False  # 小さなデータセットなのでストリーミング不要
cache_dir = None  # キャッシュディレクトリ（Noneの場合はデフォルト）

# 日本語対応の設定
format_instruction = True  # 指示形式でテキストをフォーマットするか
instruction_template = """<|system|>
あなたは親切で知識豊富なAIアシスタントです。

<|user|>
{instruction}
{input}

<|assistant|>
{output}"""  # モダンなチャット形式のテンプレート

# トークナイザー設定
tokenizer_type = "gpt2"  # "gpt2"または"custom"

# トレーニング設定（インストラクションチューニング用に調整）
# 基本設定
out_dir = 'out_dolly_ja'
eval_interval = 500
log_interval = 50
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'out_pretrain/ckpt.pt'  # 事前学習済みモデルから開始

# データ設定
gradient_accumulation_steps = 8  # より多くの勾配累積
batch_size = 4  # バッチサイズを小さくして安定性向上
block_size = 512  # より短いシーケンスで確実な学習

# モデル設定（事前学習済みモデルから継承されるため通常は変更不要）
n_layer = 16  # 事前学習と同じ（必要に応じて上書き可能）
n_head = 16   # 事前学習と同じ  
n_embd = 1024 # 事前学習と同じ
dropout = 0.1 # ファインチューニングでは少し高め
bias = False

# 最適化設定（ファインチューニング用）
learning_rate = 5e-6  # さらに低い学習率でより慎重に学習
max_iters = 3000  # 十分な学習時間を確保
weight_decay = 0.1   # より強い正則化
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 学習率スケジューラー（ファインチューニング用）
decay_lr = True
warmup_iters = 100    # 短いウォームアップ
lr_decay_iters = 5000 # 全学習期間に渡って減衰
min_lr = 1e-6         # 最小学習率

# ログ記録
wandb_log = True
wandb_project = 'nanogpt-japanese-instruction'
wandb_run_name = 'dolly-ja-finetune'

# システム設定
device = 'cuda'
dtype = 'float16'  # PyTorchインポート前なので固定値を使用
compile = False
