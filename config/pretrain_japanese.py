"""
日本語事前学習用設定ファイル

このファイルは、日本語の大規模テキストデータセットで事前学習を行うための設定を提供します。
使用データセット:
- ce-lery/mistral-3b-dataset: 高品質な日本語テキスト
- fujiki/wiki40b_ja: 日本語Wikipedia

使用例:
$ python train.py config/pretrain_japanese.py
"""

# 事前学習の設定（4つの大規模日本語データセット混合）
use_external_dataset = True  # 外部データセットを使用
use_multi_dataset = True  # 複数データセット混合モードを有効

# 混合データセットの設定（確実に動作するデータセットから開始）
mixed_datasets = [
    {
        "dataset_name": "ce-lery/mistral-3b-dataset",
        "dataset_config": None,
        "text_column": "text", 
        "streaming": True,
        "tokenizer_type": "gpt2",
        "block_size": 512,
        "batch_size": 1,
        "seed": 1337,
        "format_instruction": False,
        "instruction_template": None
    },
    {
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-103-raw-v1",
        "text_column": "text",
        "streaming": True,
        "tokenizer_type": "gpt2",
        "block_size": 512,
        "batch_size": 1,
        "seed": 1338,
        "format_instruction": False,
        "instruction_template": None
    }
]

# データセット混合比率（テスト用に2つのデータセット）
# 段階的アプローチ：まず確実に動作する2つから開始
dataset_mix_ratios = [
    0.8,  # ce-lery/mistral-3b-dataset - 高品質な日本語
    0.2   # wikitext - 英語だが安定したベースライン
]

# 学習スケジュール（全データ活用のため大幅延長）
max_iters = 1000000  # 全データセットを活用するため大幅に延長
warmup_iters = 50000  # 大規模データに対応した長いウォームアップ  
lr_decay_iters = 1000000  # 全学習期間に渡って減衰

# フォールバック設定（混合データセットが利用できない場合）
dataset_name = "ce-lery/mistral-3b-dataset"  # より確実な日本語データセット
dataset_config = None  # configなし
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
eval_interval = 5000  # 評価間隔（長期学習用に延長）
log_interval = 1000   # ログ出力間隔（長期学習用に延長）
eval_iters = 100      # 評価イテレーション数（時間節約）
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # スクラッチから開始

# モデル設定（大規模データセット対応）
n_layer = 16  # 大規模データに見合ったサイズに拡張
n_head = 16  # アテンションヘッド数
n_embd = 1024  # 埋め込み次元（約250Mパラメータ）
dropout = 0.1  # 大規模データでは過学習リスク低め
bias = False

# 学習設定（大規模データセット用）
gradient_accumulation_steps = 64  # 大規模データセットなので実効バッチサイズを大きく
batch_size = 1  # メモリ節約
block_size = 512  # シーケンス長

# 事前学習用最適化設定（全データ完全活用）
learning_rate = 3e-4  # 事前学習用学習率
max_iters = 4600000  # 全データを3周使用（151.8B ÷ 32.8K ≈ 4.6M iter）
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 学習率スケジューラー（超長期学習用）
decay_lr = True
warmup_iters = 100000  # 大規模データに対応した長いウォームアップ
lr_decay_iters = 4600000  # 全学習期間に渡って減衰
min_lr = 1e-5  # より低い最小学習率

# 評価とログの設定（長期学習用に調整）
eval_interval = 5000  # 評価間隔を長く（時間節約）
log_interval = 1000   # ログ間隔も長く
eval_iters = 100      # 評価イテレーション数を削減

# ログ記録
wandb_log = True
wandb_project = 'nanogpt-japanese-pretrain'
wandb_run_name = 'pretrain-japanese'

# システム設定
device = 'cuda'
dtype = 'float16'  # PyTorchインポート前なので固定値
compile = False  # 安定性のため無効
