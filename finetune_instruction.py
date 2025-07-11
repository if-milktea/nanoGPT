"""
インストラクションチューニング用トレーニングスクリプト
このスクリプトは、事前学習済みGPTモデルをインストラクションデータでファインチューニングします。

事前学習 → インストラクションチューニングの2段階学習の第2段階です。

実行例:
$ python finetune_instruction.py config/train_dolly_ja.py

使用方法:
1. 事前に train.py で事前学習を実行し、out_pretrain にモデルを保存しておく
2. このスクリプトで事前学習済みモデルをインストラクションデータでファインチューニング

実行例:
# 事前学習済みモデルのファインチューニング
$ python finetune_instruction.py config/train_dolly_ja.py

# 特定のチェックポイントからのファインチューニング
$ python finetune_instruction.py --init_from="out_pretrain/ckpt.pt"

シングルGPUでのデバッグモード:
$ python finetune_instruction.py --batch_size=16 --compile=False

分散データ並列（DDP）での実行:
$ torchrun --standalone --nproc_per_node=4 finetune_instruction.py config/train_dolly_ja.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from dataset_loader import create_hf_dataloader, DATASET_CONFIGS

# -----------------------------------------------------------------------------
# インストラクションチューニング用のデフォルト設定値
# 入出力設定
out_dir = 'out_dolly_ja'  # インストラクションチューニングモデルの保存先
eval_interval = 500      # インストラクションチューニングでは頻繁に評価
log_interval = 50        # ログ出力間隔（頻繁にログを出力）
eval_iters = 100         # 評価イテレーション数
eval_only = False        # Trueの場合、最初の評価後にスクリプトを終了
always_save_checkpoint = True # Trueの場合、評価後に常にチェックポイントを保存
init_from = 'out_pretrain/ckpt.pt' # 事前学習済みモデルから開始
# wandbによるログ記録
wandb_log = True # デフォルトで有効
wandb_project = 'nanogpt-japanese-instruction'
wandb_run_name = 'dolly-ja-finetune'
# データ設定
dataset = 'dolly_ja'  # インストラクション用データセット
# 外部データセット設定（インストラクションチューニング用）
use_external_dataset = True  # 外部データセットを使用
dataset_name = "kunishou/databricks-dolly-15k-ja"  # Hugging Faceデータセット名
dataset_config = None  # データセット設定
text_column = "text"  # テキストカラム名
streaming = False  # インストラクションデータは小さいのでストリーミングなし
cache_dir = None  # キャッシュディレクトリ
tokenizer_type = "gpt2"  # トークナイザーの種類
# 指示応答形式設定（インストラクションチューニングでは必須）
format_instruction = True  # インストラクション形式でフォーマット
instruction_template = "chat"  # チャット形式のテンプレートを使用
# インストラクションチューニング用バッチ設定
gradient_accumulation_steps = 4  # ファインチューニング用
batch_size = 8  # インストラクションチューニング用バッチサイズ
block_size = 512  # コンテキスト長（事前学習済みモデルと同じサイズに設定）
# モデル設定（事前学習済みモデルから継承されるため通常は不要だが、上書き可能）
n_layer = 16          # 事前学習と同じ
n_head = 16           # 事前学習と同じ
n_embd = 1024         # 事前学習と同じ
dropout = 0.1         # ファインチューニングでは少し高め
bias = False          # 事前学習と同じ
# AdamWオプティマイザー設定（ファインチューニング用）
learning_rate = 1e-5  # ファインチューニング用の低い学習率
max_iters = 5000      # インストラクションチューニングは短期間
weight_decay = 0.01   # 軽い正則化
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0       # 勾配クリッピング
# 学習率減衰設定（ファインチューニング用）
decay_lr = True       # 学習率を減衰
warmup_iters = 100    # 短いウォームアップ
lr_decay_iters = 5000 # 全学習期間に渡って減衰
min_lr = 1e-6         # 最小学習率
# DDP設定
backend = 'nccl'      # 'nccl', 'gloo'など
# システム設定
device = 'cuda'       # 'cpu', 'cuda', 'cuda:0', 'cuda:1'など、またはMacbookでは'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False       # 安定性のため無効

# エラー抑制の設定を追加
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py', encoding='utf-8').read()) # コマンドラインまたは設定ファイルからの上書き
config = {k: globals()[k] for k in config_keys} # ログ記録に使用
# -----------------------------------------------------------------------------

# 初期化、派生属性、I/O設定
ddp = int(os.environ.get('RANK', -1)) != -1 # DDPモードで実行されているか
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # このプロセスがログ記録、チェックポイント保存などを行う
    seed_offset = ddp_rank # 各プロセスに異なるシードを設定
    # プロセス数に応じて勾配累積ステップ数を調整
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # DDPでない場合は単一GPUで実行
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"インストラクションチューニングを開始します")
print(f"事前学習済みモデル読み込み元: {init_from}")
print(f"イテレーションあたりのトークン数: {tokens_per_iter:,}")
print(f"使用データセット: {dataset_name}")
print(f"保存先ディレクトリ: {out_dir}")
print(f"予定学習イテレーション: {max_iters:,}")
print(f"推定学習時間: {max_iters * 0.5 / 3600:.1f}時間（GPU性能に依存）")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # matmulでtf32を許可
torch.backends.cudnn.allow_tf32 = True # cudnnでtf32を許可
device_type = 'cuda' if 'cuda' in device else 'cpu' # 後で torch.autocast で使用
# float16データ型は自動的にGradScalerを使用
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 外部データセット用のデータローダー
external_dataloader = None
if use_external_dataset:
    print(f"インストラクション用外部データセット '{dataset_name}' を設定しています...")
    
    # 設定辞書を作成
    if dataset_name in DATASET_CONFIGS:
        hf_config = DATASET_CONFIGS[dataset_name].copy()
    else:
        hf_config = {}
    
    # 設定を上書き
    hf_config.update({
        'dataset_config': dataset_config,
        'text_column': text_column,
        'streaming': streaming,
        'cache_dir': cache_dir,
        'tokenizer_type': tokenizer_type,
        'block_size': block_size,
        'batch_size': batch_size,
        'seed': 1337,
        'format_instruction': format_instruction,
        'instruction_template': instruction_template
    })
    
    print(f"データローダーに渡すblock_size: {hf_config['block_size']}")
    print(f"ファイナル設定: {hf_config}")
    
    try:
        external_dataloader = create_hf_dataloader(dataset_name, hf_config, device)
        print(f"インストラクション用外部データセット '{dataset_name}' の初期化が完了しました")
        print(f"フォーマット設定: format_instruction={format_instruction}, template={instruction_template}")
    except Exception as e:
        print(f"外部データセットの初期化に失敗しました: {e}")
        print("インストラクションチューニングには外部データセットが必要です")
        raise

def get_batch(split):
    if use_external_dataset and external_dataloader:
        # 外部データセットを使用（インストラクションチューニングでは必須）
        try:
            x, y = external_dataloader.get_batch(split)
            print(f"取得したバッチサイズ: x.shape={x.shape}, y.shape={y.shape}")
            print(f"期待されるblock_size: {block_size}")
            return x, y
        except Exception as e:
            print(f"外部データセットからのバッチ取得エラー: {e}")
            raise
    else:
        raise ValueError("インストラクションチューニングには外部データセットが必要です")

# 変数の初期化（init_from='resume'の場合は上書きされる）
iter_num = 0
best_val_loss = 1e9

# データセットからvocab_sizeを推定
meta_vocab_size = None
if use_external_dataset and external_dataloader:
    # 外部データセットの場合
    meta_vocab_size = external_dataloader.get_vocab_size()
    print(f"外部データセットのvocab_size = {meta_vocab_size}")

# モデルの初期化
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    # スクラッチから開始（通常はインストラクションチューニングでは使用しない）
    print("新しいモデルをスクラッチから初期化します（注意：インストラクションチューニングでは通常事前学習済みモデルを使用します）")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"学習を{out_dir}から再開します")
    # チェックポイントから学習を再開
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # これらの設定属性は学習再開のために一致している必要がある
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # モデルを作成
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # state dictのキーを修正
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif os.path.exists(init_from):
    # 事前学習済みモデルからファインチューニング開始
    print(f"事前学習済みモデルから初期化: {init_from}")
    checkpoint = torch.load(init_from, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    print(f"事前学習済みモデルの設定: {checkpoint_model_args}")
    # 事前学習済みモデルの設定を使用
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # dropout率は更新可能
    model_args['dropout'] = dropout
    print(f"最終的なモデル設定: {model_args}")
    # モデルを作成
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # state dictのキーを修正
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    # インストラクションチューニングでは新しい学習を開始
    iter_num = 0
    best_val_loss = 1e9
    print(f"事前学習済みモデル（イテレーション{checkpoint.get('iter_num', 'unknown')}）からファインチューニングを開始")
elif init_from.startswith('gpt2'):
    print(f"OpenAI GPT-2の重みから初期化: {init_from}")
    # OpenAI GPT-2の重みから初期化
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # 作成された設定パラメータを読み取り
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"不明なinit_from値: {init_from}")

# 必要に応じてモデルのブロックサイズを縮小
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# GradScalerの初期化
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# オプティマイザー（インストラクションチューニング用の設定で新規作成）
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
print(f"インストラクションチューニング用オプティマイザーを作成しました（学習率: {learning_rate}）")

checkpoint = None # メモリ解放

# モデルのコンパイル
if compile:
    print("モデルをコンパイルしています...（約1分かかります）")
    unoptimized_model = model
    model = torch.compile(model)

# モデルをDDPコンテナでラップ
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 任意の分割に対して複数バッチを使用して高精度な損失を推定
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 学習率減衰スケジューラ（ウォームアップ付きコサイン減衰）
def get_lr(it):
    # 1) warmup_itersステップまで線形ウォームアップ
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) lr_decay_iters以降は最小学習率
    if it > lr_decay_iters:
        return min_lr
    # 3) その間はコサイン減衰で最小学習率まで減衰
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ログ記録の設定
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 初期評価
if master_process:
    print("インストラクションチューニング開始前の初期評価を実行します...")
    initial_losses = estimate_loss()
    print(f"初期状態: 学習損失 {initial_losses['train']:.4f}, 検証損失 {initial_losses['val']:.4f}")

# 学習ループ
X, Y = get_batch('train') # 最初のバッチを取得
t0 = time.time()
local_iter_num = 0 # このプロセスの生存期間中のイテレーション数
raw_model = model.module if ddp else model # DDPコンテナが必要な場合はアンラップ
running_mfu = -1.0

print(f"インストラクションチューニングループを開始します（最大{max_iters}イテレーション）...")

while True:

    # このイテレーションの学習率を決定して設定
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # train/valセットの損失を評価し、チェックポイントを保存
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"ステップ {iter_num}: 学習損失 {losses['train']:.4f}, 検証損失 {losses['val']:.4f}")
        
        # 早期停止の判定（インストラクションチューニングでは過学習に注意）
        if iter_num > 0:
            val_improvement = (best_val_loss - losses['val']) / best_val_loss * 100
            if val_improvement > 0:
                print(f"検証損失が改善しました ({val_improvement:.2f}%改善)")
            else:
                print(f"検証損失が悪化しました ({-val_improvement:.2f}%悪化)")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
            
        if losses['val'] < best_val_loss or always_save_checkpoint:
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"新しいベスト検証損失: {best_val_loss:.4f}")
            
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'instruction_tuned': True,  # インストラクションチューニング済みフラグ
                }
                print(f"チェックポイントを {out_dir} に保存します")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                
                # 外部データセット使用時はトークナイザーメタ情報も保存
                if use_external_dataset and external_dataloader:
                    meta_save_path = os.path.join(out_dir, 'meta.pkl')
                    external_dataloader.save_tokenizer_meta(meta_save_path)
                    
    if iter_num == 0 and eval_only:
        break

    # 勾配累積を使用してより大きなバッチサイズをシミュレート
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # 勾配累積を考慮して損失をスケーリング
        # 次のバッチを非同期にプリフェッチ
        X, Y = get_batch('train')
        # バックワードパス
        scaler.scale(loss).backward()
        
    # 勾配クリッピング
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
    # オプティマイザーのステップ
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # タイミングとログ記録
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"イテレーション {iter_num}: 損失 {lossf:.4f}, 時間 {dt*1000:.2f}ms, MFU {running_mfu*100:.2f}%")
        
    iter_num += 1
    local_iter_num += 1

    # 終了条件
    if iter_num > max_iters:
        print(f"最大イテレーション数({max_iters})に到達しました")
        break

print("インストラクションチューニングが完了しました！")
print(f"最終ベスト検証損失: {best_val_loss:.4f}")
print(f"保存先: {out_dir}")

if ddp:
    destroy_process_group()
