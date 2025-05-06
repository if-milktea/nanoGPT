"""
このトレーニングスクリプトは、以下の2つの方法で実行できます：
1. シングルGPUでのデバッグモード
2. 分散データ並列（DDP）を使用した大規模トレーニング

シングルGPUでの実行例：
$ python train.py --batch_size=32 --compile=False

1つのノードで4つのGPUを使用したDDPでの実行例：
$ torchrun --standalone --nproc_per_node=4 train.py

2つのノードで4つのGPUを使用したDDPでの実行例：
- 最初の（マスター）ノードで実行（例：IPアドレス 123.456.123.456）：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- ワーカーノードで実行：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
（InfiniBandインターコネクトがないクラスタの場合は、NCCL_IB_DISABLE=1を先頭に付けてください）
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

# -----------------------------------------------------------------------------
# OpenWebTextでGPT-2（124M）を学習するためのデフォルト設定値
# 入出力設定
out_dir = 'out'
eval_interval = 500    # 評価間隔を短縮して進捗を把握しやすく
log_interval = 5       # ログ出力間隔を増加
eval_iters = 100      # 評価イテレーション数を削減
eval_only = False # Trueの場合、最初の評価後にスクリプトを終了
always_save_checkpoint = True # Trueの場合、評価後に常にチェックポイントを保存
init_from = 'scratch' # 'scratch'または'resume'またはGPT-2モデル名
# wandbによるログ記録
wandb_log = True # デフォルトで無効
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# データ設定
dataset = 'shakespeare'  # シェイクスピアデータセット用に変更
gradient_accumulation_steps = 4    # メモリ使用量削減のため調整
batch_size = 8  # バッチサイズを削減して8GB VRAM対応
block_size = 256  # シーケンス長を削減
# モデル設定
n_layer = 8          # レイヤー数を削減
n_head = 8           # ヘッド数を削減
n_embd = 512         # 埋め込みサイズを削減
dropout = 0.1        # ドロップアウトを追加してオーバーフィッティング防止
bias = False # LayerNormとLinear層でバイアスを使用するか
# AdamWオプティマイザー設定
learning_rate = 6e-4 # 最大学習率
max_iters = 10000  # イテレーション数を削減（シェイクスピアデータセットは小規模なため）
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # この値で勾配をクリッピング（0.0の場合は無効）
# 学習率減衰設定
decay_lr = True # 学習率を減衰させるかどうか
warmup_iters = 500  # ウォームアップステップ数を削減
lr_decay_iters = 10000  # max_itersに合わせて学習率の減衰スケジュールを調整
min_lr = 6e-5 # 最小学習率（Chinchillaに従い、learning_rate/10程度にすべき）
# DDP設定
backend = 'nccl' # 'nccl', 'gloo'など
# システム設定
device = 'cuda' # 'cpu', 'cuda', 'cuda:0', 'cuda:1'など、またはMacbookでは'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', または 'float16'（後者は自動的にGradScalerを実装）
compile = False # コンパイルを無効化してTritonエラーを回避

# エラー抑制の設定を追加
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # コマンドラインまたは設定ファイルからの上書き
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
print(f"イテレーションあたりのトークン数: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # matmulでtf32を許可
torch.backends.cudnn.allow_tf32 = True # cudnnでtf32を許可
device_type = 'cuda' if 'cuda' in device else 'cpu' # 後で torch.autocast で使用
# float16データ型は自動的にGradScalerを使用
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# シンプルなデータローダー
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # メモリリーク防止のため、バッチごとにnp.memmapを再作成
    # 参照: https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # x,yを非同期でGPUに転送できるようにピン止めメモリを使用
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# 変数の初期化（init_from='resume'の場合は上書きされる）
iter_num = 0
best_val_loss = 1e9

# データセットからvocab_sizeを推定
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"vocab_size = {meta_vocab_size} を {meta_path} から読み込みました")

# モデルの初期化
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # コマンドラインからのmodel_argsで開始
if init_from == 'scratch':
    # 新しいモデルをスクラッチから初期化
    print("新しいモデルをスクラッチから初期化します")
    # スクラッチ学習用のvocab_sizeを決定
    if meta_vocab_size is None:
        print("GPT-2のvocab_sizeをデフォルトの50304（50257を効率のため切り上げ）に設定")
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
    # その他の属性（dropout等）はコマンドラインから指定可能
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # モデルを作成
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # state dictのキーを修正（デバッグ要）
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"OpenAI GPT-2の重みから初期化: {init_from}")
    # OpenAI GPT-2の重みから初期化
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # 作成された設定パラメータを読み取り、チェックポイントに正しく保存できるようにする
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# 必要に応じてモデルのブロックサイズを縮小
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # チェックポイントに正しい値を保存するため
model.to(device)

# GradScalerの初期化（enabled=Falseの場合はno-op）
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# オプティマイザー
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # メモリ解放

# モデルのコンパイル
if compile:
    print("モデルをコンパイルしています...（約1分かかります）")
    unoptimized_model = model
    model = torch.compile(model) # PyTorch 2.0が必要

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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 係数は0..1
    return min_lr + coeff * (learning_rate - min_lr)

# ログ記録の設定
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 学習ループ
X, Y = get_batch('train') # 最初のバッチを取得
t0 = time.time()
local_iter_num = 0 # このプロセスの生存期間中のイテレーション数
raw_model = model.module if ddp else model # DDPコンテナが必要な場合はアンラップ
running_mfu = -1.0
while True:

    # このイテレーションの学習率を決定して設定
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # train/valセットの損失を評価し、チェックポイントを保存
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"ステップ {iter_num}: 学習損失 {losses['train']:.4f}, 検証損失 {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # パーセンテージに変換
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"チェックポイントを {out_dir} に保存します")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # 勾配累積を使用してより大きなバッチサイズをシミュレート
    # float16データ型の場合はGradScalerを使用
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # DDPトレーニングでは最後のマイクロステップでのみ勾配の同期が必要
            # 公式の方法はmodel.no_sync()コンテキストマネージャーを使用することですが
            # コードが膨らみ、繰り返しが必要になるため好ましくありません
            # そのコンテキストマネージャーのソースを見ると、この変数を切り替えているだけです
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # 勾配累積を考慮して損失をスケーリング
        # モデルがGPUでforward passを実行している間に次のバッチを非同期にプリフェッチ
        X, Y = get_batch('train')
        # 勾配のスケーリングありでバックワードパス（fp16の場合）
        scaler.scale(loss).backward()
    # 勾配クリッピング
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # オプティマイザーのステップとfp16用のスケーラーの更新
    scaler.step(optimizer)
    scaler.update()
    # 不要な勾配をできるだけ早くフラッシュ
    optimizer.zero_grad(set_to_none=True)

    # タイミングとログ記録
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # 損失を浮動小数点として取得（これはCPU-GPU同期ポイント）
        # 上記の除算を元に戻すためスケールアップし、真の合計損失を近似
        # （正確には合計になるはずですが）
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # 学習ループが落ち着くまで待機
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"イテレーション {iter_num}: 損失 {lossf:.4f}, 時間 {dt*1000:.2f}ms, MFU {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 終了条件
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
