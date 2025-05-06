"""
学習済みモデルからサンプリングを行う
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 初期化パラメータの設定
init_from = 'resume' # 'resume'（out_dirから再開）または 'gpt2-xl'などのGPT-2モデルを指定
out_dir = 'out' # init_fromが'resume'でない場合は無視される
start = "\n" # または "<|endoftext|>" など。ファイルも指定可能："FILE:prompt.txt"
num_samples = 10 # 生成するサンプル数
max_new_tokens = 500 # 各サンプルで生成するトークン数
temperature = 0.8 # 1.0 = 変更なし, < 1.0 = よりランダム性が低い, > 1.0 = よりランダム性が高い
top_k = 200 # 最も確率の高いtop_kのトークンのみを保持し、他は確率を0に設定
seed = 1337 # 乱数シード
device = 'cuda' # 使用デバイス: 'cpu', 'cuda', 'cuda:0', 'cuda:1' など
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32'または'bfloat16'または'float16'
compile = False # PyTorch 2.0のモデルコンパイル機能を使用して高速化
exec(open('configurator.py').read()) # コマンドラインまたは設定ファイルからの上書き
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # matmulでtf32を許可
torch.backends.cudnn.allow_tf32 = True # cudnnでtf32を許可
device_type = 'cuda' if 'cuda' in device else 'cpu' # 後で torch.autocast で使用
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# モデルの初期化
if init_from == 'resume':
    # 特定のディレクトリに保存されたモデルから初期化
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # 指定されたGPT-2モデルから初期化
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # PyTorch 2.0が必要（オプション）

# データセットフォルダ内にmetaピクルファイルがあれば読み込む
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # 古いチェックポイントにはない場合がある
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"メタデータを {meta_path} から読み込んでいます...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # 将来的に任意のエンコーダー/デコーダースキームに対応させたい
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # デフォルトでGPT-2のエンコーディングを使用
    print("meta.pklが見つからないため、GPT-2エンコーディングを使用します...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# プロンプトの開始部分をエンコード
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# テキスト生成の実行
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
