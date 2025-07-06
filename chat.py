"""
モデルとの対話を行うプログラム
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 初期化パラメータの設定
# デフォルトのシステムプロンプト
system_prompt = "あなたは人間のために働くAIエージェントです。以下の質問とお問い合わせにお答えください。"

init_from = 'resume' # 'resume'（out_dirから再開）または 'gpt2-xl'などのGPT-2モデルを指定
out_dir = 'out_dolly_ja' # init_fromが'resume'でない場合は無視される
temperature = 0.8 # 1.0 = 変更なし, < 1.0 = よりランダム性が低い, > 1.0 = よりランダム性が高い
top_k = 200 # 最も確率の高いtop_kのトークンのみを保持し、他は確率を0に設定
max_new_tokens = 100  # 各応答で生成する最大トークン数
seed = 1337
device = 'cuda' # 使用デバイス: 'cpu', 'cuda', 'cuda:0', 'cuda:1' など
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # PyTorch 2.0のモデルコンパイル機能を使用して高速化
exec(open('configurator.py').read()) # コマンドラインまたは設定ファイルからの上書き
# -----------------------------------------------------------------------------

def init_model():
    """モデルの初期化を行う"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
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
        model = torch.compile(model)

    return model, ctx

def setup_tokenizer(checkpoint=None):
    """トークナイザーのセットアップ"""
    load_meta = False
    if checkpoint and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)

    if load_meta:
        print(f"メタデータを {meta_path} から読み込んでいます...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        print("GPT-2エンコーディングを使用します...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return encode, decode

def generate_response(model, ctx, encode, decode, user_input):
    """ユーザー入力に対する応答を生成"""
    # インストラクションチューニング時と同じ形式を使用
    formatted_prompt = f"""<|system|>
あなたは親切で知識豊富なAIアシスタントです。

<|user|>
{user_input}

<|assistant|>
"""
    
    input_ids = encode(formatted_prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
    
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            response = decode(y[0].tolist())
            
            # 入力プロンプト部分を除去して応答のみを取得
            # <|assistant|>以降の部分を抽出
            assistant_start = response.find("<|assistant|>\n")
            if assistant_start != -1:
                response = response[assistant_start + len("<|assistant|>\n"):]
            
            # 余分な空白や改行を整理
            response = response.strip()
            
            # 特殊トークンを除去（もしあれば）
            if "<|endoftext|>" in response:
                response = response.split("<|endoftext|>")[0].strip()
            
            return response

def main():
    """メインの対話ループ"""
    print("モデルを初期化中...")
    model, ctx = init_model()
    
    # チェックポイントの読み込み（トークナイザーのセットアップに必要）
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
    else:
        checkpoint = None
    
    encode, decode = setup_tokenizer(checkpoint)
    
    print("\n現在のシステムプロンプト:")
    print(f"「{system_prompt}」")
    print("\n対話を開始します。終了するには 'exit' と入力してください。\n")
    
    while True:
        try:
            user_input = input("\nあなた: ")
            if user_input.lower() == 'exit':
                print("\n対話を終了します。")
                break
            
            if not user_input.strip():
                print("入力が空です。もう一度入力してください。")
                continue
            
            #print("\nAI: ", end='')
            response = generate_response(model, ctx, encode, decode, user_input)
            print(f"\nAI: {response}")
            
        except KeyboardInterrupt:
            print("\n\n対話を終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {str(e)}")
            continue

if __name__ == '__main__':
    main()
