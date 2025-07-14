"""
モデルとの対話を行うプログラム
"""
import os
import pickle
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 初期化パラメータの設定
# デフォルトのシステムプロンプト
system_prompt = "あなたは人間のために働くAIエージェントです。以下の質問とお問い合わせにお答えください。"

init_from = 'resume' # 'resume'（out_dirから再開）または 'gpt2-xl'などのGPT-2モデルを指定
out_dir = 'out_dolly_ja_rtx3060' # init_fromが'resume'でない場合は無視される
temperature = 0.3 # より一貫した応答のため低めに設定
top_k = 50 # より制限的にして一貫性を向上
max_new_tokens = 200  # より長い応答を許可
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
    meta_path = None
    
    # インストラクションチューニング済みモデルの場合、out_dirからメタデータを読み込む
    if checkpoint and init_from == 'resume':
        meta_path = os.path.join(out_dir, 'meta.pkl')
        load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"インストラクションチューニング用メタデータを {meta_path} から読み込んでいます...")
    
    # 従来のデータセット固有のメタファイルもチェック
    if not load_meta and checkpoint and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"データセット用メタデータを {meta_path} から読み込んでいます...")

    if load_meta:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        # メタデータの構造を確認
        if 'stoi' in meta and 'itos' in meta:
            # 従来の文字レベルトークナイザー
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        elif 'tokenizer_type' in meta and meta['tokenizer_type'] == 'gpt2':
            # GPT-2トークナイザーを使用
            print("メタデータでGPT-2トークナイザーが指定されています...")
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)
        else:
            print("不明なメタデータ形式です。GPT-2エンコーディングを使用します...")
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)
    else:
        print("メタデータが見つかりません。GPT-2エンコーディングを使用します...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return encode, decode

def generate_response(model, ctx, encode, decode, user_input):
    """ユーザー入力に対する応答を生成"""
    # インストラクションチューニング時と同じチャット形式を使用
    formatted_prompt = f"""<|system|>
あなたは親切で知識豊富なAIアシスタントです。

<|user|>
{user_input}

<|assistant|>
"""
    
    print(f"送信プロンプト: {repr(formatted_prompt)}")  # デバッグ用
    
    input_ids = encode(formatted_prompt)
    print(f"エンコード済みトークン数: {len(input_ids)}")  # デバッグ用
    print(f"最初の10トークン: {input_ids[:10]}")  # デバッグ用
    
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
    
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            raw_response = decode(y[0].tolist())
            
            print(f"生の応答: {repr(raw_response)}")  # デバッグ用
            
            # 入力プロンプト部分を除去して応答のみを取得
            input_length = len(input_ids)
            output_ids = y[0][input_length:].tolist()
            response = decode(output_ids)
            
            # 特殊トークンで分割して最初の部分のみを取得
            for stop_seq in ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>"]:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
                    break
            
            # 余分な改行や空白を削除
            lines = response.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line:  # 空行をスキップ
                    cleaned_lines.append(line)
            
            response = '\n'.join(cleaned_lines)
            
            # 応答が空の場合のフォールバック
            if not response.strip():
                response = "申し訳ありませんが、適切な応答を生成できませんでした。"
            
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
