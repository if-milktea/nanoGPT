"""
RTX3060 8GB用対話プログラム
メモリ効率を重視した設定
"""
import os
import pickle
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import tiktoken
from model import GPTConfig, GPT

# RTX3060用の軽量設定
system_prompt = "あなたは親切で知識豊富な日本語AIアシスタントです。"

init_from = 'resume'
out_dir = 'out_dolly_ja_rtx3060'  # RTX3060用モデル
temperature = 0.7
top_k = 50
max_new_tokens = 100  # トークン数を制限してメモリ節約
seed = 1337
device = 'cuda'
dtype = 'float16'  # メモリ節約
compile = False
exec(open('configurator.py').read())

def init_model():
    """メモリ効率を重視したモデル初期化"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # CUDA設定でメモリ使用量を最適化
    torch.cuda.empty_cache()
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            print(f"警告: {ckpt_path} が見つかりません。元のモデルを試します...")
            ckpt_path = os.path.join('out_dolly_ja', 'ckpt.pt')
            if not os.path.exists(ckpt_path):
                print("エラー: モデルファイルが見つかりません。")
                return None, None
        
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
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    
    # メモリ使用量を表示
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU メモリ使用量: {memory_allocated:.2f}GB / 予約済み: {memory_reserved:.2f}GB")
    
    return model, ctx

def setup_tokenizer(checkpoint=None):
    """トークナイザーのセットアップ"""
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode

def generate_response(model, ctx, encode, decode, user_input):
    """メモリ効率を重視した応答生成"""
    formatted_prompt = f"""<|system|>
{system_prompt}

<|user|>
{user_input}

<|assistant|>
"""
    
    input_ids = encode(formatted_prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
    
    # メモリをクリア
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        with ctx:
            # シンプルな生成を使用（メモリ効率重視）
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            raw_response = decode(y[0].tolist())
            
            # 応答を抽出
            input_length = len(input_ids)
            output_ids = y[0][input_length:].tolist()
            response = decode(output_ids)
            
            # 停止トークンで切断
            for stop_seq in ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>"]:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
                    break
            
            response = response.strip()
            
            if not response:
                response = "申し訳ございませんが、適切な応答を生成できませんでした。"
            
            return response

def main():
    """メイン対話ループ"""
    print("RTX3060用チャットボットを初期化中...")
    print("メモリ制約のため、応答時間が長くなる場合があります。")
    
    model, ctx = init_model()
    if model is None:
        return
    
    # チェックポイント読み込み
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
        else:
            checkpoint = None
    else:
        checkpoint = None
    
    encode, decode = setup_tokenizer(checkpoint)
    
    print(f"\n設定: temperature={temperature}, top_k={top_k}, max_tokens={max_new_tokens}")
    print("\n対話を開始します。終了するには 'exit' と入力してください。")
    print("注意: RTX3060の制約により、長い応答や複雑な質問では品質が制限される場合があります。\n")
    
    while True:
        try:
            user_input = input("\nあなた: ")
            if user_input.lower() == 'exit':
                print("\n対話を終了します。")
                break
            
            if not user_input.strip():
                print("入力が空です。もう一度入力してください。")
                continue
            
            print("生成中...")  # ユーザーへのフィードバック
            response = generate_response(model, ctx, encode, decode, user_input)
            print(f"AI: {response}")
            
            # メモリ使用量を定期的に表示
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                if memory_allocated > 7.0:  # 7GB以上で警告
                    print(f"[警告] GPU メモリ使用量: {memory_allocated:.2f}GB")
            
        except KeyboardInterrupt:
            print("\n\n対話を終了します。")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\nGPUメモリ不足です。より短い質問を試してください。")
                torch.cuda.empty_cache()
            else:
                print(f"\nエラーが発生しました: {str(e)}")
        except Exception as e:
            print(f"\nエラーが発生しました: {str(e)}")

if __name__ == '__main__':
    main()