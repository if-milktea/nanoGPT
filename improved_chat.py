"""
改善版対話プログラム
より自然な対話のための設定調整
"""
import os
import pickle
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import tiktoken
from model import GPTConfig, GPT

# 改善された設定
system_prompt = "あなたは親切で知識豊富な日本語AIアシスタントです。"

init_from = 'resume'
out_dir = 'out_dolly_ja_improved'  # 改善版モデルを使用
temperature = 0.7  # 創造性と一貫性のバランス
top_k = 40        # 適度な制限
top_p = 0.9       # nucleus samplingを追加
max_new_tokens = 150
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())

def init_model():
    """モデルの初期化を行う"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if init_from == 'resume':
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
    
    if checkpoint and init_from == 'resume':
        meta_path = os.path.join(out_dir, 'meta.pkl')
        load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"メタデータを {meta_path} から読み込んでいます...")
    
    if not load_meta and checkpoint and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"データセット用メタデータを {meta_path} から読み込んでいます...")

    if load_meta:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        if 'stoi' in meta and 'itos' in meta:
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        elif 'tokenizer_type' in meta and meta['tokenizer_type'] == 'gpt2':
            print("GPT-2トークナイザーを使用します...")
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

def generate_response_with_nucleus(model, ctx, encode, decode, user_input):
    """Nucleus sampling (top-p) を使った応答生成"""
    formatted_prompt = f"""<|system|>
{system_prompt}

<|user|>
{user_input}

<|assistant|>
"""
    
    input_ids = encode(formatted_prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
    
    with torch.no_grad():
        with ctx:
            input_length = x.size(1)
            generated = x.clone()
            
            for _ in range(max_new_tokens):
                logits = model(generated)
                logits = logits[:, -1, :]
                
                if temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    
                    # Top-k filtering
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, -1:]] = -float('Inf')
                    
                    # Nucleus (top-p) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # 累積確率がtop_pを超える部分を除去
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = -float('Inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # 停止条件をチェック
                partial_response = decode(generated[0][input_length:].tolist())
                if any(stop_seq in partial_response for stop_seq in ["<|endoftext|>", "<|system|>", "<|user|>"]):
                    break
            
            # 応答を抽出
            output_ids = generated[0][input_length:].tolist()
            response = decode(output_ids)
            
            # 停止トークンで切断
            for stop_seq in ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>"]:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
                    break
            
            # 応答を整理
            response = response.strip()
            
            # 空の応答の場合
            if not response:
                response = "申し訳ございませんが、適切な応答を生成できませんでした。"
            
            return response

def main():
    """メインの対話ループ"""
    print("改善版チャットボットを初期化中...")
    model, ctx = init_model()
    
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
    else:
        checkpoint = None
    
    encode, decode = setup_tokenizer(checkpoint)
    
    print(f"\n現在のシステムプロンプト: 「{system_prompt}」")
    print(f"設定: temperature={temperature}, top_k={top_k}, top_p={top_p}")
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
            
            response = generate_response_with_nucleus(model, ctx, encode, decode, user_input)
            print(f"\nAI: {response}")
            
        except KeyboardInterrupt:
            print("\n\n対話を終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {str(e)}")
            continue

if __name__ == '__main__':
    main()