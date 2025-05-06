"""
GPT言語モデルの完全な定義を1つのファイルにまとめています。
参考文献：
1) OpenAIが公開した公式GPT-2 TensorFlow実装：
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch実装：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
#テスト

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ オプションのバイアスを持つLayerNorm。PyTorchはbias=Falseを直接サポートしていません """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # すべてのヘッドのkey、query、value投影をバッチで処理
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 出力の投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 正則化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # Flash AttentionでGPUの処理が高速化されますが、PyTorch >= 2.0でのみサポート
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("警告: 低速なアテンションを使用しています。Flash AttentionにはPyTorch >= 2.0が必要です")
            # 入力シーケンスの左側のみにアテンションが適用されるようにする因果マスク
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # バッチサイズ、シーケンス長、埋め込み次元数(n_embd)

        # バッチ内のすべてのヘッドのquery、key、valueを計算し、ヘッドをバッチ次元に移動
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 因果的セルフアテンション; セルフアテンド: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Flash Attention CUDAカーネルを使用した効率的なアテンション
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # アテンションの手動実装
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # すべてのヘッドの出力を並べて再構成

        # 出力の投影
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2の語彙サイズ50257を効率化のために64の倍数に調整
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: GPT-2のようにLinearとLayerNormにバイアスあり。False: 若干性能が良く高速

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # torch.compile()使用時の重み共有で警告が生成されます：
        # "UserWarning: functional_callは紐付けられた重みに対して複数の値を受け取りました。
        # この動作は非推奨で、将来のバージョンではエラーとなります"
        # 完全には理解していませんが、今のところ無害のようです。TODO: 調査が必要
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # すべての重みを初期化
        self.apply(self._init_weights)
        # GPT-2論文に従い、残差投影に特別なスケール初期化を適用
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # パラメータ数を報告
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        モデルのパラメータ数を返します。
        非埋め込みカウント（デフォルト）では、位置埋め込みが差し引かれます。
        トークン埋め込みも同様ですが、パラメータ共有により
        これらのパラメータは最終層の重みとして実際に使用されるため、含めています。
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # 1. 入力の準備と検証: バッチサイズとシーケンス長を取得し、シーケンス長を検証
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # 2. 位置情報の生成: シーケンス内の各トークンの位置を表す配列を生成
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # 3. 埋め込み処理: トークン埋め込みと位置埋め込みを計算し結合
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # 4. Transformer層の処理: 複数のTransformerブロックを通じて特徴を抽出
        for block in self.transformer.h:
            x = block(x)

        # 5. 最終の正規化: 出力を正規化
        x = self.transformer.ln_f(x)

        # 6. 出力処理: 学習時と推論時で異なる処理を実行
        if targets is not None:
            # 学習時: すべての位置でロジットを計算し、損失を計算
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推論時: 最適化のため最後の位置のみでロジットを計算
            logits = self.lm_head(x[:, [-1], :]) # リスト[-1]を使用して時間次元を保持
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # 必要に応じてブロックサイズを縮小するモデル調整
        # 例：GPT2の事前学習モデルチェックポイント（ブロックサイズ1024）を読み込んで
        # より小さく単純なモデル用に小さいブロックサイズを使用したい場合
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # dropoutのみオーバーライド可能、詳細は以下の注記を参照
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer、n_head、n_embdはmodel_typeから決定
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 1.24億パラメータ
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 3.5億パラメータ
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 7.74億パラメータ
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 15.58億パラメータ
        }[model_type]
        print("vocab_size=50257, block_size=1024, bias=Trueを強制設定")
        config_args['vocab_size'] = 50257 # GPTモデルチェックポイントでは常に50257
        config_args['block_size'] = 1024 # GPTモデルチェックポイントでは常に1024
        config_args['bias'] = True # GPTモデルチェックポイントでは常にTrue
        # 必要に応じてドロップアウト率をオーバーライド可能
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # 新規にminGPTモデルを初期化して作成
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # このマスク/バッファは破棄（パラメータではない）

        # huggingface/transformersモデルを初期化
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # すべてのパラメータの名前と形状が一致することを確認しながらコピー
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # これらは無視（単なるバッファ）
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # 同様（単なるマスク（バッファ））
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本的にOpenAIのチェックポイントは"Conv1D"モジュールを使用しますが、通常のLinearのみを使用したい
        # そのため、これらの重みをインポート時に転置する必要があります
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 転置が必要なConv1D重みの特別な処理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # その他のパラメータは通常通りコピー
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # すべての候補パラメータから開始
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 勾配が不要なものをフィルタリング
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 最適化グループを作成。2次元のパラメータは重み減衰を適用し、それ以外は適用しない
        # つまり、行列積と埋め込みの重みテンソルには減衰を適用し、バイアスとレイヤーノームには適用しない
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # AdamWオプティマイザを作成し、利用可能な場合はfused版を使用
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ A100 bfloat16のピークFLOPS単位でモデルのFLOPS使用率（MFU）を推定 """
        # まず、イテレーションあたりのFLOPS数を推定
        # 参考：PaLM論文付録B https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # FLOPS処理能力をA100 bfloat16のピークFLOPSに対する比率として表現
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        条件付けシーケンスのインデックスidx（形状(b,t)のLongTensor）を受け取り、
        予測を毎回モデルにフィードバックしながら、シーケンスをmax_new_tokens回完成させます。
        この操作には通常model.eval()モードを使用することを推奨します。
        """
        for _ in range(max_new_tokens):
            # シーケンスコンテキストが長くなりすぎる場合、block_sizeで切り取る
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # シーケンス内のインデックスのロジットを取得するためにモデルを順伝播
            logits, _ = self(idx_cond)
            # 最終ステップのロジットを取得し、指定された温度でスケーリング
            logits = logits[:, -1, :] / temperature
            # オプションで上位k個のオプションにロジットを制限
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # ソフトマックスを適用してロジットを（正規化された）確率に変換
            probs = F.softmax(logits, dim=-1)
            # 分布からサンプリング
            idx_next = torch.multinomial(probs, num_samples=1)
            # サンプリングされたインデックスを実行中のシーケンスに追加して続行
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
