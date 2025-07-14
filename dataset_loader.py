"""
Hugging Face Datasetsライブラリを使用した外部データセットの読み込み機能

このモジュールは、以下の機能を提供します：
1. Hugging Faceからのデータセットストリーミング
2. テキストデータの前処理とトークナイゼーション
3. nanoGPTの形式に合わせたデータローダー
"""

import os
import pickle
from typing import Iterator, Optional, Dict, Any
from contextlib import nullcontext

import torch
import numpy as np
from datasets import load_dataset, IterableDataset, Dataset
import tiktoken


class HuggingFaceDataLoader:
    """
    Hugging Face Datasetsを使用したデータローダー
    ストリーミングモードをサポートし、大規模データセットにも対応
    指示応答形式のデータセットもサポート
    """
    
    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        text_column: str = "text",
        streaming: bool = True,
        cache_dir: Optional[str] = None,
        tokenizer_type: str = "gpt2",
        block_size: int = 1024,
        batch_size: int = 8,
        device: str = "cuda",
        seed: int = 1337,
        format_instruction: bool = False,
        instruction_template: Optional[str] = None
    ):
        """
        データローダーの初期化
        
        Args:
            dataset_name: Hugging Faceデータセット名（例："wikitext", "openwebtext"）
            dataset_config: データセット設定（例："wikitext-103-raw-v1"）
            text_column: テキストデータが含まれるカラム名
            streaming: ストリーミングモードを使用するか
            cache_dir: キャッシュディレクトリ
            tokenizer_type: トークナイザーの種類（"gpt2"または"custom"）
            block_size: シーケンス長
            batch_size: バッチサイズ
            device: 使用デバイス
            seed: ランダムシード
            format_instruction: 指示応答形式でフォーマットするか
            instruction_template: 指示応答のテンプレート文字列
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_column = text_column
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.tokenizer_type = tokenizer_type
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.format_instruction = format_instruction
        
        # 事前定義されたテンプレート
        predefined_templates = {
            "default": """### 指示:
{instruction}

### 入力:
{input}

### 回答:
{output}""",
            "chat": """<|system|>
あなたは親切で知識豊富なAIアシスタントです。

<|user|>
{instruction}{input}

<|assistant|>
{output}""",
            "alpaca": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
        }
        
        # テンプレートの設定
        if instruction_template in predefined_templates:
            self.instruction_template = predefined_templates[instruction_template]
        elif instruction_template:
            self.instruction_template = instruction_template
        else:
            self.instruction_template = predefined_templates["default"]
        
        # トークナイザーの初期化
        self._init_tokenizer()
        
        # データセットの読み込み
        self._load_dataset()
        
        # ランダムシードの設定
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def _init_tokenizer(self):
        """トークナイザーの初期化"""
        if self.tokenizer_type == "gpt2":
            print("GPT-2エンコーディングを使用します...")
            self.enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: self.enc.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: self.enc.decode(l)
            self.vocab_size = self.enc.n_vocab
        else:
            # カスタムトークナイザーの場合（将来の拡張用）
            raise NotImplementedError("カスタムトークナイザーは未実装です")
    
    def _load_dataset(self):
        """データセットの読み込み"""
        print(f"データセット '{self.dataset_name}' を読み込んでいます...")
        
        try:
            # mc4やcc100などのデータセットでtrust_remote_codeが必要な場合
            trust_remote_code = self.dataset_name in ["mc4", "cc100"]
            
            # データセットの読み込み
            if self.dataset_config:
                self.dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    streaming=self.streaming,
                    cache_dir=self.cache_dir,
                    trust_remote_code=trust_remote_code
                )
            else:
                self.dataset = load_dataset(
                    self.dataset_name,
                    streaming=self.streaming,
                    cache_dir=self.cache_dir,
                    trust_remote_code=trust_remote_code
                )
            
            print(f"データセットの読み込みが完了しました: {self.dataset}")
            
            # 指示応答形式の場合はデータセットを前処理
            if self.format_instruction:
                self._format_instruction_dataset()
            
        except Exception as e:
            print(f"データセットの読み込みに失敗しました: {e}")
            raise
    
    def _format_instruction_dataset(self):
        """指示応答形式でデータセットをフォーマット"""
        print("指示応答形式でデータセットをフォーマットしています...")
        
        def format_example(example):
            """個別の例を指示応答形式でフォーマット"""
            # 入力が空の場合は空文字列を使用
            input_text = example.get('input', '') or ''
            
            formatted_text = self.instruction_template.format(
                instruction=example.get('instruction', ''),
                input=input_text,
                output=example.get('output', '')
            )
            
            example[self.text_column] = formatted_text
            return example
        
        # データセットの各分割に対してフォーマットを適用
        if self.streaming:
            # ストリーミングデータセットの場合
            for split_name in self.dataset.keys():
                self.dataset[split_name] = self.dataset[split_name].map(format_example)
        else:
            # 通常のデータセットの場合
            self.dataset = self.dataset.map(format_example)
        
        print("データセットのフォーマットが完了しました")
    
    def _tokenize_text(self, text: str) -> list:
        """テキストをトークナイズ"""
        if not text or not isinstance(text, str):
            return []
        
        # テキストの前処理（必要に応じて）
        text = text.strip()
        if not text:
            return []
        
        return self.encode(text)
    
    def _process_batch(self, texts: list) -> torch.Tensor:
        """テキストのバッチを処理してトークンに変換"""
        all_tokens = []
        
        for text in texts:
            tokens = self._tokenize_text(text)
            if tokens:
                all_tokens.extend(tokens)
                # 文書間の区切りを追加
                all_tokens.append(self.enc.eot_token)  # <|endoftext|>
        
        return torch.tensor(all_tokens, dtype=torch.long)
    
    def get_batch(self, split: str = "train") -> tuple:
        """
        バッチデータを取得
        
        Args:
            split: データセットの分割（"train", "validation", "test"）
            
        Returns:
            (x, y): 入力とターゲットのテンソル
        """
        if split not in self.dataset:
            # 利用可能な分割を確認
            available_splits = list(self.dataset.keys())
            print(f"利用可能な分割: {available_splits}")
            if "train" in available_splits:
                split = "train"
            else:
                split = available_splits[0]
            print(f"分割 '{split}' を使用します")
        
        dataset_split = self.dataset[split]
        
        # ストリーミングデータセットからテキストを取得
        texts = []
        for i, example in enumerate(dataset_split):
            if i >= self.batch_size * 10:  # 十分なデータを収集
                break
            
            text = example.get(self.text_column, "")
            if text:
                texts.append(text)
        
        if not texts:
            raise ValueError(f"分割 '{split}' からテキストデータを取得できませんでした")
        
        # トークンに変換
        tokens = self._process_batch(texts)
        
        if len(tokens) < self.block_size + 1:
            # データが不足している場合は警告
            print(f"警告: データが不足しています。取得トークン数: {len(tokens)}")
            if len(tokens) < 2:
                # 最小限のダミーデータを作成
                tokens = torch.tensor([0] * (self.block_size + 1), dtype=torch.long)
        
        # バッチの作成
        max_start_idx = len(tokens) - self.block_size
        if max_start_idx <= 0:
            # トークンが不足している場合はパディング
            padding_length = self.block_size + 1 - len(tokens)
            tokens = torch.cat([tokens, torch.zeros(padding_length, dtype=torch.long)])
            max_start_idx = 1
        
        # ランダムな開始位置を選択
        start_indices = torch.randint(0, max_start_idx, (self.batch_size,))
        
        x = torch.stack([tokens[i:i+self.block_size] for i in start_indices])
        y = torch.stack([tokens[i+1:i+1+self.block_size] for i in start_indices])
        
        # デバイスに転送
        if self.device != "cpu":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        
        return x, y
    
    def create_iterator(self, split: str = "train", num_samples: Optional[int] = None):
        """
        データセットのイテレータを作成
        
        Args:
            split: データセットの分割
            num_samples: 取得するサンプル数（Noneの場合は無制限）
            
        Yields:
            (x, y): 入力とターゲットのテンソルペア
        """
        count = 0
        while num_samples is None or count < num_samples:
            try:
                x, y = self.get_batch(split)
                yield x, y
                count += 1
            except Exception as e:
                print(f"バッチ取得エラー: {e}")
                break
    
    def get_vocab_size(self) -> int:
        """語彙サイズを取得"""
        return self.vocab_size
    
    def save_tokenizer_meta(self, save_path: str):
        """トークナイザーのメタ情報を保存"""
        meta = {
            'vocab_size': self.vocab_size,
            'tokenizer_type': self.tokenizer_type,
            'dataset_name': self.dataset_name,
            'dataset_config': self.dataset_config
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"トークナイザーメタ情報を保存しました: {save_path}")


def create_hf_dataloader(
    dataset_name: str,
    config: Dict[str, Any],
    device: str = "cuda"
) -> HuggingFaceDataLoader:
    """
    設定辞書からHuggingFaceDataLoaderを作成
    
    Args:
        dataset_name: データセット名
        config: 設定辞書
        device: デバイス
        
    Returns:
        HuggingFaceDataLoader: データローダーインスタンス
    """
    return HuggingFaceDataLoader(
        dataset_name=dataset_name,
        dataset_config=config.get('dataset_config'),
        text_column=config.get('text_column', 'text'),
        streaming=config.get('streaming', True),
        cache_dir=config.get('cache_dir'),
        tokenizer_type=config.get('tokenizer_type', 'gpt2'),
        block_size=config.get('block_size', 1024),
        batch_size=config.get('batch_size', 8),
        device=device,
        seed=config.get('seed', 1337),
        format_instruction=config.get('format_instruction', False),
        instruction_template=config.get('instruction_template')
    )


class MultiDatasetLoader:
    """
    複数のデータセットを混合して使用するデータローダー
    各データセットから指定された比率でサンプリング
    """
    
    def __init__(
        self,
        dataset_configs: list,
        mix_ratios: list = None,
        device: str = "cuda",
        seed: int = 1337
    ):
        """
        複数データセットの初期化
        
        Args:
            dataset_configs: 各データセットの設定辞書のリスト
            mix_ratios: 各データセットの混合比率（Noneの場合は均等）
            device: 使用デバイス
            seed: ランダムシード
        """
        self.dataset_configs = dataset_configs
        self.mix_ratios = mix_ratios or [1.0] * len(dataset_configs)
        self.device = device
        self.seed = seed
        
        # 比率を正規化
        total_ratio = sum(self.mix_ratios)
        self.mix_ratios = [r / total_ratio for r in self.mix_ratios]
        
        # 各データセットローダーを初期化
        self.loaders = []
        for i, config in enumerate(dataset_configs):
            print(f"データセット {i+1}/{len(dataset_configs)} を初期化中: {config['dataset_name']}")
            # configをコピーしてseed、deviceを個別に設定
            config_copy = config.copy()
            config_copy['device'] = device
            config_copy['seed'] = seed + i
            loader = HuggingFaceDataLoader(**config_copy)
            self.loaders.append(loader)
        
        # 統一されたvocab_sizeを使用（最初のローダーから）
        self.vocab_size = self.loaders[0].get_vocab_size()
        
        print(f"混合データセットの初期化完了 - {len(self.loaders)}個のデータセット")
        print(f"混合比率: {[f'{r:.2f}' for r in self.mix_ratios]}")
    
    def get_batch(self, split):
        """
        混合データセットからバッチを取得
        どのデータセットからサンプリングしているかの統計を保持
        """
        import random
        
        # 混合比率に基づいてデータセットを選択
        dataset_idx = random.choices(range(len(self.loaders)), weights=self.mix_ratios)[0]
        
        # 統計記録（オプション）
        if not hasattr(self, 'batch_count'):
            self.batch_count = 0
            self.dataset_usage = [0] * len(self.loaders)
        
        self.batch_count += 1
        self.dataset_usage[dataset_idx] += 1
        
        # 1000回に1回使用統計を表示
        if self.batch_count % 1000 == 0:
            print(f"\n📊 データセット使用統計 (過去1000バッチ):")
            for i, usage in enumerate(self.dataset_usage):
                percentage = (usage / self.batch_count) * 100
                expected = self.mix_ratios[i] * 100
                dataset_name = self.dataset_configs[i]['dataset_name']
                print(f"  {dataset_name}: {percentage:.1f}% (期待値: {expected:.1f}%)")
            print("")
        
        # 選択されたデータセットからバッチを取得
        return self.loaders[dataset_idx].get_batch(split)
    
    def get_vocab_size(self):
        """ボキャブラリサイズを返す"""
        return self.vocab_size
    
    def save_tokenizer_meta(self, path):
        """トークナイザーメタ情報を保存（最初のローダーから）"""
        return self.loaders[0].save_tokenizer_meta(path)


def create_multi_hf_dataloader(dataset_configs, mix_ratios=None, device="cuda"):
    """
    複数のHugging Faceデータセットを混合したデータローダーを作成
    
    Args:
        dataset_configs: データセット設定のリスト
        mix_ratios: 混合比率のリスト
        device: 使用デバイス
    
    Returns:
        MultiDatasetLoader: 複数データセット対応ローダー
    """
    return MultiDatasetLoader(dataset_configs, mix_ratios, device)


# よく使用されるデータセットの設定例
DATASET_CONFIGS = {
    "wikitext": {
        "dataset_config": "wikitext-103-raw-v1",
        "text_column": "text",
        "streaming": True
    },
    "openwebtext": {
        "dataset_config": None,
        "text_column": "text", 
        "streaming": True
    },
    "bookcorpus": {
        "dataset_config": None,
        "text_column": "text",
        "streaming": True
    },
    "c4": {
        "dataset_config": "en",
        "text_column": "text",
        "streaming": True
    },
    "pile": {
        "dataset_config": None,
        "text_column": "text",
        "streaming": True
    },
    "kunishou/databricks-dolly-15k-ja": {
        "dataset_config": None,
        "text_column": "formatted_text",
        "streaming": False,
        "format_instruction": True
    },
    "ce-lery/mistral-3b-dataset": {
        "dataset_config": None,
        "text_column": "text",
        "streaming": True,
        "format_instruction": False
    },
    "fujiki/wiki40b_ja": {
        "dataset_config": "default",  # wiki40b_jaの正しいconfig
        "text_column": "text", 
        "streaming": True,
        "format_instruction": False
    },
    # 大規模日本語データセット
    "mc4": {
        "dataset_config": "ja", 
        "text_column": "text",
        "streaming": True,
        "format_instruction": False
    },
    "cc100": {
        "dataset_config": "ja",
        "text_column": "text", 
        "streaming": True,
        "format_instruction": False
    }
}


if __name__ == "__main__":
    # テスト用のコード
    print("HuggingFace DataLoaderのテストを実行します...")
    
    # WikiTextデータセットでテスト
    try:
        config = DATASET_CONFIGS["wikitext"].copy()
        config.update({
            "block_size": 256,
            "batch_size": 4
        })
        
        dataloader = create_hf_dataloader("wikitext", config)
        
        # バッチを取得してテスト
        x, y = dataloader.get_batch("train")
        print(f"バッチ形状: x={x.shape}, y={y.shape}")
        print(f"語彙サイズ: {dataloader.get_vocab_size()}")
        
        # トークナイザーのテスト
        sample_text = "Hello, this is a test."
        tokens = dataloader.encode(sample_text)
        decoded = dataloader.decode(tokens)
        print(f"エンコード/デコードテスト: '{sample_text}' -> {tokens} -> '{decoded}'")
        
        print("テストが正常に完了しました！")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
