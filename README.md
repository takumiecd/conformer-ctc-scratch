# Conformer-CTC 日本語音声認識

スクラッチで実装したConformer-CTCモデルによる日本語音声認識システム。

## 特徴

- **軽量設計**: Tiny (10M) / Small (30M) / Medium (50M) の3サイズ
- **純粋なPyTorch実装**: 外部ASRフレームワーク不要
- **日本語特化**: ReazonSpeechデータセット対応

## 目次

1. [環境構築](#環境構築)
2. [Hugging Face認証](#hugging-face認証)
3. [トークナイザー学習](#トークナイザー学習)
4. [モデル学習](#モデル学習)
5. [推論](#推論)
6. [モデルアーキテクチャ](#モデルアーキテクチャ)

---

## 環境構築

### 方法1: conda（ローカル環境推奨）

```bash
# 仮想環境作成
conda create -n conformer-ctc python=3.10 -y
conda activate conformer-ctc

# 依存関係インストール
pip install -r requirements.txt
```

### 方法2: pip（Jupyter/Kubernetes環境）

```bash
pip install -r requirements.txt
```

### 依存関係の確認

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'torchaudio: {torchaudio.__version__}')"
```

---

## Hugging Face認証

ReazonSpeechはgated datasetのため、Hugging Faceでの認証が必要です。

### Step 1: アクセス申請

1. [ReazonSpeech データセットページ](https://huggingface.co/datasets/reazon-research/reazonspeech) にアクセス
2. 利用規約に同意してアクセスをリクエスト

### Step 2: トークン取得

1. [Hugging Face Tokens](https://huggingface.co/settings/tokens) にアクセス
2. `New token` でトークンを作成（Read権限でOK）

### Step 3: 認証設定

#### 方法A: 環境変数（Kubernetes/Jupyter推奨）

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```

Jupyterの場合、ノートブックの最初のセルで：
```python
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxxx"
```

#### 方法B: hf CLI

```bash
pip install huggingface_hub
hf auth login
# トークンを入力
```

#### 方法C: 設定ファイル

```bash
# ~/.huggingface/token にトークンを保存
mkdir -p ~/.huggingface
echo "hf_xxxxxxxxxxxxxxxxxxxxx" > ~/.huggingface/token
```

### 認証確認

```bash
python -c "from datasets import load_dataset; ds = load_dataset('reazon-research/reazonspeech', 'small', split='train', streaming=True); print('認証OK:', next(iter(ds))['transcription'][:50])"
```

---

## トークナイザー学習

SentencePieceトークナイザーを学習します（初回のみ）。

```bash
python scripts/train_tokenizer.py --subset small --vocab_size 5000
```

### オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--subset` | small | データセットサイズ (small/medium/large) |
| `--vocab_size` | 5000 | 語彙サイズ |
| `--model_type` | unigram | トークナイザータイプ (unigram/bpe/char) |
| `--output_dir` | tokenizer | 出力ディレクトリ |
| `--max_samples` | None | 使用するサンプル数上限 |

### 出力

```
tokenizer/
├── tokenizer.model  # SentencePieceモデル
├── tokenizer.vocab  # 語彙ファイル
└── train_texts.txt  # 学習に使用したテキスト
```

---

## モデル学習

### 基本的な学習

```bash
# Tinyモデル（パイプライン検証用）
python scripts/train.py --config configs/tiny.yaml --subset small

# Smallモデル（本格学習）
python scripts/train.py --config configs/small.yaml --subset medium

# Mediumモデル（性能重視）
python scripts/train.py --config configs/medium.yaml --subset large
```

### オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--config` | 必須 | 設定ファイルパス |
| `--tokenizer` | tokenizer/tokenizer.model | トークナイザーパス |
| `--subset` | small | データセットサイズ |
| `--resume` | None | チェックポイントから再開 |
| `--num_workers` | 4 | DataLoaderワーカー数 |

### 学習の再開

```bash
python scripts/train.py --config configs/tiny.yaml --subset small --resume checkpoints/latest.pt
```

### 出力

```
checkpoints/
├── latest.pt                        # 最新チェックポイント
├── best.pt                          # ベストCER
├── checkpoint_step1000_cer15.23.pt  # Top-k チェックポイント
└── logs/                            # TensorBoardログ
```

### TensorBoardで学習監視

```bash
tensorboard --logdir checkpoints/logs
```

---

## 推論

### 単一ファイルの推論

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best.pt \
    --audio path/to/audio.wav \
    --config configs/tiny.yaml
```

### ビームサーチ（精度向上）

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best.pt \
    --audio path/to/audio.wav \
    --beam_search \
    --beam_width 10
```

### Pythonから使用

```python
import torch
from src.model import ConformerCTC
from src.data import AudioProcessor, Tokenizer
from src.utils import load_config

# モデルロード
config = load_config("configs/tiny.yaml")
tokenizer = Tokenizer("tokenizer/tokenizer.model")
model = ConformerCTC.from_config(config, vocab_size=tokenizer.get_vocab_size())

checkpoint = torch.load("checkpoints/best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 推論
audio_processor = AudioProcessor.from_config(config)
features = audio_processor.process_file("audio.wav").unsqueeze(0)

with torch.no_grad():
    log_probs, _ = model(features)
    pred_ids = log_probs.argmax(dim=-1)[0].tolist()
    text = tokenizer.decode_ctc(pred_ids)
    print(text)
```

---

## モデルアーキテクチャ

| モデル | パラメータ | encoder_dim | layers | heads |
|--------|-----------|-------------|--------|-------|
| Tiny   | ~10M      | 176         | 12     | 4     |
| Small  | ~30M      | 256         | 16     | 4     |
| Medium | ~50M      | 320         | 18     | 8     |

### Conformerブロック構成

```
Input
  ↓
Feed Forward Module (×0.5)
  ↓
Multi-Head Self-Attention (相対位置エンコーディング)
  ↓
Convolution Module
  ↓
Feed Forward Module (×0.5)
  ↓
Layer Norm
  ↓
Output
```

---

## データセット

### ReazonSpeech

| サブセット | 音声時間 | 用途 |
|-----------|---------|------|
| small | ~200時間 | 開発・テスト |
| medium | ~1,000時間 | 本格学習 |
| large | ~3,000時間 | 高精度モデル |

---

## テスト

```bash
# 全テスト実行
python -m pytest tests/ -v

# モデルテストのみ
python -m pytest tests/test_model.py -v
```

---

## トラブルシューティング

### CUDA Out of Memory

バッチサイズを小さくするか、gradient accumulationを増やす：

```yaml
# configs/tiny.yaml
training:
  batch_size: 16  # 32から16に
  accumulate_grad_batches: 2  # 1から2に
```

### Hugging Face認証エラー

```
DatasetNotFoundError: Dataset 'reazon-research/reazonspeech' is a gated dataset
```

→ [Hugging Face認証](#hugging-face認証) の手順を確認

### トークナイザーが見つからない

```
RuntimeError: Tokenizer not loaded
```

→ 先に `python scripts/train_tokenizer.py` を実行

---

## ライセンス

MIT License

## 参考文献

- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- [ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/)
