# Conformer-CTC 日本語音声認識

スクラッチで実装したConformer-CTCモデルによる日本語音声認識システム。

## 特徴

- **軽量設計**: Tiny (10M) / Small (30M) / Medium (50M) の3サイズ
- **純粋なPyTorch実装**: 外部ASRフレームワーク不要
- **日本語特化**: ReazonSpeechデータセット対応

## インストール

```bash
pip install -r requirements.txt
```

## クイックスタート

### 1. データ準備

```bash
python scripts/prepare_data.py --dataset reazon
```

### 2. トークナイザー学習

```bash
python scripts/train_tokenizer.py --vocab_size 5000
```

### 3. 学習

```bash
python scripts/train.py --config configs/tiny.yaml
```

### 4. 推論

```bash
python scripts/inference.py --checkpoint checkpoints/best.pt --audio test.wav
```

## モデルアーキテクチャ

| モデル | パラメータ | encoder_dim | layers |
|--------|-----------|-------------|--------|
| Tiny   | ~10M      | 176         | 12     |
| Small  | ~30M      | 256         | 16     |
| Medium | ~50M      | 320         | 18     |

## ライセンス

MIT License

## 参考文献

- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- [ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/)
