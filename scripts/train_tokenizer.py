#!/usr/bin/env python3
"""Train SentencePiece tokenizer from ReazonSpeech dataset."""

import argparse
import os
from datasets import load_dataset
from tqdm import tqdm


def extract_texts(
    output_file: str,
    subset: str = "small",
    max_samples: int = None,
):
    """Extract transcriptions from ReazonSpeech dataset."""
    print(f"Loading ReazonSpeech ({subset}) in streaming mode...")
    dataset = load_dataset(
        "reazon-research/reazonspeech",
        subset,
        split="train",
        streaming=True,  # Stream data to avoid loading audio
    )
    
    print(f"Extracting texts to {output_file}...")
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(tqdm(dataset, desc="Processing")):
            if max_samples and i >= max_samples:
                break
            text = sample["transcription"]
            if text.strip():
                f.write(text.strip() + "\n")
                count += 1
                
    print(f"Extracted {count} transcriptions")


def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 5000,
    model_type: str = "unigram",
):
    """Train SentencePiece tokenizer."""
    import sentencepiece as spm
    
    print(f"Training {model_type} tokenizer with vocab_size={vocab_size}...")
    
    # Reserve 1 slot for blank token (we'll add it after training)
    actual_vocab_size = vocab_size - 1
    
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=actual_vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        pad_id=-1,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        user_defined_symbols=[],
        max_sentence_length=4096,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count(),
    )
    
    print(f"Tokenizer saved to {model_prefix}.model")


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument(
        "--vocab_size", type=int, default=5000,
        help="Vocabulary size (including blank token)"
    )
    parser.add_argument(
        "--model_type", type=str, default="unigram",
        choices=["unigram", "bpe", "char"],
        help="SentencePiece model type"
    )
    parser.add_argument(
        "--output_dir", type=str, default="tokenizer",
        help="Output directory for tokenizer"
    )
    parser.add_argument(
        "--subset", type=str, default="small",
        choices=["small", "medium", "large", "all"],
        help="ReazonSpeech subset to use"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to use for training"
    )
    parser.add_argument(
        "--text_file", type=str, default=None,
        help="Use existing text file instead of extracting from dataset"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Text file for training
    text_file = args.text_file or os.path.join(args.output_dir, "train_texts.txt")
    
    # Extract texts if not provided
    if not args.text_file or not os.path.exists(text_file):
        extract_texts(text_file, args.subset, args.max_samples)
        
    # Train tokenizer
    model_prefix = os.path.join(args.output_dir, "tokenizer")
    train_tokenizer(
        input_file=text_file,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )
    
    # Test tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    
    test_text = "これはテストです"
    encoded = sp.Encode(test_text)
    decoded = sp.Decode(encoded)
    
    print(f"\nTest: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Vocab size: {sp.GetPieceSize()} (+ 1 blank = {sp.GetPieceSize() + 1})")


if __name__ == "__main__":
    main()
