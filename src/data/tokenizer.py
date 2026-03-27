"""SentencePiece Tokenizer Wrapper"""

import os
from typing import List, Optional, Union
import sentencepiece as spm


class Tokenizer:
    """SentencePiece tokenizer wrapper for CTC.
    
    Vocabulary structure:
        0: <blank> (CTC blank token)
        1: <unk>
        2: <s> (BOS - optional)
        3: </s> (EOS - optional)
        4+: actual tokens
    """
    
    BLANK_TOKEN = "<blank>"
    BLANK_ID = 0
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 5000,
    ):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            
    def load(self, model_path: str):
        """Load trained SentencePiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path
        # vocab_size includes blank token
        self.vocab_size = self.sp.GetPieceSize() + 1
        
    def train(
        self,
        input_file: str,
        model_prefix: str,
        vocab_size: int = 5000,
        model_type: str = "unigram",
        character_coverage: float = 0.9995,
        **kwargs,
    ):
        """Train SentencePiece model.
        
        Args:
            input_file: Path to text file with training data
            model_prefix: Output model prefix
            vocab_size: Target vocabulary size (excluding blank)
            model_type: Model type (unigram, bpe, char, word)
            character_coverage: Character coverage for training
        """
        # Reserve 1 slot for blank token
        actual_vocab_size = vocab_size - 1
        
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=actual_vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=-1,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            **kwargs,
        )
        
        self.load(f"{model_prefix}.model")
        self.vocab_size = vocab_size
        
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Add BOS token
            add_eos: Add EOS token
            
        Returns:
            List of token IDs (shifted by 1 to accommodate blank at 0)
        """
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded. Call load() or train() first.")
            
        ids = self.sp.Encode(text)
        
        # Shift all IDs by 1 (blank token at 0)
        ids = [i + 1 for i in ids]
        
        if add_bos:
            ids = [self.sp.bos_id() + 1] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id() + 1]
            
        return ids
    
    def decode(self, ids: List[int], remove_blank: bool = True) -> str:
        """Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            remove_blank: Remove blank tokens
            
        Returns:
            Decoded text
        """
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded. Call load() or train() first.")
            
        if remove_blank:
            ids = [i for i in ids if i != self.BLANK_ID]
            
        # Shift IDs back (remove blank offset)
        ids = [i - 1 for i in ids if i > 0]
        
        return self.sp.Decode(ids)
    
    def decode_ctc(self, ids: List[int]) -> str:
        """Decode CTC output (collapse repeated tokens, remove blanks).
        
        Args:
            ids: List of token IDs from CTC output
            
        Returns:
            Decoded text
        """
        # Remove consecutive duplicates
        collapsed = []
        prev_id = None
        for i in ids:
            if i != prev_id:
                collapsed.append(i)
                prev_id = i
                
        return self.decode(collapsed, remove_blank=True)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size including blank token."""
        return self.vocab_size
    
    def id_to_token(self, id: int) -> str:
        """Convert token ID to token string."""
        if id == self.BLANK_ID:
            return self.BLANK_TOKEN
        if self.sp is None:
            return f"<id_{id}>"
        return self.sp.IdToPiece(id - 1)
    
    def token_to_id(self, token: str) -> int:
        """Convert token string to token ID."""
        if token == self.BLANK_TOKEN:
            return self.BLANK_ID
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded.")
        return self.sp.PieceToId(token) + 1
