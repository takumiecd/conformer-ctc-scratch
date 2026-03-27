"""Evaluation Metrics for Speech Recognition"""

from typing import List, Tuple
import re


def _edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Calculate edit distance between reference and hypothesis.
    
    Returns:
        Tuple of (substitutions, insertions, deletions, reference_length)
    """
    m, n = len(ref), len(hyp)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # substitution
                    dp[i - 1][j] + 1,       # deletion
                    dp[i][j - 1] + 1,       # insertion
                )
                
    # Backtrack to get S, I, D
    i, j = m, n
    S, I, D = 0, 0, 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1
            j -= 1
        else:
            D += 1
            i -= 1
            
    return S, I, D, m


def compute_wer(
    references: List[str],
    hypotheses: List[str],
) -> dict:
    """Compute Word Error Rate (WER).
    
    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions
        
    Returns:
        Dictionary with WER, substitutions, insertions, deletions, reference words
    """
    total_S, total_I, total_D, total_N = 0, 0, 0, 0
    
    for ref, hyp in zip(references, hypotheses):
        # Tokenize by whitespace (for Japanese, use character-level or MeCab)
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        
        S, I, D, N = _edit_distance(ref_words, hyp_words)
        total_S += S
        total_I += I
        total_D += D
        total_N += N
        
    wer = (total_S + total_I + total_D) / max(total_N, 1)
    
    return {
        "wer": wer * 100,
        "substitutions": total_S,
        "insertions": total_I,
        "deletions": total_D,
        "reference_words": total_N,
    }


def compute_cer(
    references: List[str],
    hypotheses: List[str],
) -> dict:
    """Compute Character Error Rate (CER).
    
    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions
        
    Returns:
        Dictionary with CER, substitutions, insertions, deletions, reference chars
    """
    total_S, total_I, total_D, total_N = 0, 0, 0, 0
    
    for ref, hyp in zip(references, hypotheses):
        # Remove spaces for character-level comparison
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        
        S, I, D, N = _edit_distance(ref_chars, hyp_chars)
        total_S += S
        total_I += I
        total_D += D
        total_N += N
        
    cer = (total_S + total_I + total_D) / max(total_N, 1)
    
    return {
        "cer": cer * 100,
        "substitutions": total_S,
        "insertions": total_I,
        "deletions": total_D,
        "reference_chars": total_N,
    }


def normalize_text(text: str) -> str:
    """Normalize text for evaluation.
    
    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    text = text.lower()
    # Keep Japanese characters, alphanumeric, and spaces
    text = re.sub(r'[^\w\s\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
