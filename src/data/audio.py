"""Audio Processing Module"""

import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple, Union
import numpy as np


class AudioProcessor:
    """Audio processor for extracting mel spectrograms."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        normalize: bool = True,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalize = normalize
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate // 2,
        )
        
        # Global statistics for normalization (can be updated)
        self.global_mean = 0.0
        self.global_std = 1.0
        
    def load_audio(
        self,
        path: str,
        target_sr: Optional[int] = None,
    ) -> torch.Tensor:
        """Load audio file.
        
        Args:
            path: Path to audio file
            target_sr: Target sample rate (uses self.sample_rate if None)
            
        Returns:
            Waveform tensor (1, samples)
        """
        target_sr = target_sr or self.sample_rate
        waveform, sr = torchaudio.load(path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Resample if needed
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
            
        return waveform
    
    def extract_features(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Extract mel spectrogram features.
        
        Args:
            waveform: Audio waveform (1, samples) or (samples,)
            
        Returns:
            Mel spectrogram (time, n_mels)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Compute mel spectrogram
        mel = self.mel_transform(waveform)  # (1, n_mels, time)
        
        # Convert to log scale
        mel = torch.log(mel + 1e-9)
        
        # Remove channel dimension and transpose: (n_mels, time) -> (time, n_mels)
        mel = mel.squeeze(0).transpose(0, 1)
        
        # Normalize
        if self.normalize:
            mel = (mel - self.global_mean) / self.global_std
            
        return mel
    
    def process_file(self, path: str) -> torch.Tensor:
        """Load and process audio file.
        
        Args:
            path: Path to audio file
            
        Returns:
            Mel spectrogram (time, n_mels)
        """
        waveform = self.load_audio(path)
        return self.extract_features(waveform)
    
    def get_output_length(self, input_samples: int) -> int:
        """Calculate output length in frames."""
        return (input_samples - self.win_length) // self.hop_length + 1
    
    def set_stats(self, mean: float, std: float):
        """Set global normalization statistics."""
        self.global_mean = mean
        self.global_std = std
        
    @classmethod
    def from_config(cls, config: dict) -> "AudioProcessor":
        """Create processor from config dictionary."""
        audio_config = config.get("audio", config)
        return cls(
            sample_rate=audio_config.get("sample_rate", 16000),
            n_mels=audio_config.get("n_mels", 80),
            n_fft=audio_config.get("n_fft", 400),
            hop_length=audio_config.get("hop_length", 160),
            win_length=audio_config.get("win_length", 400),
        )
