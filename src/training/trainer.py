"""Trainer for Conformer-CTC Model"""

import os
import time
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .optimizer import get_optimizer, get_scheduler
from .loss import CTCLoss
from ..utils.metrics import compute_cer


class Trainer:
    """Trainer for Conformer-CTC model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        tokenizer = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.tokenizer = tokenizer
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = self.model.to(self.device)
        
        # Training config
        train_config = self.config.get("training", {})
        self.max_epochs = train_config.get("max_epochs", 100)
        self.gradient_clip = train_config.get("gradient_clip", 5.0)
        self.accumulate_grad_batches = train_config.get("accumulate_grad_batches", 1)
        self.log_interval = train_config.get("log_interval", 100)
        self.eval_interval = train_config.get("eval_interval", 1000)
        
        # Optimizer and scheduler
        lr = train_config.get("learning_rate", 0.001)
        warmup_steps = train_config.get("warmup_steps", 10000)
        
        self.optimizer = get_optimizer(model, learning_rate=lr)
        self.scheduler = get_scheduler(
            self.optimizer, 
            warmup_steps=warmup_steps, 
            peak_lr=lr
        )
        
        # Loss
        self.criterion = CTCLoss()
        
        # Checkpoint
        checkpoint_config = self.config.get("checkpoint", {})
        self.save_dir = checkpoint_config.get("save_dir", "checkpoints")
        self.save_top_k = checkpoint_config.get("save_top_k", 3)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Logging
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "logs"))
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_cer = float("inf")
        self.best_checkpoints = []  # List of (cer, path)
        
    def train(self):
        """Run training loop."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            self._train_epoch()
            
            if self.val_loader is not None:
                val_cer = self._validate()
                self._save_checkpoint(val_cer)
            else:
                self._save_checkpoint(None)
                
        self.writer.close()
        print("Training complete!")
        
    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            loss = self._train_step(batch)
            
            # Gradient accumulation
            loss = loss / self.accumulate_grad_batches
            loss.backward()
            
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip
                    )
                    
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
            epoch_loss += loss.item() * self.accumulate_grad_batches
            num_batches += 1
            
            # Logging
            if self.global_step % self.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                self.writer.add_scalar("train/lr", lr, self.global_step)
                
            # Mid-epoch validation
            if self.val_loader and self.global_step % self.eval_interval == 0:
                val_cer = self._validate()
                self._save_checkpoint(val_cer)
                self.model.train()
                
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step."""
        features = batch["features"].to(self.device)
        labels = batch["labels"].to(self.device)
        input_lengths = batch["input_lengths"].to(self.device)
        label_lengths = batch["label_lengths"].to(self.device)
        mask = batch["mask"].to(self.device)
        
        # Forward
        log_probs, output_lengths = self.model(features, mask)
        
        # Loss
        loss = self.criterion(log_probs, labels, output_lengths, label_lengths)
        
        return loss
    
    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_refs = []
        all_hyps = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)
            label_lengths = batch["label_lengths"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            # Forward
            log_probs, output_lengths = self.model(features, mask)
            
            # Loss
            loss = self.criterion(log_probs, labels, output_lengths, label_lengths)
            total_loss += loss.item()
            num_batches += 1
            
            # Decode for CER calculation
            if self.tokenizer is not None:
                predictions = log_probs.argmax(dim=-1)  # (batch, time)
                
                for i in range(predictions.size(0)):
                    pred_ids = predictions[i, :output_lengths[i]].cpu().tolist()
                    label_ids = labels[i, :label_lengths[i]].cpu().tolist()
                    
                    hyp = self.tokenizer.decode_ctc(pred_ids)
                    ref = self.tokenizer.decode(label_ids)
                    
                    all_hyps.append(hyp)
                    all_refs.append(ref)
                    
        avg_loss = total_loss / max(num_batches, 1)
        
        # Calculate CER
        if all_refs and all_hyps:
            cer_result = compute_cer(all_refs, all_hyps)
            cer = cer_result["cer"]
        else:
            cer = 0.0
            
        print(f"\nValidation - Loss: {avg_loss:.4f}, CER: {cer:.2f}%")
        self.writer.add_scalar("val/loss", avg_loss, self.global_step)
        self.writer.add_scalar("val/cer", cer, self.global_step)
        
        return cer
    
    def _save_checkpoint(self, val_cer: Optional[float]):
        """Save checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "config": self.config,
            "val_cer": val_cer,
        }
        
        # Save latest
        latest_path = os.path.join(self.save_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best
        if val_cer is not None and val_cer < self.best_val_cer:
            self.best_val_cer = val_cer
            best_path = os.path.join(self.save_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"New best CER: {val_cer:.2f}%")
            
        # Save top-k checkpoints
        if val_cer is not None:
            ckpt_path = os.path.join(
                self.save_dir, 
                f"checkpoint_step{self.global_step}_cer{val_cer:.2f}.pt"
            )
            torch.save(checkpoint, ckpt_path)
            
            self.best_checkpoints.append((val_cer, ckpt_path))
            self.best_checkpoints.sort(key=lambda x: x[0])
            
            # Remove old checkpoints
            while len(self.best_checkpoints) > self.save_top_k:
                _, old_path = self.best_checkpoints.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)
                    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["epoch"]
        
        if "val_cer" in checkpoint and checkpoint["val_cer"] is not None:
            self.best_val_cer = checkpoint["val_cer"]
            
        print(f"Loaded checkpoint from step {self.global_step}")
