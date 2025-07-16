"""
Atlas Training Script
Implementation of training loop for Atlas model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import wandb
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, List, Optional
import json
import math

from model import AtlasModel, AtlasConfig


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            # Tokenize and create sliding window examples
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            # Create overlapping sequences
            for i in range(0, len(tokens) - max_length + 1, max_length // 2):
                sequence = tokens[i:i + max_length]
                if len(sequence) == max_length:
                    self.examples.append(sequence)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        sequence = self.examples[idx]
        return {
            'input_ids': torch.tensor(sequence[:-1], dtype=torch.long),
            'labels': torch.tensor(sequence[1:], dtype=torch.long)
        }


class AtlasTrainer:
    """Training manager for Atlas model"""
    
    def __init__(self, config: AtlasConfig, model: AtlasModel, device: str = 'cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Initialize wandb
        wandb.init(
            project="atlas-training",
            config=config.__dict__,
            name=f"atlas-{config.hidden_size}-{config.num_layers}layers"
        )
    
    def warmup_lr_schedule(self, step: int) -> float:
        """Warmup learning rate schedule"""
        if step < self.config.warmup_steps:
            return float(step) / float(max(1, self.config.warmup_steps))
        return 1.0
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for language modeling"""
        # Flatten for loss computation
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        # Compute cross-entropy loss
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        loss = self.compute_loss(logits, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update learning rate
        if self.step < self.config.warmup_steps:
            lr_scale = self.warmup_lr_schedule(self.step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale
        else:
            self.scheduler.step()
        
        # Calculate perplexity
        perplexity = torch.exp(loss).item()
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'perplexity': perplexity,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids)
                loss = self.compute_loss(logits, labels)
                
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
        
        avg_loss = total_loss / total_samples
        perplexity = math.exp(avg_loss)
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Checkpoint loaded from {filepath}")
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None,
              save_dir: str = "checkpoints", eval_steps: int = 1000):
        """Main training loop"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {self.config.max_steps} steps...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        running_loss = 0.0
        log_steps = 100
        
        while self.step < self.config.max_steps:
            self.epoch += 1
            
            for batch in train_dataloader:
                if self.step >= self.config.max_steps:
                    break
                
                # Training step
                metrics = self.train_step(batch)
                running_loss += metrics['loss']
                
                # Logging
                if self.step % log_steps == 0:
                    avg_loss = running_loss / log_steps
                    
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/perplexity': math.exp(avg_loss),
                        'train/learning_rate': metrics['learning_rate'],
                        'step': self.step
                    })
                    
                    print(f"Step {self.step}: Loss = {avg_loss:.4f}, "
                          f"PPL = {math.exp(avg_loss):.2f}, "
                          f"LR = {metrics['learning_rate']:.2e}")
                    
                    running_loss = 0.0
                
                # Evaluation
                if eval_dataloader and self.step % eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    wandb.log({
                        'eval/loss': eval_metrics['eval_loss'],
                        'eval/perplexity': eval_metrics['eval_perplexity'],
                        'step': self.step
                    })
                    
                    print(f"Eval - Loss: {eval_metrics['eval_loss']:.4f}, "
                          f"PPL: {eval_metrics['eval_perplexity']:.2f}")
                    
                    # Save best model
                    if eval_metrics['eval_loss'] < self.best_loss:
                        self.best_loss = eval_metrics['eval_loss']
                        self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'))
                
                # Regular checkpoint saving
                if self.step % 5000 == 0:
                    self.save_checkpoint(os.path.join(save_dir, f'checkpoint_step_{self.step}.pt'))
        
        print("Training completed!")
        
        # Final save
        self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'))


def create_sample_dataset(tokenizer, num_samples: int = 1000, max_length: int = 512) -> List[str]:
    """Create a sample dataset for demonstration"""
    # This is a simple example - in practice, you'd load real text data
    texts = []
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "Once upon a time, in a distant galaxy, there lived a wise old wizard. " * 15,
        "Machine learning is a subset of artificial intelligence that focuses on algorithms. " * 12,
        "The Atlas model introduces a novel approach to long-term memory in transformers. " * 18,
        "Context memorization is crucial for understanding long sequences in language models. " * 16,
    ]
    
    for i in range(num_samples):
        text = sample_texts[i % len(sample_texts)]
        # Add some variation
        text += f" This is sample number {i}. " * 5
        texts.append(text)
    
    return texts


def main():
    """Main training function"""
    
    # Configuration
    config = AtlasConfig(
        vocab_size=50257,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        max_seq_length=1024,
        dropout=0.1,
        memory_depth=2,
        memory_hidden_size=256,
        polynomial_degree=3,
        context_window_size=256,
        batch_size=4,
        learning_rate=1e-4,
        max_steps=10000,
        warmup_steps=500
    )
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("Creating datasets...")
    sample_texts = create_sample_dataset(tokenizer, num_samples=800)
    
    # Split into train/eval
    split_idx = int(0.9 * len(sample_texts))
    train_texts = sample_texts[:split_idx]
    eval_texts = sample_texts[split_idx:]
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=config.max_seq_length // 2)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=config.max_seq_length // 2)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Eval dataset: {len(eval_dataset)} samples")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AtlasModel(config)
    
    # Initialize trainer
    trainer = AtlasTrainer(config, model, device)
    
    # Start training
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        save_dir="checkpoints",
        eval_steps=500
    )


if __name__ == "__main__":
    main()
