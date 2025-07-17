#!/usr/bin/env python3
"""
Test the actual training loop to find the bottleneck
"""

import torch
import time
from model import AtlasModel, AtlasConfig
from train import AtlasTrainer
from config import Config
from data_utils import create_dataloaders

def test_training_loop():
    """Test the actual training loop step by step"""
    print("Testing training loop step by step...")
    
    # Create minimal configuration
    config = Config()
    config.training.max_steps = 3  # Very small for testing
    config.training.batch_size = 2
    config.data.max_length = 32
    config.data.num_workers = 0
    
    # Mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000  # Smaller vocab for testing
            self.eos_token = '</s>'
            self.pad_token = '</s>'
            self.eos_token_id = 999
            
        def encode(self, text, add_special_tokens=True, return_tensors=None):
            tokens = [hash(word) % self.vocab_size for word in text.split()]
            if add_special_tokens:
                tokens = [0] + tokens + [self.eos_token_id]
            return tokens
            
        def decode(self, tokens, skip_special_tokens=True):
            return " ".join([f"token_{t}" for t in tokens if isinstance(t, int)])
    
    tokenizer = MockTokenizer()
    
    # Create small model config
    atlas_config = AtlasConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,  # Very small
        num_layers=1,
        num_heads=2,
        polynomial_degree=2,
        memory_depth=1,
        context_window_size=16,
        max_steps=3
    )
    
    print("1. Creating model...")
    start = time.time()
    model = AtlasModel(atlas_config)
    print(f"   Model created in {time.time() - start:.4f}s")
    
    print("2. Creating trainer...")
    start = time.time()
    trainer = AtlasTrainer(atlas_config, model, 'cpu')
    print(f"   Trainer created in {time.time() - start:.4f}s")
    
    print("3. Creating dataloaders...")
    start = time.time()
    dataloaders = create_dataloaders(config, tokenizer)
    train_loader = dataloaders['train']
    print(f"   Dataloaders created in {time.time() - start:.4f}s")
    print(f"   Train loader has {len(train_loader)} batches")
    
    print("4. Starting manual training loop...")
    step = 0
    max_steps = 3
    
    for epoch in range(1):  # Just one epoch
        for batch_idx, batch in enumerate(train_loader):
            if step >= max_steps:
                break
                
            print(f"   Step {step}: Processing batch {batch_idx}")
            
            start = time.time()
            metrics = trainer.train_step(batch)
            step_time = time.time() - start
            
            print(f"     - Step time: {step_time:.4f}s")
            print(f"     - Loss: {metrics['loss']:.4f}")
            
            step += 1
            
            if step >= max_steps:
                break
    
    print("✓ Manual training loop completed!")

def test_trainer_train_method():
    """Test the trainer.train method directly"""
    print("\nTesting trainer.train() method...")
    
    config = Config()
    config.training.max_steps = 3
    config.training.batch_size = 2
    config.data.max_length = 32
    config.data.num_workers = 0
    
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            self.eos_token = '</s>'
            self.pad_token = '</s>'
            self.eos_token_id = 999
            
        def encode(self, text, add_special_tokens=True, return_tensors=None):
            tokens = [hash(word) % self.vocab_size for word in text.split()]
            if add_special_tokens:
                tokens = [0] + tokens + [self.eos_token_id]
            return tokens
            
        def decode(self, tokens, skip_special_tokens=True):
            return " ".join([f"token_{t}" for t in tokens if isinstance(t, int)])
    
    tokenizer = MockTokenizer()
    
    atlas_config = AtlasConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        polynomial_degree=2,
        memory_depth=1,
        context_window_size=16,
        max_steps=3
    )
    
    model = AtlasModel(atlas_config)
    trainer = AtlasTrainer(atlas_config, model, 'cpu')
    
    dataloaders = create_dataloaders(config, tokenizer)
    
    print("   Calling trainer.train()...")
    start = time.time()
    
    # Create a timeout to prevent hanging
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Training loop timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    try:
        trainer.train(
            train_dataloader=dataloaders['train'],
            eval_dataloader=dataloaders['val'],
            save_dir='/tmp/test_checkpoints',
            eval_steps=10  # Won't trigger with only 3 steps
        )
        
        end_time = time.time()
        print(f"   trainer.train() completed in {end_time - start:.4f}s")
        print("✓ trainer.train() method successful!")
        
    except TimeoutError:
        print("✗ trainer.train() method timed out!")
        return False
    except Exception as e:
        print(f"✗ trainer.train() method failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        signal.alarm(0)  # Cancel the alarm
    
    return True

if __name__ == "__main__":
    print("Training Loop Test")
    print("=" * 40)
    
    try:
        test_training_loop()
        test_trainer_train_method()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()