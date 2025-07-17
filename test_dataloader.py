#!/usr/bin/env python3
"""
Test data loading to isolate issues
"""

import torch
import time
from config import Config
from data_utils import create_dataloaders

def test_mock_tokenizer():
    """Test the mock tokenizer implementation"""
    print("Testing mock tokenizer...")
    
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 50257
            self.eos_token = '</s>'
            self.pad_token = '</s>'
            self.eos_token_id = 50256
            
        def encode(self, text, add_special_tokens=True, return_tensors=None):
            # Simple word-based tokenization for testing
            tokens = [hash(word) % self.vocab_size for word in text.split()]
            if add_special_tokens:
                tokens = [0] + tokens + [self.eos_token_id]
            if return_tensors == 'pt':
                return torch.tensor([tokens])
            return tokens
            
        def decode(self, tokens, skip_special_tokens=True):
            # Simple mock decode
            return " ".join([f"token_{t}" for t in tokens if isinstance(t, int)])
    
    tokenizer = MockTokenizer()
    
    # Test tokenization
    test_text = "This is a test sentence."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Test text: {test_text}")
    print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
    print(f"Decoded: {decoded[:50]}...")  # Show first 50 chars
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    return tokenizer

def test_dataloader_creation():
    """Test dataloader creation"""
    print("\nTesting dataloader creation...")
    
    tokenizer = test_mock_tokenizer()
    config = Config()
    
    # Override config for small test
    config.data.max_length = 64
    config.training.batch_size = 2
    config.data.num_workers = 0  # No multiprocessing for debugging
    
    start_time = time.time()
    
    try:
        dataloaders = create_dataloaders(config, tokenizer)
        end_time = time.time()
        
        print(f"Dataloader creation time: {end_time - start_time:.4f}s")
        print(f"Train batches: {len(dataloaders['train'])}")
        print(f"Val batches: {len(dataloaders['val'])}")
        print(f"Recall batches: {len(dataloaders['recall'])}")
        
        return dataloaders
        
    except Exception as e:
        print(f"Dataloader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dataloader_iteration():
    """Test iterating through dataloader"""
    print("\nTesting dataloader iteration...")
    
    tokenizer = test_mock_tokenizer()
    config = Config()
    config.data.max_length = 32
    config.training.batch_size = 2
    config.data.num_workers = 0
    
    dataloaders = create_dataloaders(config, tokenizer)
    if dataloaders is None:
        return False
    
    train_loader = dataloaders['train']
    
    print("Iterating through first 3 batches...")
    start_time = time.time()
    
    try:
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Only test first 3 batches
                break
                
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            print(f"Batch {i}: input_ids shape = {input_ids.shape}, labels shape = {labels.shape}")
            
            if i == 0:
                print(f"Sample input_ids: {input_ids[0][:10]}...")
                print(f"Sample labels: {labels[0][:10]}...")
        
        end_time = time.time()
        print(f"Iteration time: {end_time - start_time:.4f}s")
        print("✓ Dataloader iteration successful!")
        return True
        
    except Exception as e:
        print(f"Dataloader iteration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step_with_dataloader():
    """Test a complete training step with dataloader"""
    print("\nTesting training step with real dataloader...")
    
    from model import AtlasModel, AtlasConfig
    
    tokenizer = test_mock_tokenizer()
    config = Config()
    config.data.max_length = 32
    config.training.batch_size = 2
    config.data.num_workers = 0
    
    # Create small model
    atlas_config = AtlasConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        polynomial_degree=2,
        memory_depth=1,
        context_window_size=16
    )
    
    model = AtlasModel(atlas_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    dataloaders = create_dataloaders(config, tokenizer)
    if dataloaders is None:
        return False
    
    train_loader = dataloaders['train']
    
    print("Running one training step...")
    start_time = time.time()
    
    try:
        batch = next(iter(train_loader))
        
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        print(f"Batch shapes: input_ids={input_ids.shape}, labels={labels.shape}")
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = torch.nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end_time = time.time()
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Training step time: {end_time - start_time:.4f}s")
        print("✓ Training step with dataloader successful!")
        
        return True
        
    except Exception as e:
        print(f"Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Data Loading Test")
    print("=" * 40)
    
    success = True
    
    # Test dataloader creation
    success &= (test_dataloader_creation() is not None)
    
    # Test dataloader iteration
    success &= test_dataloader_iteration()
    
    # Test training step
    success &= test_training_step_with_dataloader()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All dataloader tests passed!")
    else:
        print("✗ Some dataloader tests failed!")