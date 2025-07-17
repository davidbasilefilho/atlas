#!/usr/bin/env python3
"""
Minimal test to isolate Atlas model issues
"""

import torch
import time
from model import AtlasModel, AtlasConfig

def test_model_forward():
    """Test basic model forward pass"""
    print("Testing Atlas model forward pass...")
    
    # Create a small config for testing
    config = AtlasConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_length=64,
        memory_depth=1,
        polynomial_degree=2,
        context_window_size=32
    )
    
    print(f"Config: {config.hidden_size}d, {config.num_layers} layers, {config.num_heads} heads")
    
    # Create model
    model = AtlasModel(config)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Test forward pass
    print("Running forward pass...")
    start_time = time.time()
    
    with torch.no_grad():
        try:
            output = model(input_ids)
            end_time = time.time()
            
            print(f"Output shape: {output.shape}")
            print(f"Forward pass time: {end_time - start_time:.4f}s")
            print("✓ Forward pass successful!")
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_attention_module():
    """Test just the attention module"""
    print("\nTesting AtlasAttention module...")
    
    from model import AtlasAttention
    
    config = AtlasConfig(
        hidden_size=128,
        num_heads=4,
        polynomial_degree=2,
        memory_depth=1,
        context_window_size=32
    )
    
    attention = AtlasAttention(config)
    attention.eval()
    
    # Test input
    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    print(f"Attention input shape: {hidden_states.shape}")
    
    start_time = time.time()
    
    with torch.no_grad():
        try:
            output = attention(hidden_states)
            end_time = time.time()
            
            print(f"Attention output shape: {output.shape}")
            print(f"Attention time: {end_time - start_time:.4f}s")
            print("✓ Attention module successful!")
            
        except Exception as e:
            print(f"✗ Attention module failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    config = AtlasConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        polynomial_degree=2,
        memory_depth=1,
        context_window_size=16
    )
    
    model = AtlasModel(config)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create test batch
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Training input shape: {input_ids.shape}")
    
    start_time = time.time()
    
    try:
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
        print("✓ Training step successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Atlas Model Testing")
    print("=" * 40)
    
    success = True
    
    # Test attention module first
    success &= test_attention_module()
    
    # Test model forward pass
    success &= test_model_forward()
    
    # Test training step
    success &= test_training_step()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")