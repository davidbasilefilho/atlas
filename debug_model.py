#!/usr/bin/env python3
"""
Debug the exact location where training hangs
"""

import torch
import time
import sys
from model import AtlasModel, AtlasConfig

def debug_model_forward():
    """Debug model forward pass step by step"""
    print("Debugging model forward pass...")
    
    config = AtlasConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        polynomial_degree=1,
        memory_depth=1,
        context_window_size=16,
        use_muon_optimizer=False
    )
    
    model = AtlasModel(config)
    
    # Create test input
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Debug embeddings
    print("1. Testing embeddings...")
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    token_embeds = model.token_embedding(input_ids)
    pos_embeds = model.position_embedding(position_ids)
    hidden_states = token_embeds + pos_embeds
    print(f"   Embeddings shape: {hidden_states.shape}")
    
    # Debug transformer blocks
    print("2. Testing transformer blocks...")
    for i, block in enumerate(model.blocks):
        print(f"   Block {i}...")
        start = time.time()
        
        # Debug layer norm
        ln1_out = block.ln_1(hidden_states)
        print(f"     LN1 time: {time.time() - start:.4f}s")
        
        # Debug attention - this is likely where it hangs
        print(f"     Testing attention...")
        start = time.time()
        
        try:
            attn_out = block.attn(ln1_out)
            print(f"     Attention time: {time.time() - start:.4f}s")
        except Exception as e:
            print(f"     Attention failed: {e}")
            return False
        
        # Add residual
        hidden_states = hidden_states + attn_out
        
        # Debug MLP
        start = time.time()
        ln2_out = block.ln_2(hidden_states)
        mlp_out = block.mlp(ln2_out)
        hidden_states = hidden_states + mlp_out
        print(f"     MLP time: {time.time() - start:.4f}s")
    
    # Debug final layers
    print("3. Testing final layers...")
    start = time.time()
    hidden_states = model.ln_f(hidden_states)
    logits = model.lm_head(hidden_states)
    print(f"   Final layers time: {time.time() - start:.4f}s")
    print(f"   Output shape: {logits.shape}")
    
    print("✓ Model forward debugging completed!")
    return True

def debug_attention_module():
    """Debug the attention module specifically"""
    print("\nDebugging AtlasAttention module...")
    
    from model import AtlasAttention
    
    config = AtlasConfig(
        hidden_size=64,
        num_heads=2,
        polynomial_degree=1,  # Minimal
        memory_depth=1,
        context_window_size=8,
        use_muon_optimizer=False
    )
    
    attention = AtlasAttention(config)
    
    batch_size, seq_len = 2, 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    print(f"Input shape: {hidden_states.shape}")
    
    # Step by step debugging
    print("1. Linear projections...")
    start = time.time()
    queries = attention.q_proj(hidden_states).view(batch_size, seq_len, attention.num_heads, attention.head_dim)
    keys = attention.k_proj(hidden_states).view(batch_size, seq_len, attention.num_heads, attention.head_dim)
    values = attention.v_proj(hidden_states).view(batch_size, seq_len, attention.num_heads, attention.head_dim)
    print(f"   Projections time: {time.time() - start:.4f}s")
    
    print("2. Polynomial feature mapping...")
    start = time.time()
    try:
        poly_queries = attention.poly_map(queries)
        poly_keys = attention.poly_map(keys)
        print(f"   Polynomial mapping time: {time.time() - start:.4f}s")
        print(f"   Poly queries shape: {poly_queries.shape}")
    except Exception as e:
        print(f"   Polynomial mapping failed: {e}")
        return False
    
    print("3. Memory processing...")
    start = time.time()
    try:
        # Reshape for batch processing
        poly_q_flat = poly_queries.reshape(-1, poly_queries.shape[-1])
        
        # This might be where it hangs
        print(f"   Flattened shape: {poly_q_flat.shape}")
        memory_output = attention.memory(poly_q_flat)
        print(f"   Memory processing time: {time.time() - start:.4f}s")
        print(f"   Memory output shape: {memory_output.shape}")
    except Exception as e:
        print(f"   Memory processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("4. Output projection...")
    start = time.time()
    try:
        # Project back to head_dim
        if memory_output.shape[-1] >= attention.head_dim:
            attention_output = memory_output[..., :attention.head_dim]
        else:
            pad_size = attention.head_dim - memory_output.shape[-1]
            attention_output = torch.nn.functional.pad(memory_output, (0, pad_size))
        
        # Reshape back
        attention_output = attention_output.reshape(batch_size, seq_len, attention.num_heads, attention.head_dim)
        attention_output = attention_output.reshape(batch_size, seq_len, attention.hidden_size)
        
        # Final projection
        output = attention.o_proj(attention_output)
        print(f"   Output projection time: {time.time() - start:.4f}s")
        print(f"   Final output shape: {output.shape}")
    except Exception as e:
        print(f"   Output projection failed: {e}")
        return False
    
    print("✓ Attention debugging completed!")
    return True

if __name__ == "__main__":
    print("Model Forward Pass Debugging")
    print("=" * 50)
    
    success = True
    
    # Debug attention first
    success &= debug_attention_module()
    
    # Then debug full model
    if success:
        success &= debug_model_forward()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All debugging tests passed!")
    else:
        print("✗ Some debugging tests failed!")