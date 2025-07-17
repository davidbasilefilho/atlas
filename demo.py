"""
Demo script for Atlas model
Simple example showing text generation and evaluation
"""

import torch
from transformers import GPT2Tokenizer

from model import AtlasModel, AtlasConfig
from evaluate import AtlasEvaluator


def demo_atlas_model():
    """Demonstrate Atlas model capabilities"""
    
    print("🚀 Atlas Model Demo")
    print("=" * 50)
    
    # Initialize tokenizer with fallback
    print("📚 Loading tokenizer...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print("   ✓ GPT2 tokenizer loaded successfully")
    except Exception as e:
        print(f"   ⚠️  Could not load GPT2 tokenizer: {e}")
        print("   Using mock tokenizer for demo...")
        
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 50257
                self.eos_token = '</s>'
                self.pad_token = '</s>'
                self.eos_token_id = 50256
                
            def encode(self, text, add_special_tokens=True, return_tensors=None):
                tokens = [hash(word) % self.vocab_size for word in text.split()]
                if add_special_tokens:
                    tokens = [0] + tokens + [self.eos_token_id]
                if return_tensors == 'pt':
                    import torch
                    return torch.tensor([tokens])
                return tokens
                
            def decode(self, tokens, skip_special_tokens=True):
                return " ".join([f"token_{t}" for t in tokens if isinstance(t, int)])
        
        tokenizer = MockTokenizer()
    
    # Create small model for demo
    print("🏗️  Creating Atlas model...")
    config = AtlasConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=288,        # Divisible by 4, 6, 8, 9, 12 for flexibility
        num_layers=4,           # Few layers for demo
        num_heads=12,
        max_seq_length=512,
        memory_depth=2,
        polynomial_degree=2,
        context_window_size=128
    )
    
    model = AtlasModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Initialize evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    evaluator = AtlasEvaluator(model, tokenizer, device)
    
    # Demo 1: Text Generation
    print("\n🎨 Text Generation Demo")
    print("-" * 30)
    
    prompts = [
        "The Atlas model is",
        "In the future, artificial intelligence will",
        "The key to long-term memory is"
    ]
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        generated = evaluator.generate_text(prompt, max_length=30, temperature=0.8)
        print(f"Generated: '{generated}'")
        print()
    
    # Demo 2: Memory Capacity Test
    print("🧠 Memory Capacity Test")
    print("-" * 30)
    
    capacity_score = evaluator.memory_capacity_test(num_pairs=20)
    print(f"Memory capacity score: {capacity_score:.4f}")
    print("(Higher is better - indicates ability to memorize key-value pairs)")
    
    # Demo 3: Context Length Scaling
    print("\n📏 Context Length Scaling Test")
    print("-" * 30)
    
    base_text = """
    The Atlas model introduces several key innovations for long-term memory in transformers.
    First, it uses deep memory modules that store abstractions in neural network parameters.
    Second, polynomial feature mappings enhance memory capacity beyond linear scaling.
    Third, the Omega rule enables context memorization instead of individual token memorization.
    Finally, the Muon optimizer provides better memory management through second-order information.
    """
    
    scaling_results = evaluator.context_length_scaling_test(
        base_text, 
        lengths=[128, 256, 512]
    )
    
    print("Context Length -> Perplexity:")
    for length, ppl in scaling_results.items():
        print(f"  {length:3d} tokens: {ppl:.2f}")
    
    # Demo 4: Architecture Comparison
    print("\n🏛️  Architecture Comparison")
    print("-" * 30)
    
    print("Atlas Features:")
    print(f"  ✓ Deep Memory Depth: {config.memory_depth}")
    print(f"  ✓ Polynomial Degree: {config.polynomial_degree}")
    print(f"  ✓ Context Window: {config.context_window_size}")
    print(f"  ✓ Muon Optimizer: {config.use_muon_optimizer}")
    
    # Calculate memory capacity enhancement
    standard_capacity = config.hidden_size // config.num_heads  # Standard attention
    polynomial_capacity = standard_capacity * (config.polynomial_degree + 1)
    enhancement = polynomial_capacity / standard_capacity
    
    print(f"\nMemory Capacity Enhancement:")
    print(f"  Standard attention capacity: ~{standard_capacity}")
    print(f"  Atlas polynomial capacity: ~{polynomial_capacity}")
    print(f"  Enhancement factor: {enhancement:.1f}x")
    
    # Demo 5: Simple Needle-in-Haystack
    print("\n🔍 Simple Needle-in-Haystack Test")
    print("-" * 30)
    
    haystack = """
    There are many interesting facts about machine learning and artificial intelligence.
    Deep learning has revolutionized computer vision and natural language processing.
    Transformers have become the dominant architecture for sequence modeling tasks.
    The secret code is Alpha-7829-Beta for accessing the secure system database.
    Research continues to advance in areas like reinforcement learning and robotics.
    Neural networks can approximate complex functions with sufficient parameters.
    """
    
    needle = "The secret code is Alpha-7829-Beta"
    question = "What is the secret code?"
    
    results = evaluator.needle_in_haystack_test(needle, haystack, question)
    
    print("Retrieval confidence by position:")
    for pos, score in results.items():
        position = float(pos.split('_')[1])
        print(f"  Position {position:.1f}: {score:.3f}")
    
    print("\n✨ Demo completed!")
    print("   This demonstrates Atlas model capabilities on a small scale.")
    print("   For full performance, train on larger datasets with bigger models.")


if __name__ == "__main__":
    demo_atlas_model()
