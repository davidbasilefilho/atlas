"""
Evaluation utilities for Atlas model
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import AtlasModel, AtlasConfig


class AtlasEvaluator:
    """Evaluation suite for Atlas model"""
    
    def __init__(self, model: AtlasModel, tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def compute_perplexity(self, texts: List[str]) -> float:
        """Compute perplexity on a list of texts"""
        total_log_prob = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Computing perplexity"):
                tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
                
                if tokens.size(1) <= 1:
                    continue
                
                # Forward pass
                logits = self.model(tokens)
                
                # Compute log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get target log probabilities
                targets = tokens[:, 1:]
                target_log_probs = log_probs[:, :-1].gather(2, targets.unsqueeze(-1)).squeeze(-1)
                
                total_log_prob += target_log_probs.sum().item()
                total_tokens += targets.numel()
        
        if total_tokens == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def needle_in_haystack_test(self, needle: str, haystack: str, 
                               question: str, max_length: int = 2048) -> Dict[str, float]:
        """
        Test model's ability to retrieve information from long contexts
        (Inspired by the needle-in-haystack evaluation from the paper)
        """
        # Insert needle at different positions in haystack
        results = {}
        haystack_tokens = self.tokenizer.encode(haystack)
        needle_tokens = self.tokenizer.encode(needle)
        question_tokens = self.tokenizer.encode(question)
        
        # Test different positions
        positions = [0.1, 0.3, 0.5, 0.7, 0.9]  # Relative positions in haystack
        
        for pos in positions:
            # Insert needle at position
            insert_pos = int(pos * len(haystack_tokens))
            context_tokens = (haystack_tokens[:insert_pos] + 
                            needle_tokens + 
                            haystack_tokens[insert_pos:])
            
            # Truncate if too long
            if len(context_tokens) + len(question_tokens) > max_length:
                context_tokens = context_tokens[:max_length - len(question_tokens)]
            
            # Create full prompt
            full_tokens = context_tokens + question_tokens
            input_ids = torch.tensor([full_tokens], device=self.device)
            
            with torch.no_grad():
                logits = self.model(input_ids)
                
                # Get probability of correct answer (simplified)
                # In practice, you'd need more sophisticated answer extraction
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                
                # Store max probability as a proxy for confidence
                results[f'position_{pos}'] = probs.max().item()
        
        return results
    
    def memory_capacity_test(self, num_pairs: int = 100, key_dim: int = 64) -> float:
        """
        Test the memory capacity of the Atlas model
        """
        # Generate random key-value pairs
        keys = torch.randn(num_pairs, key_dim, device=self.device)
        values = torch.randn(num_pairs, key_dim, device=self.device)
        
        # Create a sequence with these pairs
        sequence = []
        for i in range(num_pairs):
            # Convert to tokens (simplified representation)
            key_str = f"KEY_{i}: " + " ".join([f"{x:.2f}" for x in keys[i][:5]])  # Use first 5 dims
            val_str = f"VAL_{i}: " + " ".join([f"{x:.2f}" for x in values[i][:5]])
            sequence.append(key_str + " " + val_str + " ")
        
        # Create input sequence
        text = " ".join(sequence)
        tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        if tokens.size(1) > self.model.config.max_seq_length:
            tokens = tokens[:, :self.model.config.max_seq_length]
        
        with torch.no_grad():
            logits = self.model(tokens)
            
            # Compute loss (how well it predicts the sequence)
            targets = tokens[:, 1:]
            logits = logits[:, :-1]
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                 targets.reshape(-1))
            
            # Return negative loss as a capacity measure (higher is better)
            return -loss.item()
    
    def context_length_scaling_test(self, base_text: str, 
                                  lengths: List[int] = [512, 1024, 2048, 4096]) -> Dict[int, float]:
        """
        Test how performance scales with context length
        """
        results = {}
        
        for length in lengths:
            # Create text of target length
            repeated_text = (base_text + " ") * (length // len(base_text) + 1)
            tokens = self.tokenizer.encode(repeated_text)[:length]
            
            if len(tokens) < 10:  # Skip if too short
                continue
            
            input_ids = torch.tensor([tokens], device=self.device)
            
            with torch.no_grad():
                logits = self.model(input_ids)
                
                # Compute perplexity on last portion
                targets = input_ids[:, 1:]
                logits = logits[:, :-1]
                
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                     targets.reshape(-1))
                
                perplexity = math.exp(loss.item())
                results[length] = perplexity
        
        return results
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 1.0, top_k: int = 50) -> str:
        """
        Generate text using the Atlas model
        """
        self.model.eval()
        
        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        generated = tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.model(generated)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text[len(prompt):]  # Return only the generated part
    
    def run_full_evaluation(self, eval_texts: List[str], 
                           save_plots: bool = True) -> Dict[str, any]:
        """
        Run comprehensive evaluation suite
        """
        results = {}
        
        print("Running perplexity evaluation...")
        results['perplexity'] = self.compute_perplexity(eval_texts)
        
        print("Running memory capacity test...")
        results['memory_capacity'] = self.memory_capacity_test()
        
        print("Running context length scaling test...")
        base_text = " ".join(eval_texts[:5]) if eval_texts else "The quick brown fox jumps over the lazy dog."
        results['context_scaling'] = self.context_length_scaling_test(base_text)
        
        print("Running needle-in-haystack test...")
        if eval_texts:
            haystack = " ".join(eval_texts[:3])
            needle = "The secret number is 42."
            question = "What is the secret number?"
            results['needle_haystack'] = self.needle_in_haystack_test(needle, haystack, question)
        
        print("Testing text generation...")
        sample_prompts = [
            "Once upon a time",
            "The future of artificial intelligence",
            "In a world where machines can think"
        ]
        
        results['generated_texts'] = {}
        for prompt in sample_prompts:
            generated = self.generate_text(prompt, max_length=50)
            results['generated_texts'][prompt] = generated
        
        # Plot results if requested
        if save_plots:
            self.plot_results(results)
        
        return results
    
    def plot_results(self, results: Dict[str, any], save_dir: str = "plots"):
        """
        Create visualizations of evaluation results
        """
        try:
            import os
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            os.makedirs(save_dir, exist_ok=True)
            
            # Plot context length scaling
            if 'context_scaling' in results:
                plt.figure(figsize=(10, 6))
                lengths = list(results['context_scaling'].keys())
                perplexities = list(results['context_scaling'].values())
                
                plt.plot(lengths, perplexities, 'bo-', linewidth=2, markersize=8)
                plt.xlabel('Context Length')
                plt.ylabel('Perplexity')
                plt.title('Atlas Model: Context Length vs Perplexity')
                plt.grid(True, alpha=0.3)
                plt.xscale('log')
                
                # Only use log scale for y-axis if all values are positive
                if all(p > 0 for p in perplexities):
                    plt.yscale('log')
                
                plt.savefig(f"{save_dir}/context_scaling.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot needle-in-haystack results
            if 'needle_haystack' in results:
                plt.figure(figsize=(10, 6))
                positions = [float(k.split('_')[1]) for k in results['needle_haystack'].keys()]
                scores = list(results['needle_haystack'].values())
                
                plt.plot(positions, scores, 'ro-', linewidth=2, markersize=8)
                plt.xlabel('Needle Position (Relative)')
                plt.ylabel('Retrieval Confidence')
                plt.title('Atlas Model: Needle-in-Haystack Performance')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)
                plt.savefig(f"{save_dir}/needle_haystack.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Plots saved to {save_dir}/")
            
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
            print("Continuing without plots...")


def main():
    """Example usage of the evaluator"""
    from transformers import GPT2Tokenizer
    
    # Load model and tokenizer
    config = AtlasConfig(hidden_size=512, num_layers=8)
    model = AtlasModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize evaluator
    evaluator = AtlasEvaluator(model, tokenizer, device='cpu')
    
    # Sample evaluation texts
    eval_texts = [
        "The Atlas model represents a significant advancement in transformer architectures.",
        "Context memorization is crucial for long-sequence understanding in language models.",
        "Deep memory modules can enhance the capacity of neural networks significantly."
    ]
    
    # Run evaluation
    results = evaluator.run_full_evaluation(eval_texts, save_plots=True)
    
    print("\n=== Evaluation Results ===")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Memory Capacity Score: {results['memory_capacity']:.4f}")
    print("\nContext Scaling Results:")
    for length, ppl in results['context_scaling'].items():
        print(f"  Length {length}: PPL = {ppl:.2f}")


if __name__ == "__main__":
    main()
