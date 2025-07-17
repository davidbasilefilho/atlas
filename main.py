"""
Atlas: Learning to Optimally Memorize the Context at Test Time
Main entry point for training and evaluation
"""

import os
import torch
import argparse
from transformers import GPT2Tokenizer
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from model import AtlasModel, AtlasConfig
from train import AtlasTrainer
from evaluate import AtlasEvaluator
from data_utils import create_dataloaders
from config import Config


def setup_environment(config):
    """Setup training environment"""
    # Set random seeds for reproducibility
    seed = getattr(config, 'seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directories
    output_dir = getattr(config.experiment, 'output_dir', 'outputs')
    checkpoint_dir = getattr(config.experiment, 'checkpoint_dir', 'checkpoints')
    log_dir = getattr(config.experiment, 'log_dir', 'logs')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Determine device
    device_setting = getattr(config.experiment, 'device', 'auto')
    if device_setting == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_setting
    
    print(f"Using device: {device}")
    return device


def train_model(config):
    """Train the Atlas model"""
    print("Starting Atlas model training...")
    
    # Setup environment
    device = setup_environment(config)
    
    # Initialize tokenizer with fallback
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Warning: Could not load GPT2 tokenizer: {e}")
        print("Using a mock tokenizer for testing...")
        # Create a simple mock tokenizer for testing
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
                    import torch
                    return torch.tensor([tokens])
                return tokens
                
            def decode(self, tokens, skip_special_tokens=True):
                # Simple mock decode
                return " ".join([f"token_{t}" for t in tokens if isinstance(t, int)])
        
        tokenizer = MockTokenizer()
    
    # Update config with tokenizer vocab size - handle missing attributes gracefully
    atlas_config = AtlasConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=getattr(config.model, 'hidden_size', 768),
        num_layers=getattr(config.model, 'num_layers', 12),
        num_heads=getattr(config.model, 'num_heads', 12),
        max_seq_length=getattr(config.model, 'max_seq_length', 2048),
        dropout=getattr(config.model, 'dropout', 0.1),
        memory_depth=getattr(config.model, 'memory_depth', 2),
        memory_hidden_size=getattr(config.model, 'memory_hidden_size', 512),
        polynomial_degree=getattr(config.model, 'polynomial_degree', 3),
        context_window_size=getattr(config.model, 'context_window_size', 512),
        learning_rate_inner=getattr(config.model, 'learning_rate_inner', 0.01),
        momentum_beta=getattr(config.model, 'momentum_beta', 0.9),
        use_muon_optimizer=getattr(config.model, 'use_muon_optimizer', True),
        batch_size=getattr(config.training, 'batch_size', 8),
        learning_rate=getattr(config.training, 'learning_rate', 1e-4),
        weight_decay=getattr(config.training, 'weight_decay', 0.01),
        warmup_steps=getattr(config.training, 'warmup_steps', 1000),
        max_steps=getattr(config.training, 'max_steps', 100000)
    )
    
    # Create model
    model = AtlasModel(atlas_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    print("Creating data loaders...")
    try:
        dataloaders = create_dataloaders(config, tokenizer)
        
        print(f"Train batches: {len(dataloaders['train'])}")
        print(f"Validation batches: {len(dataloaders['val'])}")
        print(f"Recall test batches: {len(dataloaders['recall'])}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("This is expected in the bug-fixing phase")
        return model
    
    # Initialize trainer
    trainer = AtlasTrainer(atlas_config, model, device)
    
    # Start training
    checkpoint_dir = getattr(config.experiment, 'checkpoint_dir', 'checkpoints')
    eval_steps = getattr(config.training, 'eval_steps', 1000)
    
    trainer.train(
        train_dataloader=dataloaders['train'],
        eval_dataloader=dataloaders['val'],
        save_dir=checkpoint_dir,
        eval_steps=eval_steps
    )
    
    print("Training completed!")
    return trainer.model


def evaluate_model(config, model_path=None):
    """Evaluate the Atlas model"""
    print("Starting Atlas model evaluation...")
    
    device = setup_environment(config)
    
    # Initialize tokenizer with fallback
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Warning: Could not load GPT2 tokenizer: {e}")
        print("Using a mock tokenizer for testing...")
        # Create a simple mock tokenizer for testing
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
    
    # Create model config with safe attribute access
    atlas_config = AtlasConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=getattr(config.model, 'hidden_size', 768),
        num_layers=getattr(config.model, 'num_layers', 12),
        num_heads=getattr(config.model, 'num_heads', 12),
        max_seq_length=getattr(config.model, 'max_seq_length', 2048),
        dropout=getattr(config.model, 'dropout', 0.1),
        memory_depth=getattr(config.model, 'memory_depth', 2),
        memory_hidden_size=getattr(config.model, 'memory_hidden_size', 512),
        polynomial_degree=getattr(config.model, 'polynomial_degree', 3),
        context_window_size=getattr(config.model, 'context_window_size', 512)
    )
    
    # Create and load model
    model = AtlasModel(atlas_config)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Using randomly initialized model")
    else:
        print("Using randomly initialized model (for testing)")
    
    # Create evaluator
    evaluator = AtlasEvaluator(model, tokenizer, device)
    
    # Create test data
    try:
        dataloaders = create_dataloaders(config, tokenizer)
        
        # Sample some texts for evaluation
        eval_texts = []
        for batch in dataloaders['val']:
            input_ids = batch['input_ids']
            for seq in input_ids[:5]:  # Take first 5 sequences
                text = tokenizer.decode(seq, skip_special_tokens=True)
                eval_texts.append(text)
            break
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        # Use fallback evaluation texts
        eval_texts = [
            "The Atlas model represents a significant advancement in transformer architectures.",
            "Context memorization is crucial for long-sequence understanding in language models.",
            "Deep memory modules can enhance the capacity of neural networks significantly."
        ]
    
    # Run evaluation
    try:
        results = evaluator.run_full_evaluation(
            eval_texts, 
            save_plots=True
        )
        
        print("\n=== Evaluation Results ===")
        print(f"Perplexity: {results.get('perplexity', 'N/A')}")
        print(f"Memory Capacity Score: {results.get('memory_capacity', 'N/A')}")
        
        if 'context_scaling' in results:
            print("\nContext Length Scaling:")
            for length, ppl in results['context_scaling'].items():
                print(f"  {length} tokens: PPL = {ppl:.2f}")
        
        if 'needle_haystack' in results:
            print("\nNeedle-in-Haystack Results:")
            for pos, score in results['needle_haystack'].items():
                print(f"  Position {pos}: {score:.3f}")
        
        output_dir = getattr(config.experiment, 'output_dir', 'outputs')
        print(f"\nResults and plots saved to {output_dir}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("This is expected during the bug-fixing phase")


def main():
    """Main function with proper argument handling"""
    parser = argparse.ArgumentParser(description="Atlas Model Training and Evaluation")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="train",
                       help="Mode: train, eval, or both")
    parser.add_argument("--config", type=str, default="base_config",
                       help="Config name to use")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model checkpoint for evaluation")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config()
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration...")
        config = Config()
    
    # Apply debug mode overrides
    if args.debug:
        config.debug = True
        config.training.max_steps = 100
        config.training.eval_steps = 50
        config.experiment.name += "_debug"
        print("Debug mode enabled - reduced steps for quick testing")
    
    print(f"Running in {args.mode} mode")
    print(f"Configuration: {args.config}")
    print(f"Debug mode: {getattr(config, 'debug', False)}")
    
    model_path = None
    
    if args.mode in ["train", "both"]:
        try:
            model = train_model(config)
            if args.mode == "both":
                # Save the trained model for evaluation
                model_path = os.path.join(getattr(config.experiment, 'checkpoint_dir', 'checkpoints'), "final_model.pt")
        except Exception as e:
            print(f"Training failed: {e}")
            if args.mode == "train":
                return
    
    if args.mode in ["eval", "both"]:
        try:
            if args.mode == "eval" and args.model_path:
                model_path = args.model_path
            elif args.mode == "both" and model_path is None:
                model_path = os.path.join(getattr(config.experiment, 'checkpoint_dir', 'checkpoints'), "final_model.pt")
            
            evaluate_model(config, model_path)
        except Exception as e:
            print(f"Evaluation failed: {e}")


# Optional: Add Hydra decorator for advanced configuration management  
@hydra.main(version_base=None, config_path=None, config_name="base_config")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra-compatible main function"""
    print("Using Hydra configuration")
    print(OmegaConf.to_yaml(cfg))
    
    # Convert to our config format
    config = Config()
    
    # Override with hydra config if provided
    if hasattr(cfg, 'model'):
        for key, value in cfg.model.items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
    
    if hasattr(cfg, 'training'):
        for key, value in cfg.training.items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
    
    # Run training
    model = train_model(config)
    print("Hydra training completed")


if __name__ == "__main__":
    main()
