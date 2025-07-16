"""
Atlas: Learning to Optimally Memorize the Context at Test Time
Main entry point for training and evaluation
"""

import os
import torch
import argparse
from transformers import GPT2Tokenizer
import wandb

from model import AtlasModel, AtlasConfig
from train import AtlasTrainer
from evaluate import AtlasEvaluator
from data_utils import create_dataloaders
from config import Config


def setup_environment(config):
    """Setup training environment"""
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create output directories
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    os.makedirs(config.experiment.checkpoint_dir, exist_ok=True)
    os.makedirs(config.experiment.log_dir, exist_ok=True)
    
    # Determine device
    if config.experiment.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.experiment.device
    
    print(f"Using device: {device}")
    return device


def train_model(config):
    """Train the Atlas model"""
    print("Starting Atlas model training...")
    
    # Setup environment
    device = setup_environment(config)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Update config with tokenizer vocab size
    atlas_config = AtlasConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        max_seq_length=config.model.max_seq_length,
        dropout=config.model.dropout,
        memory_depth=config.model.memory_depth,
        memory_hidden_size=config.model.memory_hidden_size,
        polynomial_degree=config.model.polynomial_degree,
        context_window_size=config.model.context_window_size,
        learning_rate_inner=config.model.learning_rate_inner,
        momentum_beta=config.model.momentum_beta,
        use_muon_optimizer=config.model.use_muon_optimizer,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps
    )
    
    # Create model
    model = AtlasModel(atlas_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    print("Creating data loaders...")
    dataloaders = create_dataloaders(config, tokenizer)
    
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Validation batches: {len(dataloaders['val'])}")
    print(f"Recall test batches: {len(dataloaders['recall'])}")
    
    # Initialize trainer
    trainer = AtlasTrainer(atlas_config, model, device)
    
    # Start training
    trainer.train(
        train_dataloader=dataloaders['train'],
        eval_dataloader=dataloaders['val'],
        save_dir=config.experiment.checkpoint_dir,
        eval_steps=config.training.eval_steps
    )
    
    print("Training completed!")
    return trainer.model


def evaluate_model(config, model_path=None):
    """Evaluate the Atlas model"""
    print("Starting Atlas model evaluation...")
    
    device = setup_environment(config)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model config
    atlas_config = AtlasConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        max_seq_length=config.model.max_seq_length,
        dropout=config.model.dropout,
        memory_depth=config.model.memory_depth,
        memory_hidden_size=config.model.memory_hidden_size,
        polynomial_degree=config.model.polynomial_degree,
        context_window_size=config.model.context_window_size
    )
    
    # Create and load model
    model = AtlasModel(atlas_config)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Using randomly initialized model (for testing)")
    
    # Create evaluator
    evaluator = AtlasEvaluator(model, tokenizer, device)
    
    # Create test data
    dataloaders = create_dataloaders(config, tokenizer)
    
    # Sample some texts for evaluation
    eval_texts = []
    for batch in dataloaders['val']:
        input_ids = batch['input_ids']
        for seq in input_ids[:5]:  # Take first 5 sequences
            text = tokenizer.decode(seq, skip_special_tokens=True)
            eval_texts.append(text)
        break
    
    # Run evaluation
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
    
    print(f"\nResults and plots saved to {config.experiment.output_dir}")


def main():
    """Main function"""
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
    config = Config()
    
    if args.debug:
        config.debug = True
        config.training.max_steps = 100
        config.training.eval_steps = 50
        config.experiment.name += "_debug"
    
    print(f"Running in {args.mode} mode")
    print(f"Configuration: {args.config}")
    print(f"Debug mode: {config.debug}")
    
    if args.mode in ["train", "both"]:
        model = train_model(config)
        
        if args.mode == "both":
            # Save the trained model for evaluation
            model_path = os.path.join(config.experiment.checkpoint_dir, "final_model.pt")
        else:
            model_path = None
    
    if args.mode in ["eval", "both"]:
        if args.mode == "eval" and args.model_path:
            model_path = args.model_path
        elif args.mode == "both":
            model_path = os.path.join(config.experiment.checkpoint_dir, "final_model.pt")
        else:
            model_path = None
        
        evaluate_model(config, model_path)


if __name__ == "__main__":
    main()
