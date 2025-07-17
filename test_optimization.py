"""
Extended testing of the MCTS optimization system
"""

from optimize import AtlasOptimizer, AtlasConfigOptimizer
import json
import time

def test_different_initial_configs():
    """Test optimization from different starting points"""
    
    configs_to_test = [
        # Small efficient config
        AtlasConfigOptimizer(
            hidden_size=256,
            num_layers=4,
            memory_depth=1,
            polynomial_degree=1,
            context_window_size=128
        ),
        
        # Medium config with higher polynomial degree
        AtlasConfigOptimizer(
            hidden_size=512,
            num_layers=6,
            memory_depth=2,
            polynomial_degree=3,
            context_window_size=256
        ),
        
        # Larger config with different balance
        AtlasConfigOptimizer(
            hidden_size=768,
            num_layers=8,
            memory_depth=3,
            polynomial_degree=2,
            context_window_size=512
        )
    ]
    
    results = []
    all_training_data = []
    
    for i, config in enumerate(configs_to_test):
        print(f"\n{'='*60}")
        print(f"TESTING CONFIGURATION {i+1}/3")
        print(f"{'='*60}")
        print(f"Initial config: {config.to_dict()}")
        
        # Run shorter optimization for testing
        optimizer = AtlasOptimizer(config, max_simulations=10)
        
        start_time = time.time()
        best_config, best_reward, training_data = optimizer.optimize()
        end_time = time.time()
        
        result = {
            'initial_config': config.to_dict(),
            'best_config': best_config.to_dict(),
            'best_reward': best_reward,
            'time_taken': end_time - start_time,
            'num_training_examples': len(training_data)
        }
        
        results.append(result)
        all_training_data.extend(training_data)
        
        print(f"\nResult {i+1}:")
        print(f"  Best reward: {best_reward:.4f}")
        print(f"  Time taken: {end_time - start_time:.2f}s")
        print(f"  Training examples: {len(training_data)}")
    
    # Save comprehensive results
    with open('optimization_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save all training data
    with open('extended_reasoner_training_data.jsonl', 'w') as f:
        for data_point in all_training_data:
            f.write(json.dumps(data_point) + '\n')
    
    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*60}")
    print(f"Total configurations tested: {len(results)}")
    print(f"Total training examples generated: {len(all_training_data)}")
    print(f"Results saved to optimization_test_results.json")
    print(f"Training data saved to extended_reasoner_training_data.jsonl")
    
    # Print summary
    print(f"\nSUMMARY:")
    for i, result in enumerate(results):
        print(f"  Config {i+1}: reward {result['best_reward']:.4f} in {result['time_taken']:.1f}s")

if __name__ == "__main__":
    test_different_initial_configs()