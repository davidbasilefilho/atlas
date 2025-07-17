"""
MCTS Optimization for Atlas Model Configuration
Based on Absolute-Zero-Reasoner approach
"""

import json
import math
import random
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from collections import defaultdict

@dataclass
class AtlasConfigOptimizer:
    """Configuration for Atlas model optimization"""
    # Default Atlas configuration values
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    memory_depth: int = 2
    memory_hidden_size: int = 512
    polynomial_degree: int = 3
    context_window_size: int = 512
    learning_rate_inner: float = 0.01
    momentum_beta: float = 0.9
    use_muon_optimizer: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def from_dict(self, config_dict: Dict) -> 'AtlasConfigOptimizer':
        """Create from dictionary"""
        new_config = AtlasConfigOptimizer()
        for key, value in config_dict.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
        return new_config


class MCTSNode:
    """MCTS Node representing an Atlas configuration state"""
    
    def __init__(self, config: AtlasConfigOptimizer, parent: Optional['MCTSNode'] = None, action: Optional[Dict] = None):
        self.config = config
        self.parent = parent
        self.action = action  # The action that led to this node
        self.children: List['MCTSNode'] = []
        
        # MCTS statistics
        self.visits = 0
        self.total_reward = 0.0
        self.q_value = 0.0
        
        # For tracking which actions have been tried
        self.tried_actions: List[Dict] = []
        
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried"""
        # For now, we'll consider a node fully expanded after trying several actions
        return len(self.tried_actions) >= 8  # Limit expansion for computational efficiency
    
    def best_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        """Select best child using PUCT formula"""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                score = float('inf')  # Prioritize unvisited nodes
            else:
                # PUCT formula: Q(s,a) + P(s,a) * c_puct * sqrt(N(s)) / (1 + N(s,a))
                q_value = child.q_value
                # For simplicity, we use uniform prior P(s,a) = 1/num_children
                prior_prob = 1.0 / len(self.children) if len(self.children) > 0 else 1.0
                exploration_term = c_puct * prior_prob * math.sqrt(self.visits) / (1 + child.visits)
                score = q_value + exploration_term
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def update(self, reward: float):
        """Update node statistics with reward"""
        self.visits += 1
        self.total_reward += reward
        self.q_value = self.total_reward / self.visits
    
    def add_child(self, config: AtlasConfigOptimizer, action: Dict) -> 'MCTSNode':
        """Add a child node"""
        child = MCTSNode(config, parent=self, action=action)
        self.children.append(child)
        self.tried_actions.append(action)
        return child


class AtlasReasoner:
    """Reasoner component that generates configuration mutations with structured reasoning"""
    
    def __init__(self):
        # Define the parameter space for Atlas model
        self.param_ranges = {
            'hidden_size': [256, 512, 768, 1024],
            'num_layers': [4, 6, 8, 12, 16, 24],
            'memory_depth': [1, 2, 3, 4],
            'polynomial_degree': [1, 2, 3, 4, 5],
            'context_window_size': [128, 256, 512, 1024],
            'learning_rate_inner': [0.001, 0.01, 0.1],
            'momentum_beta': [0.8, 0.9, 0.95, 0.99],
            'use_muon_optimizer': [True, False]
        }
    
    def generate_reasoning_and_actions(self, current_config: AtlasConfigOptimizer, num_actions: int = 3) -> Tuple[str, List[Dict]]:
        """
        Generate reasoning and candidate actions for the current configuration
        Returns: (reasoning_text, list_of_actions)
        """
        # Analyze current configuration
        current_dict = current_config.to_dict()
        
        # Generate reasoning based on Atlas architecture principles
        reasoning_parts = []
        actions = []
        
        # Reasoning about memory architecture
        if current_dict['memory_depth'] <= 2:
            reasoning_parts.append(
                f"The current memory_depth is {current_dict['memory_depth']}, which may limit the "
                "DeepMemoryModule's ability to store complex abstractions. I will test increasing it to "
                "enhance memory capacity."
            )
            actions.append({'memory_depth': min(4, current_dict['memory_depth'] + 1)})
        
        # Reasoning about polynomial features
        if current_dict['polynomial_degree'] <= 2:
            reasoning_parts.append(
                f"The PolynomialFeatureMap degree is {current_dict['polynomial_degree']}, which may "
                "not fully exploit higher-order feature interactions. Testing higher degree to improve "
                "memory capacity through enhanced feature representation."
            )
            actions.append({'polynomial_degree': min(5, current_dict['polynomial_degree'] + 1)})
        
        # Reasoning about context window and Omega rule
        if current_dict['context_window_size'] < 512:
            reasoning_parts.append(
                f"The context_window_size is {current_dict['context_window_size']}, limiting the "
                "Omega Rule's ability to perform long-range memory updates. Testing larger window "
                "to improve context memorization."
            )
            actions.append({'context_window_size': min(1024, current_dict['context_window_size'] * 2)})
        
        # Reasoning about model scale and computational efficiency
        if len(actions) < num_actions and current_dict['hidden_size'] < 768:
            reasoning_parts.append(
                f"The hidden_size is {current_dict['hidden_size']}, which may limit model capacity. "
                "Testing larger hidden size to improve representational power while monitoring "
                "computational efficiency."
            )
            actions.append({'hidden_size': min(1024, current_dict['hidden_size'] * 2)})
        
        # Add some exploration actions if we don't have enough
        while len(actions) < num_actions:
            param_name = random.choice(list(self.param_ranges.keys()))
            current_value = current_dict.get(param_name)
            possible_values = self.param_ranges[param_name]
            
            # Choose a different value
            new_value = random.choice([v for v in possible_values if v != current_value])
            action = {param_name: new_value}
            
            if action not in actions:
                reasoning_parts.append(
                    f"Exploring {param_name} = {new_value} to test its impact on performance."
                )
                actions.append(action)
        
        # Format reasoning with required tags
        reasoning_text = "<R>" + " ".join(reasoning_parts[:3]) + "</R>"  # Limit reasoning length
        
        return reasoning_text, actions[:num_actions]


class AtlasOptimizer:
    """MCTS-based optimizer for Atlas model configurations"""
    
    def __init__(self, initial_config: AtlasConfigOptimizer, max_simulations: int = 50):
        self.initial_config = initial_config
        self.root = MCTSNode(initial_config)
        self.max_simulations = max_simulations
        self.reasoner = AtlasReasoner()
        
        # Data collection for reasoner training
        self.training_data = []
        
    def simulate_config(self, config: AtlasConfigOptimizer) -> float:
        """
        Simulate a configuration by running training and evaluation
        Returns reward score
        """
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config.to_dict(), f)
            config_path = f.name
        
        try:
            # Run subprocess to evaluate configuration
            cmd = [
                'python', 'main.py',
                '--mode', 'eval',
                '--config-json', config_path,
                '--output-json',
                '--debug'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                # Parse JSON output
                output_lines = result.stdout.strip().split('\n')
                json_line = output_lines[-1]  # Last line should be JSON
                metrics = json.loads(json_line)
                
                # Calculate reward
                reward = self.calculate_reward(metrics)
                return reward
            else:
                print(f"Subprocess failed: {result.stderr}")
                return 0.0  # Return low reward for failed configurations
                
        except subprocess.TimeoutExpired:
            print(f"Simulation timeout for config: {config.to_dict()}")
            return 0.0
        except Exception as e:
            print(f"Simulation error: {e}")
            return 0.0
        finally:
            # Clean up temporary file
            try:
                os.unlink(config_path)
            except:
                pass
    
    def calculate_reward(self, metrics: Dict) -> float:
        """Calculate reward from evaluation metrics"""
        # Normalize perplexity (lower is better)
        ppl = metrics.get("perplexity", 100.0)
        ppl_score = max(0, 1 - (ppl / 100.0))
        
        # Needle accuracy is already 0-1
        needle_acc = metrics.get("needle_accuracy", 0.0)
        
        # Normalize memory capacity score
        mem_score = max(0, min(1, (metrics.get("memory_capacity_score", 0.0) + 10) / 10.0))
        
        # Weighted sum for final reward
        reward = (0.4 * ppl_score) + (0.4 * needle_acc) + (0.2 * mem_score)
        return reward
    
    def selection(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using PUCT until leaf"""
        current = node
        path = [current]
        
        while not current.is_leaf():
            current = current.best_child()
            path.append(current)
        
        return current
    
    def expansion(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expansion phase: add new child nodes using reasoner"""
        if node.is_fully_expanded():
            return None
        
        # Generate reasoning and actions using the reasoner
        reasoning, actions = self.reasoner.generate_reasoning_and_actions(node.config)
        
        # Filter out actions we've already tried
        new_actions = [action for action in actions if action not in node.tried_actions]
        
        if not new_actions:
            return None
        
        # Select first new action
        selected_action = new_actions[0]
        
        # Create new configuration by applying the action
        new_config_dict = node.config.to_dict()
        new_config_dict.update(selected_action)
        new_config = AtlasConfigOptimizer().from_dict(new_config_dict)
        
        # Add child node
        child = node.add_child(new_config, selected_action)
        
        # Store data for reasoner training
        state_dict = node.config.to_dict()
        policy = {str(action): 1.0 / len(actions) for action in actions}  # Uniform policy for now
        
        self.training_data.append({
            'state': state_dict,
            'reasoning': reasoning,
            'policy': policy,
            'actions': actions
        })
        
        return child
    
    def simulation(self, node: MCTSNode) -> float:
        """Simulation phase: evaluate configuration"""
        return self.simulate_config(node.config)
    
    def backpropagation(self, node: MCTSNode, reward: float):
        """Backpropagation phase: update statistics along path"""
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent
    
    def mcts_iteration(self) -> float:
        """Single MCTS iteration"""
        # 1. Selection
        leaf = self.selection(self.root)
        
        # 2. Expansion
        if not leaf.is_fully_expanded():
            child = self.expansion(leaf)
            if child is not None:
                leaf = child
        
        # 3. Simulation
        reward = self.simulation(leaf)
        
        # 4. Backpropagation
        self.backpropagation(leaf, reward)
        
        return reward
    
    def optimize(self) -> Tuple[AtlasConfigOptimizer, float, List[Dict]]:
        """
        Run MCTS optimization
        Returns: (best_config, best_reward, training_data)
        """
        print(f"Starting MCTS optimization with {self.max_simulations} simulations...")
        print(f"Initial configuration: {self.initial_config.to_dict()}")
        
        best_reward = -float('inf')
        best_config = self.initial_config
        
        for i in range(self.max_simulations):
            print(f"\nSimulation {i+1}/{self.max_simulations}")
            
            reward = self.mcts_iteration()
            
            if reward > best_reward:
                best_reward = reward
                # Find the best leaf node
                current = self.root
                while not current.is_leaf():
                    current = max(current.children, key=lambda x: x.q_value)
                best_config = current.config
                print(f"New best reward: {best_reward:.4f}")
                print(f"Best config: {best_config.to_dict()}")
            
            # Print current statistics
            if i % 5 == 0:
                print(f"Root visits: {self.root.visits}, Q-value: {self.root.q_value:.4f}")
                print(f"Children: {len(self.root.children)}")
        
        return best_config, best_reward, self.training_data
    
    def save_training_data(self, filename: str = "reasoner_training_data.jsonl"):
        """Save training data for reasoner improvement"""
        with open(filename, 'w') as f:
            for data_point in self.training_data:
                # Add final reward for each data point
                data_point['reward'] = self.root.q_value  # Use root's final Q-value as reward
                f.write(json.dumps(data_point) + '\n')
        
        print(f"Saved {len(self.training_data)} training examples to {filename}")


def main():
    """Main optimization loop"""
    # Initial configuration (small for fast testing)
    initial_config = AtlasConfigOptimizer(
        hidden_size=256,
        num_layers=4,
        memory_depth=1,
        polynomial_degree=2,
        context_window_size=128
    )
    
    # Create optimizer
    optimizer = AtlasOptimizer(initial_config, max_simulations=20)
    
    # Run optimization
    start_time = time.time()
    best_config, best_reward, training_data = optimizer.optimize()
    end_time = time.time()
    
    print(f"\n{'='*50}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*50}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Best reward: {best_reward:.4f}")
    print(f"Best configuration:")
    for key, value in best_config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Save training data
    optimizer.save_training_data()
    
    # Save best configuration
    with open('best_atlas_config.json', 'w') as f:
        json.dump(best_config.to_dict(), f, indent=2)
    
    print(f"\nBest configuration saved to best_atlas_config.json")
    print(f"Training data saved to reasoner_training_data.jsonl")


if __name__ == "__main__":
    main()