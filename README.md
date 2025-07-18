# Atlas: Learning to Optimally Memorize the Context at Test Time

This repository contains a PyTorch implementation of the Atlas model from the paper ["Atlas: Learning to Optimally Memorize the Context at Test Time"](https://arxiv.org/abs/2505.23735). Atlas introduces a novel approach to long-term memory in transformers, enabling context memorization rather than individual token memorization.

## 🚀 Key Features

- **Deep Memory Modules**: Neural networks that store abstractions in parameters
- **Polynomial Feature Mapping**: Enhanced memory capacity through higher-order features  
- **Omega Rule**: Context-aware memory updates using sliding windows
- **Muon Optimizer**: Second-order optimization for better memory management
- **DeepTransformers**: Strict generalization of traditional Transformers

## 📋 Architecture Overview

Atlas addresses three key limitations in existing recurrent models:

1. **Online Nature**: Traditional models only optimize memory on current tokens
2. **Limited Capacity**: Memory capacity bounded by architecture constraints  
3. **Memory Management**: First-order optimization leads to suboptimal solutions

### Key Components

- **AtlasAttention**: Attention mechanism with polynomial features and deep memory
- **DeepMemoryModule**: Multi-layer neural memory for storing abstractions
- **PolynomialFeatureMap**: Higher-order feature expansion for enhanced capacity
- **AtlasMemoryUpdate**: Omega rule with Muon optimizer for context memorization

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd atlas
```

2. Install dependencies with uv:
```bash
uv sync
```

Required packages are automatically installed from `pyproject.toml`:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Datasets >= 2.14.0
- Weights & Biases (wandb)
- NumPy, tqdm, accelerate
- Hydra-core, OmegaConf

## 🚦 Quick Start

### Training

Train an Atlas model with default configuration:

```bash
uv run python main.py --mode train
```

Train with specific model size:
```bash
uv run python main.py --mode train --config small  # 512 hidden, 8 layers
uv run python main.py --mode train --config medium # 768 hidden, 12 layers  
uv run python main.py --mode train --config large  # 1024 hidden, 24 layers
```

Debug mode (quick testing):
```bash
uv run python main.py --mode train --debug
```

### Evaluation

Evaluate a trained model:
```bash
uv run python main.py --mode eval --model-path checkpoints/best_model.pt
```

Train and evaluate:
```bash
uv run python main.py --mode both
```

### Configuration

The model supports flexible configuration through dataclasses:

```python
from model import AtlasConfig

config = AtlasConfig(
    hidden_size=768,
    num_layers=12,
    memory_depth=2,              # Deep memory layers
    polynomial_degree=3,         # Polynomial feature degree
    context_window_size=512,     # Omega rule window size
    use_muon_optimizer=True      # Enable Muon optimizer
)
```

## 📊 Model Variants

### Atlas Models
- **Atlas**: Full model with Muon optimizer and polynomial features
- **OmegaNet**: Uses standard gradient descent with polynomial features
- **DeepTransformer**: Deep memory without sliding window updates
- **DLA (Deep Linear Attention)**: Baseline with deep memory

### Configuration Examples

```python
# Small Atlas model
AtlasConfig(
    hidden_size=512,
    num_layers=8,
    memory_depth=2,
    polynomial_degree=3,
    context_window_size=256
)

# Large Atlas model  
AtlasConfig(
    hidden_size=1024,
    num_layers=24,
    memory_depth=3,
    polynomial_degree=4,
    context_window_size=1024
)
```

## 🔬 Evaluation Suite

The implementation includes comprehensive evaluation tools:

### Perplexity Evaluation
```python
from evaluate import AtlasEvaluator

evaluator = AtlasEvaluator(model, tokenizer)
perplexity = evaluator.compute_perplexity(test_texts)
```

### Needle-in-Haystack Test
Tests ability to retrieve information from long contexts:
```python
results = evaluator.needle_in_haystack_test(
    needle="The secret number is 42",
    haystack=long_context_text,
    question="What is the secret number?"
)
```

### Memory Capacity Test
Evaluates how many key-value pairs the model can memorize:
```python
capacity_score = evaluator.memory_capacity_test(num_pairs=100)
```

### Context Length Scaling
Tests performance across different context lengths:
```python
scaling_results = evaluator.context_length_scaling_test(
    base_text=sample_text,
    lengths=[512, 1024, 2048, 4096]
)
```

## 📈 Training Features

### Data Loading
- **LongContextDataset**: Sliding window dataset for long sequences
- **RecallDataset**: Synthetic recall tasks (BABIL-style)
- **BookDataset**: Load from text files and JSON

### Training Loop
- Gradient clipping and warmup scheduling
- Mixed precision training support
- Wandb logging integration
- Automatic checkpointing
- Evaluation during training

### Memory Management
- Context window buffer for Omega rule
- Dynamic decay weights for past tokens
- Polynomial feature caching
- Efficient parallelizable training

## 🧪 Experimental Results

Based on the paper, Atlas achieves:

- **+80% accuracy** on 10M context length BABILong benchmark
- **Superior performance** on recall-intensive tasks
- **Better scaling** to longer sequences than Transformers
- **Enhanced memory capacity** through polynomial features

### Ablation Studies

The implementation supports ablation studies:

```python
# Disable polynomial features
config.polynomial_degree = 1

# Use standard gradient descent
config.use_muon_optimizer = False

# Reduce memory depth
config.memory_depth = 1

# Disable context memorization  
config.context_window_size = 1
```

## 🔧 Advanced Usage

### MCTS Optimization with Absolute-Zero-Reasoner

Atlas now includes a Monte Carlo Tree Search (MCTS) optimization system based on the Absolute-Zero-Reasoner approach for automatically discovering optimal model configurations.

#### Running MCTS Optimization

```bash
uv run python optimize.py
```

This will:
- Start with a base Atlas configuration
- Use MCTS to explore configuration space
- Generate reasoning for each configuration change
- Evaluate configurations through training/testing
- Save the best configuration and training data

#### How It Works

The optimization system adapts the Absolute-Zero-Reasoner paper's MCTS approach:

1. **State**: Each node represents an `AtlasConfig` with specific parameter values
2. **Actions**: Configuration mutations (e.g., `{"polynomial_degree": 4}`)
3. **Simulation**: Fast training/evaluation runs using subprocess calls
4. **Reasoning**: Structured explanations for parameter changes using `<R>...</R>` tags
5. **Reward**: Weighted combination of perplexity, needle accuracy, and memory capacity

#### MCTS Components

**Selection**: Uses PUCT (Polynomial Upper Confidence Trees) formula:
```
PUCT(s,a) = Q(s,a) + P(s,a) * c_puct * sqrt(N(s)) / (1 + N(s,a))
```

**Expansion**: Reasoner generates candidate configurations with Atlas-specific reasoning:
```
<R>The current context_window_size is small, limiting the Omega Rule's ability to 
perform long-range memory updates. I will test increasing it to 1024 to improve 
performance on the needle-in-haystack task.</R>
```

**Simulation**: Empirical evaluation through subprocess execution:
```bash
uv run python main.py --mode eval --config-json config.json --output-json --debug
```

**Backpropagation**: Updates Q-values and visit counts along the path to root.

#### Configuration Space

The optimizer explores these Atlas-specific parameters:

- `memory_depth`: Depth of DeepMemoryModule (1-4 layers)
- `polynomial_degree`: PolynomialFeatureMap degree (1-5)
- `context_window_size`: Omega Rule window size (128-1024)
- `hidden_size`: Model hidden dimensions (256-1024)
- `num_layers`: Transformer layers (4-24)
- `learning_rate_inner`: Inner optimization rate for Muon
- `use_muon_optimizer`: Enable/disable second-order optimization

#### Output Files

After optimization:

- `best_atlas_config.json`: Optimal configuration found
- `reasoner_training_data.jsonl`: Training data for reasoner improvement

#### Custom Optimization

```python
from optimize import AtlasOptimizer, AtlasConfigOptimizer

# Custom initial configuration
initial_config = AtlasConfigOptimizer(
    hidden_size=512,
    memory_depth=2,
    polynomial_degree=3,
    context_window_size=256
)

# Run optimization
optimizer = AtlasOptimizer(initial_config, max_simulations=100)
best_config, best_reward, training_data = optimizer.optimize()
```

#### Programmatic Configuration Testing

Test specific configurations:

```bash
# Create config file
echo '{"memory_depth": 3, "polynomial_degree": 4}' > test_config.json

# Test configuration
uv run python main.py --mode eval --config-json test_config.json --output-json --debug
```

Returns structured metrics:
```json
{
  "perplexity": 45.2,
  "memory_capacity_score": -2.1,
  "needle_accuracy": 0.73
}
```

#### Reward Function

The optimization uses a weighted reward combining:

```python
def calculate_reward(metrics):
    ppl_score = max(0, 1 - (perplexity / 100.0))      # 40% weight
    needle_acc = needle_accuracy                        # 40% weight  
    mem_score = (memory_capacity_score + 10) / 10.0   # 20% weight
    return 0.4 * ppl_score + 0.4 * needle_acc + 0.2 * mem_score
```

This balances language modeling performance, retrieval capability, and memory efficiency.

### Custom Memory Architectures

```python
from model import DeepMemoryModule

# Custom memory module
memory = DeepMemoryModule(
    input_dim=768,
    hidden_dim=512, 
    num_layers=3
)
```

### Custom Feature Mappings

```python
from model import PolynomialFeatureMap

# Higher-order polynomial
poly_map = PolynomialFeatureMap(
    input_dim=64,
    degree=5
)
```

### Training with Custom Data

```python
from data_utils import LongContextDataset

dataset = LongContextDataset(
    texts=your_texts,
    tokenizer=tokenizer,
    max_length=2048,
    overlap_ratio=0.2
)
```

## 📝 Implementation Notes

### Key Design Decisions

1. **Memory Update**: Simplified Omega rule implementation for efficiency
2. **Polynomial Features**: Element-wise powers for computational efficiency
3. **Muon Optimizer**: Simplified approximation of second-order information
4. **Context Buffer**: Fixed-size circular buffer for context window

### Performance Considerations

- Use mixed precision training for large models
- Gradient checkpointing for memory efficiency  
- Parallel data loading with multiple workers
- Model compilation for faster training (PyTorch 2.0+)

### Limitations

- Simplified Muon optimizer (full implementation more complex)
- Element-wise polynomial features (tensor products more powerful)
- Fixed context window size (adaptive sizing could improve performance)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Full Muon optimizer implementation
- Tensor product polynomial features
- Adaptive context window sizing
- Additional evaluation benchmarks
- Memory-efficient implementations

## 📚 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{behrouz2025atlas,
  title={Atlas: Learning to Optimally Memorize the Context at Test Time},
  author={Behrouz, Ali and Li, Zeman and Kacham, Praneeth and others},
  journal={arXiv preprint arXiv:2505.23735},
  year={2025}
}
```

## 📄 License

This implementation is provided for research purposes. Please refer to the original paper for licensing terms.

## 🔗 References

- [Original Paper](https://arxiv.org/abs/2505.23735)
- [Transformers Library](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
