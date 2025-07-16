"""
Configuration management for Atlas training
"""

from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_seq_length: int = 2048
    dropout: float = 0.1
    
    # Atlas-specific parameters
    memory_depth: int = 2
    memory_hidden_size: int = 512
    polynomial_degree: int = 3
    context_window_size: int = 512
    learning_rate_inner: float = 0.01
    momentum_beta: float = 0.9
    use_muon_optimizer: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_steps: int = 1000
    log_steps: int = 100
    save_steps: int = 5000
    
    # Optimization
    gradient_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass  
class DataConfig:
    """Data configuration"""
    dataset_path: str = "data/"
    train_split: float = 0.9
    max_length: int = 512
    num_workers: int = 4
    
    # Data processing
    preprocessing: bool = True
    cache_dir: str = ".cache"


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    name: str = "atlas_experiment"
    project: str = "atlas-training"
    tags: List[str] = field(default_factory=lambda: ["atlas", "transformer"])
    notes: str = ""
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Hardware
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: bool = True
    compile_model: bool = False


@dataclass
class Config:
    """Main configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Overrides
    debug: bool = False
    seed: int = 42


# Register configurations with Hydra
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="model", name="small", node=ModelConfig(hidden_size=512, num_layers=8, num_heads=8))
cs.store(group="model", name="medium", node=ModelConfig(hidden_size=768, num_layers=12, num_heads=12))
cs.store(group="model", name="large", node=ModelConfig(hidden_size=1024, num_layers=24, num_heads=16))

cs.store(group="training", name="quick", node=TrainingConfig(max_steps=1000, eval_steps=100))
cs.store(group="training", name="full", node=TrainingConfig(max_steps=100000, eval_steps=1000))
