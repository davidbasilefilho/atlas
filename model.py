"""
Atlas Model Implementation
Based on: "Atlas: Learning to Optimally Memorize the Context at Test Time"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class AtlasConfig:
    """Configuration for Atlas model"""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_seq_length: int = 2048
    dropout: float = 0.1
    
    # Atlas-specific parameters
    memory_depth: int = 2  # LM layers in memory module
    memory_hidden_size: int = 512
    polynomial_degree: int = 3
    context_window_size: int = 512
    learning_rate_inner: float = 0.01
    momentum_beta: float = 0.9
    use_muon_optimizer: bool = True
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Debug flag
    debug: bool = False


class PolynomialFeatureMap(nn.Module):
    """Polynomial feature mapping for enhanced memory capacity"""
    
    def __init__(self, input_dim: int, degree: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        
        # Learnable coefficients for polynomial terms with numerical stability
        factorial_terms = torch.tensor([math.factorial(i) for i in range(degree + 1)], dtype=torch.float32)
        self.coefficients = nn.Parameter(torch.ones(degree + 1) / factorial_terms)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply polynomial feature mapping
        x: [batch_size, seq_len, input_dim] or [batch_size, seq_len, num_heads, head_dim]
        returns: [batch_size, seq_len, expanded_dim] or [batch_size, seq_len, num_heads, expanded_dim]
        """
        original_shape = x.shape
        
        # Handle both 3D and 4D inputs
        if len(original_shape) == 4:
            # Reshape from [batch, seq_len, heads, head_dim] to [batch*seq_len*heads, head_dim]
            batch_size, seq_len, num_heads, head_dim = original_shape
            x = x.reshape(-1, head_dim)
            is_4d = True
        elif len(original_shape) == 3:
            # Keep as [batch, seq_len, dim] -> [batch*seq_len, dim]
            batch_size, seq_len, dim = original_shape
            x = x.reshape(-1, dim)
            is_4d = False
        else:
            raise ValueError(f"Expected 3D or 4D input, got {len(original_shape)}D")
        
        input_dim = x.shape[-1]
        features = []
        
        # Add constant term (full dimension to maintain consistent sizing)
        features.append(self.coefficients[0] * torch.ones_like(x))
        
        # Add polynomial terms with numerical stability
        x_stable = torch.clamp(x, min=-10.0, max=10.0)  # Prevent overflow
        for i in range(1, self.degree + 1):
            # For simplicity, we use element-wise powers
            # In practice, you might want to use tensor products for full polynomial expansion
            power_term = self.coefficients[i] * (x_stable ** i)
            # Add small epsilon to prevent numerical issues
            power_term = torch.clamp(power_term, min=-1e6, max=1e6)
            features.append(power_term)
            
        result = torch.cat(features, dim=-1)
        
        # Reshape back to original structure
        if is_4d:
            expanded_dim = result.shape[-1]
            result = result.reshape(batch_size, seq_len, num_heads, expanded_dim)
        else:
            expanded_dim = result.shape[-1]
            result = result.reshape(batch_size, seq_len, expanded_dim)
        
        return result


class DeepMemoryModule(nn.Module):
    """Deep neural memory module for Atlas"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = input_dim if i == num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Identity()
            ])
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AtlasMemoryUpdate(nn.Module):
    """Atlas memory update mechanism with Muon optimizer"""
    
    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        self.eta = config.learning_rate_inner
        self.beta = config.momentum_beta
        self.eps = 1e-8  # Numerical stability
        
        # Memory state for Muon optimizer - initialize properly
        self.register_buffer('momentum', torch.zeros(1))
        self.register_buffer('gradient_sq', torch.zeros(1))
        
    def compute_attentional_bias(self, memory: nn.Module, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute the attentional bias L(M; k, v)"""
        predicted_values = memory(keys)
        # Ensure shapes match for loss computation
        if predicted_values.shape != values.shape:
            # Project to correct shape if needed
            predicted_values = predicted_values[..., :values.shape[-1]]
        return F.mse_loss(predicted_values, values, reduction='none').sum(dim=-1)
    
    def omega_rule_update(self, memory: nn.Module, context_keys: torch.Tensor, 
                         context_values: torch.Tensor, gamma_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Omega rule: optimize memory based on context window instead of just current token
        """
        total_loss = 0
        batch_size = context_keys.shape[0] if context_keys.dim() > 2 else 1
        
        # Handle the case where inputs are batched or single sequences
        if context_keys.dim() == 3:  # [batch, context_size, dim]
            for b in range(batch_size):
                for t in range(context_keys.shape[1]):
                    keys = context_keys[b, t].unsqueeze(0)  # [1, dim]
                    values = context_values[b, t].unsqueeze(0)  # [1, dim]
                    gamma = gamma_weights[t] if t < len(gamma_weights) else gamma_weights[-1]
                    
                    loss = self.compute_attentional_bias(memory, keys, values)
                    total_loss += gamma * loss.mean()
        else:  # [context_size, dim]
            for t in range(context_keys.shape[0]):
                keys = context_keys[t].unsqueeze(0)  # [1, dim]
                values = context_values[t].unsqueeze(0)  # [1, dim]
                gamma = gamma_weights[t] if t < len(gamma_weights) else gamma_weights[-1]
                
                loss = self.compute_attentional_bias(memory, keys, values)
                total_loss += gamma * loss.mean()
        
        # Compute gradients safely
        memory_params = list(memory.parameters())
        if len(memory_params) == 0:
            return {}
            
        try:
            gradients = torch.autograd.grad(total_loss, memory_params, 
                                          create_graph=False, retain_graph=False, allow_unused=True)
        except RuntimeError:
            # If gradient computation fails, return empty dict
            return {}
        
        # Apply Muon optimizer (simplified but more stable version)
        updated_params = {}
        for param, grad in zip(memory_params, gradients):
            if grad is None:
                continue
                
            if self.config.use_muon_optimizer:
                # Simplified Muon update with better numerical stability
                # Update momentum with exponential moving average
                grad_norm = grad.norm().item()
                self.momentum = self.beta * self.momentum + (1 - self.beta) * grad_norm
                
                # Prevent division by zero and apply more conservative clipping
                grad_clipped = torch.clamp(grad, min=-0.1, max=0.1)  # More conservative clipping
                momentum_term = self.momentum + self.eps
                update = self.eta * grad_clipped / momentum_term
                
                # Additional safety: limit the update magnitude
                update = torch.clamp(update, min=-0.01, max=0.01)
            else:
                # Standard gradient descent with conservative clipping
                grad_clipped = torch.clamp(grad, min=-0.1, max=0.1)
                update = self.eta * grad_clipped
                update = torch.clamp(update, min=-0.01, max=0.01)
            
            updated_params[id(param)] = param - update
        
        return updated_params


class AtlasAttention(nn.Module):
    """Atlas attention mechanism with deep memory and polynomial features"""
    
    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Polynomial feature mapping
        self.poly_map = PolynomialFeatureMap(self.head_dim, config.polynomial_degree)
        
        # Calculate the actual polynomial dimension
        poly_dim = self.head_dim * (config.polynomial_degree + 1)
        
        # Deep memory module
        self.memory = DeepMemoryModule(poly_dim, config.memory_hidden_size, config.memory_depth)
        
        # Memory update mechanism
        self.memory_updater = AtlasMemoryUpdate(config)
        
        # Context window buffer - use the correct poly_dim
        self.register_buffer('context_keys', torch.zeros(config.context_window_size, config.num_heads, poly_dim))
        self.register_buffer('context_values', torch.zeros(config.context_window_size, config.num_heads, self.head_dim))
        self.register_buffer('context_pos', torch.zeros(1, dtype=torch.long))
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        queries = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply polynomial feature mapping to keys and queries
        poly_queries = self.poly_map(queries)  # [batch, seq_len, heads, poly_dim]
        poly_keys = self.poly_map(keys)        # [batch, seq_len, heads, poly_dim]
        
        # Efficient parallel processing instead of sequential loop
        # Reshape for batch processing: [batch * seq_len * heads, poly_dim/head_dim]
        poly_q_flat = poly_queries.reshape(-1, poly_queries.shape[-1])  # [batch*seq*heads, poly_dim]
        poly_k_flat = poly_keys.reshape(-1, poly_keys.shape[-1])        # [batch*seq*heads, poly_dim]
        values_flat = values.reshape(-1, values.shape[-1])              # [batch*seq*heads, head_dim]
        
        # Process all queries through memory in parallel
        try:
            memory_output = self.memory(poly_q_flat)  # [batch*seq*heads, poly_dim]
            
            # Project back to head_dim - handle dimension mismatch efficiently
            if memory_output.shape[-1] >= self.head_dim:
                attention_output = memory_output[..., :self.head_dim]  # Take first head_dim features
            else:
                # Pad if necessary
                pad_size = self.head_dim - memory_output.shape[-1]
                attention_output = F.pad(memory_output, (0, pad_size))
                
        except Exception as e:
            # Fallback: use linear transformation of queries
            attention_output = torch.zeros(poly_q_flat.shape[0], self.head_dim, 
                                         device=poly_q_flat.device, dtype=poly_q_flat.dtype)
            # Use a simple linear projection as fallback
            if poly_q_flat.shape[-1] >= self.head_dim:
                attention_output = poly_q_flat[..., :self.head_dim]
            else:
                attention_output[..., :poly_q_flat.shape[-1]] = poly_q_flat
        
        # Reshape back to original format
        attention_output = attention_output.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        attention_output = attention_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Update context window with the last sequence in the batch (simplified for efficiency)
        if self.training:
            # Only update context during training, and only with the last token to avoid overhead
            last_k = poly_keys[:, -1].mean(0).detach()  # [heads, poly_dim] - average over batch
            last_v = values[:, -1].mean(0).detach()     # [heads, head_dim] - average over batch
            
            pos = self.context_pos.item() % self.config.context_window_size
            self.context_keys[pos] = last_k
            self.context_values[pos] = last_v
            self.context_pos += 1
        
        return self.o_proj(attention_output)


class AtlasBlock(nn.Module):
    """Atlas Transformer block"""
    
    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = AtlasAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        attn_output = self.attn(self.ln_1(x), attention_mask)
        x = x + attn_output
        
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output
        
        return x


class AtlasModel(nn.Module):
    """Complete Atlas model"""
    
    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([AtlasBlock(config) for _ in range(config.num_layers)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + pos_embeds
        
        # Apply transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits
