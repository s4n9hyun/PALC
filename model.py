"""
PALC (Preference Alignment via Logit Calibration) - Core Model Components

PALC is a parameter-efficient fine-tuning method designed for preference alignment.
Instead of updating the entire model, PALC trains only a small "correction engine"
while keeping the base language model completely frozen.

🏗️ Architecture Overview:
┌─────────────────────────────────────────────────────────────────┐
│                        PALC Model                               │
├─────────────────────────────────────────────────────────────────┤
│  Input → Base Model (FROZEN) → Last Layer Hidden States        │
│                    ↓                    ↓                       │
│               Base Logits         Correction Engine             │
│                    ↓                    ↓                       │
│               (unchanged)         Correction Logits             │
│                    ↓                    ↓                       │
│                 Final Logits = Base + Scale × Correction        │
└─────────────────────────────────────────────────────────────────┘

🔧 Key Components:
1. BaseLanguageModel: Frozen 7B+ model (e.g., LLaMA, Tulu) - provides base predictions
2. LogitCorrectionEngine: Tiny trainable network (~0.03% of total params) - learns corrections
3. PALC: Orchestrates the combination with learnable scaling

🎯 Why This Works:
- Base model provides strong general capabilities (frozen = stable)
- Correction engine learns task-specific adjustments (tiny = efficient)
- Simple addition preserves base model's knowledge while adding alignment

📊 Training Efficiency:
- Only ~2M parameters trainable (vs 7B+ total)
- No reference model needed (unlike DPO)
- Direct preference optimization with SimplePreferenceLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Optional, Dict, Any, Tuple
import math


class BaseLanguageModel(nn.Module):
    """
    🧠 Frozen Base Language Model Wrapper

    This class wraps a large pretrained language model (7B+ parameters) and keeps
    it completely frozen during training. It serves as the "knowledge base" that
    provides both base predictions and hidden representations.

    🔒 Key Principle: FROZEN = STABLE
    - All parameters have requires_grad=False
    - Model stays in eval() mode
    - Provides consistent base predictions

    📤 What it provides:
    - Base logits: Original model's predictions (before calibration)
    - Hidden states: Last layer features for the calibration engine

    💡 Why Last Layer?
    - Contains the most refined high-level representations
    - Right before final logit computation
    - Optimal for steering final predictions
    """

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        """
        Load and freeze a pretrained language model.

        Args:
            model_name: HuggingFace model ID
            device: Target device ("cuda" or "cpu")
            torch_dtype: Data type for memory efficiency
        """
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_name = model_name

        # Load model with memory optimization for large models
        if "7b" in model_name.lower() or "llama" in model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map=None
            )
        else:
            # Standard loading for smaller models
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=None
            )

        # Freeze all parameters - core of PALC architecture
        for param in self.model.parameters():
            param.requires_grad = False

        # Store model configuration
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        # Move to device and set eval mode
        self.to(device)
        self.eval()

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False):
        """
        Forward pass through the base model.

        Returns logits and hidden states for correction engine.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True
        )

    def count_parameters(self) -> int:
        """Count total number of parameters in the base model."""
        return sum(p.numel() for p in self.model.parameters())

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the base model."""
        param_count = self.count_parameters()
        return {
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "parameters": param_count,
            "parameters_B": param_count / 1e9,
        }


class LogitCalibrationEngine(nn.Module):
    """
    Trainable calibration engine for PALC.

    Small bottleneck network that generates calibration logits to adjust
    the base model's predictions. This is the only trainable component.

    Architecture: Hidden States -> Compress -> Expand -> Calibration Logits
                  [batch, seq, 4096] -> [bottleneck] -> [vocab_size]

    Parameter efficiency examples:
    - Bottleneck 32:  ~130K parameters (0.002% of 7B model)
    - Bottleneck 128: ~540K parameters (0.008% of 7B model)
    - Bottleneck 256: ~8.4M parameters (0.12% of 7B model)
    """

    def __init__(self, hidden_size: int = 4096, vocab_size: int = 32000,
                 bottleneck_dim: int = 32,
                 device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        """
        Initialize the calibration engine.

        Args:
            hidden_size: Hidden state dimension from base model
            vocab_size: Vocabulary size
            bottleneck_dim: Bottleneck layer size (smaller = more efficient)
            device: Target device
            torch_dtype: Data type for weights
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.bottleneck_dim = bottleneck_dim

        # Two-layer bottleneck architecture
        self.compress = nn.Linear(hidden_size, bottleneck_dim, bias=False, dtype=torch_dtype)
        self.generate = nn.Linear(bottleneck_dim, vocab_size, bias=False, dtype=torch_dtype)
        self.dropout = nn.Dropout(0.1)

        # Initialize parameters
        self._initialize_parameters()
        self.to(device)

    def _initialize_parameters(self):
        """Initialize parameters for stable training."""
        with torch.no_grad():
            # Compress layer: Kaiming initialization
            nn.init.kaiming_uniform_(self.compress.weight, a=math.sqrt(5))

            # Generate layer: Small initialization so corrections start small
            # This prevents the model from making drastic changes initially
            nn.init.normal_(self.generate.weight, mean=0.0, std=0.01)

    def count_parameters(self) -> int:
        """Count trainable parameters in the calibration engine."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_engine_info(self) -> Dict[str, Any]:
        """Get detailed information about the calibration engine."""
        param_count = self.count_parameters()
        return {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "bottleneck_dim": self.bottleneck_dim,
            "parameters": param_count,
            "parameters_M": param_count / 1e6,
        }

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate calibration logits from hidden states.

        Args:
            hidden_states: Base model hidden states [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask (unused)

        Returns:
            Dictionary with calibration logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Match data type
        if hidden_states.dtype != self.compress.weight.dtype:
            hidden_states = hidden_states.to(self.compress.weight.dtype)

        # Apply dropout
        contextualized = self.dropout(hidden_states)

        # Compress to bottleneck then expand to vocab
        compressed = self.compress(contextualized)

        # Generate calibration logits
        logit_calibrations = self.generate(compressed)

        return {
            "logit_calibrations": logit_calibrations,
        }


class PALC(nn.Module):
    """
    Main PALC (Preference Alignment via Logit Calibration) model.

    Combines frozen base model with trainable calibration engine.
    Final logits = base_logits + calibration_scale * calibration_logits

    Architecture:
    1. Input -> Base model -> base_logits + hidden_states
    2. hidden_states[last] -> Calibration engine -> calibration_logits
    3. Final: base_logits + scale * calibration_logits

    Benefits:
    - Parameter efficient: ~0.03% trainable parameters
    - Last layer steering for optimal control
    - Scalable intervention strength
    """

    def __init__(self, base_model: BaseLanguageModel, calibration_engine: LogitCalibrationEngine, calibration_scale: float = 5.0):
        """
        Initialize PALC model.

        Args:
            base_model: Frozen pretrained language model
            calibration_engine: Trainable calibration network
            calibration_scale: Scaling factor for calibration logits
        """
        super().__init__()
        self.base_model = base_model
        self.calibration_engine = calibration_engine
        self.device = base_model.device
        self.calibration_scale = calibration_scale

        # Ensure vocabulary sizes match
        assert base_model.vocab_size == calibration_engine.vocab_size, \
            f"Vocabulary mismatch: base={base_model.vocab_size}, engine={calibration_engine.vocab_size}"

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the PALC system."""
        base_info = self.base_model.get_model_info()
        engine_info = self.calibration_engine.get_engine_info()
        return {
            "system": "PALC v2.0",
            "base_model": base_info,
            "calibration_engine": engine_info,
            "total_parameters": base_info["parameters"] + engine_info["parameters"],
            "trainable_parameters": engine_info["parameters"],
            "trainable_percentage": 100 * engine_info["parameters"] / (base_info["parameters"] + engine_info["parameters"]),
            "calibration_scale": self.calibration_scale,
        }

    def get_trainable_parameters(self):
        """Get iterator over trainable parameters (only calibration engine)."""
        return self.calibration_engine.parameters()

    def get_parameter_count(self) -> Dict[str, Any]:
        """Get detailed parameter count breakdown."""
        base_params = self.base_model.count_parameters()
        engine_params = self.calibration_engine.count_parameters()
        return {
            'total': base_params + engine_params,
            'trainable': engine_params,
            'frozen': base_params,
            'trainable_percentage': 100 * engine_params / (base_params + engine_params)
        }

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, return_components=False):
        """
        Forward pass: base_logits + correction_scale * correction_logits

        Process:
        1. Base model: input_ids -> base_logits + hidden_states
        2. Calibration engine: last_hidden -> calibration_logits
        3. Combine: base_logits + scale * correction_logits

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_components: Return (corrected, base, correction) if True

        Returns:
            corrected_logits or tuple if return_components=True
        """
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids, attention_mask, position_ids, past_key_values, use_cache
        )

        # Extract last layer hidden states
        # Use the final layer hidden states for calibration
        last_hidden_states = base_outputs.hidden_states[-1]  # Last layer hidden states
        calibration_outputs = self.calibration_engine(last_hidden_states, attention_mask)

        # Step 3: Scale and add calibration logits to base logits
        logit_calibrations = calibration_outputs["logit_calibrations"]

        # Scale and add calibrations
        scaled_calibrations = logit_calibrations * self.calibration_scale
        calibrated_logits = base_outputs.logits + scaled_calibrations

        if return_components:
            return (
                calibrated_logits,
                base_outputs.logits,
                logit_calibrations
            )
        else:
            return calibrated_logits

    def save_checkpoint(self, path: str, optimizer=None, scheduler=None, global_step=None, args=None):
        """Save complete training checkpoint."""
        checkpoint = {
            'calibration_engine': self.calibration_engine.state_dict(),
            'config': {
                'hidden_size': self.calibration_engine.hidden_size,
                'vocab_size': self.calibration_engine.vocab_size,
                'bottleneck_dim': self.calibration_engine.bottleneck_dim,
            }
        }

        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler'] = scheduler.state_dict()
        if global_step is not None:
            checkpoint['global_step'] = global_step
        if args is not None:
            checkpoint['args'] = args

        torch.save(checkpoint, path)

    def save_weights(self, path: str):
        """Save only model weights (for final/best model)."""
        state_dict = {
            'calibration_engine': self.calibration_engine.state_dict(),
            'config': {
                'hidden_size': self.calibration_engine.hidden_size,
                'vocab_size': self.calibration_engine.vocab_size,
                'bottleneck_dim': self.calibration_engine.bottleneck_dim,
            }
        }
        torch.save(state_dict, path)

    def load_checkpoint(self, path: str, optimizer=None, scheduler=None, map_location=None):
        """Load complete training checkpoint."""
        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(path, map_location=map_location)

        # Load model weights
        if 'calibration_engine' in checkpoint:
            self.calibration_engine.load_state_dict(checkpoint['calibration_engine'])
        elif 'correction_engine' in checkpoint:  # Backward compatibility
            self.calibration_engine.load_state_dict(checkpoint['correction_engine'])
        else:
            self.calibration_engine.load_state_dict(checkpoint)

        # Load training states
        states_loaded = {}
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            states_loaded['optimizer'] = True

        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            states_loaded['scheduler'] = True

        global_step = checkpoint.get('global_step', 0)
        args = checkpoint.get('args', None)

        return global_step, args, states_loaded

    def load_weights(self, path: str):
        """Load only model weights."""
        state = torch.load(path, map_location=self.device)
        if 'calibration_engine' in state:
            self.calibration_engine.load_state_dict(state['calibration_engine'])
        elif 'correction_engine' in state:  # Backward compatibility
            self.calibration_engine.load_state_dict(state['correction_engine'])
        else:
            self.calibration_engine.load_state_dict(state)


def create_palc_model(model_name: str = "argsearch/llama-7b-sft-float32",
                      device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16,
                      bottleneck: int = 32) -> PALC:
    """
    Create complete PALC model for training or inference.

    Creates frozen base model + trainable calibration engine.
    Uses calibration_scale=1.0 for training (adjustable later for inference).
    No reference model needed for SimplePreferenceLoss.

    Args:
        model_name: HuggingFace model identifier
        device: Target device ("cuda" or "cpu")
        torch_dtype: Model precision
        bottleneck: Calibration engine bottleneck dimension

    Returns:
        PALC model with calibration_scale=1.0
    """

    base_model = BaseLanguageModel(model_name, device, torch_dtype)
    calibration_engine = LogitCalibrationEngine(
        hidden_size=base_model.hidden_size,
        vocab_size=base_model.vocab_size,
        bottleneck_dim=bottleneck,
        device=device,
        torch_dtype=torch_dtype
    )

    palc = PALC(base_model, calibration_engine, calibration_scale=1.0)
    return palc


def create_palc_for_inference(model_name: str = "argsearch/llama-7b-sft-float32",
                              device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16,
                              bottleneck: int = 32, calibration_scale: float = 5.0) -> PALC:
    """
    Create PALC model for inference (no reference model).

    Memory-efficient version without DPO reference model.

    Args:
        model_name: HuggingFace model identifier
        device: Target device
        torch_dtype: Model precision
        bottleneck: Calibration engine bottleneck dimension
        calibration_scale: Scaling factor for calibrations

    Returns:
        Memory-efficient PALC model
    """
    model = create_palc_model(model_name, device, torch_dtype, bottleneck)
    model.calibration_scale = calibration_scale  # Set inference scale
    return model