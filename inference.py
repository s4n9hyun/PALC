"""
PALC Inference System

This module provides comprehensive text generation capabilities for PALC models,
including advanced sampling methods, debugging tools, and analysis functions.

Key features:
- Multiple sampling strategies (greedy, temperature, nucleus, top-k)
- Debug mode with detailed logit/probability analysis
- Base vs PALC comparison tools
- Model analysis and statistics

Usage:
    # Basic generation
    palc_inference = create_inference_wrapper(model_path, device="cuda")
    response = palc_inference.generate("Hello, how are you?")

    # Debug generation
    response, debug_info = palc_inference.generate(prompt, return_debug_info=True)

    # Analysis
    analysis = palc_inference.analyze_calibrations(prompt)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, List, Tuple
from .model import create_palc_for_inference, PALC
import re




def generate(model: PALC, tokenizer: AutoTokenizer, prompt: str,
             max_tokens: int = 50, device: str = "cuda",
             return_debug_info: bool = False,
             temperature: float = 1.0,
             top_p: Optional[float] = None,
             top_k: Optional[int] = None,
             clean_output: bool = False) -> tuple:
    """
    Advanced text generation with PALC model.

    Supports multiple sampling strategies and optional debugging information.
    This is the core generation function used by the PalcInference wrapper.

    Sampling strategies:
    - Greedy (temperature=1.0, no top_p/top_k): Deterministic, picks highest probability
    - Temperature sampling: Controls randomness (>1.0 = more random, <1.0 = more focused)
    - Nucleus (top_p): Only consider tokens in top p% of probability mass
    - Top-k: Only consider top k most likely tokens

    Args:
        model: PALC model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        device: Device to run inference on ("cuda" or "cpu")
        return_debug_info: If True, return detailed per-token analysis
        temperature: Sampling temperature (1.0=greedy, >1.0=random, <1.0=focused)
        top_p: Nucleus sampling threshold (0.0-1.0), None to disable
        top_k: Top-k sampling threshold, None to disable

    Returns:
        If return_debug_info=False:
            (full_response, generated_part): Complete text and newly generated part
        If return_debug_info=True:
            (full_response, generated_part, debug_info): Plus per-token analysis data
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    original_length = inputs['input_ids'].shape[1]

    debug_info = [] if return_debug_info else None

    with torch.no_grad():
        for step in range(max_tokens):
            if return_debug_info:
                calibrated_logits, base_logits, calibration_logits = model(
                    inputs['input_ids'],
                    inputs.get('attention_mask'),
                    return_components=True
                )
            else:
                calibrated_logits = model(inputs['input_ids'], inputs.get('attention_mask'))

            # Apply temperature (prevent division by zero)
            if temperature <= 0:
                # Use greedy decoding for temperature=0
                logits = calibrated_logits[:, -1, :]
            else:
                logits = calibrated_logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            if (temperature <= 0 or temperature == 1.0) and top_p is None and top_k is None:
                # Greedy decoding for temperature=0 or temperature=1.0 with no sampling
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Sampling
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Store debug info
            if return_debug_info:
                selected_token_id = next_token.item()
                token_text = tokenizer.decode(selected_token_id, skip_special_tokens=True)

                debug_info.append({
                    "step": step,
                    "token": token_text,
                    "token_id": selected_token_id,
                    "base_logit": base_logits[0, -1, selected_token_id].item(),
                    "calibration_logit": calibration_logits[0, -1, selected_token_id].item(),
                    "final_logit": calibrated_logits[0, -1, selected_token_id].item(),
                    "base_prob": F.softmax(base_logits[0, -1, :], dim=-1)[selected_token_id].item(),
                    "calibration_prob": F.softmax(calibration_logits[0, -1, :], dim=-1)[selected_token_id].item(),
                    "final_prob": F.softmax(calibrated_logits[0, -1, :], dim=-1)[selected_token_id].item(),
                })

            # Update inputs
            inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=-1)

            if 'attention_mask' in inputs:
                attention_ones = torch.ones((inputs['attention_mask'].shape[0], 1),
                                          dtype=inputs['attention_mask'].dtype,
                                          device=inputs['attention_mask'].device)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], attention_ones], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    full_response = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    generated_part = tokenizer.decode(inputs['input_ids'][0][original_length:], skip_special_tokens=True)

    # Apply cleaning if requested (DEPRECATED)
    if clean_output:
        # clean_response function removed for academic honesty
        # Return raw output regardless of clean_output setting
        pass

    if return_debug_info:
        return full_response, generated_part, debug_info
    else:
        return full_response, generated_part


class PalcInference:
    """
    High-level wrapper for PALC inference with enhanced functionality.

    This class provides a user-friendly interface for PALC text generation,
    along with powerful analysis and debugging tools. It handles model
    loading, tokenization, and provides convenient methods for different
    types of inference tasks.

    Key features:
    - Simple text generation with multiple sampling strategies
    - Detailed calibration analysis and debugging
    - Base model vs PALC comparison
    - Model statistics and information
    """

    def __init__(self, model: PALC, tokenizer: AutoTokenizer, device: str = "cuda"):
        """
        Initialize the inference wrapper.

        Args:
            model: Loaded PALC model
            tokenizer: HuggingFace tokenizer matching the base model
            device: Device for inference ("cuda" or "cpu")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()  # Ensure model is in evaluation mode

    def generate_raw(self, prompt: str, max_tokens: int = 50, **kwargs):
        """
        Generate raw text with PALC - NO post-processing applied.

        This method returns the model's actual output without any cleaning or
        post-processing. Use this for academic evaluation and fair comparison.

        Args:
            prompt: Input text
            max_tokens: How many tokens to generate
            **kwargs: Optional sampling parameters:
                - temperature: float = 1.0 (1.0=greedy, >1.0=random)
                - top_p: float = None (nucleus sampling)
                - top_k: int = None (top-k sampling)
                - return_debug_info: bool = False (detailed analysis)

        Returns:
            (full_text, generated_text) or (full_text, generated_text, debug_info)

        Examples:
            # Raw generation for evaluation
            full, generated = palc.generate_raw("Hello")

            # With debugging
            full, generated, debug = palc.generate_raw("Hello", return_debug_info=True)
        """
        kwargs['clean_output'] = False  # Force raw output
        return generate(self.model, self.tokenizer, prompt,
                       max_tokens=max_tokens, device=self.device, **kwargs)

    def generate(self, prompt: str, max_tokens: int = 50, **kwargs):
        """
        Generate text with PALC - defaults to RAW output for academic honesty.

        Args:
            prompt: Input text
            max_tokens: How many tokens to generate
            **kwargs: Optional sampling parameters:
                - temperature: float = 1.0 (1.0=greedy, >1.0=random)
                - top_p: float = None (nucleus sampling)
                - top_k: int = None (top-k sampling)
                - return_debug_info: bool = False (detailed analysis)
                - clean_output: bool = False (DEPRECATED - use for legacy compatibility only)

        Returns:
            (full_text, generated_text) or (full_text, generated_text, debug_info)

        Examples:
            # Raw generation (default)
            full, generated = palc.generate("Hello")

            # With sampling
            full, generated = palc.generate("Hello", temperature=1.2, top_p=0.9)

            # With debugging
            full, generated, debug = palc.generate("Hello", return_debug_info=True)
        """
        # Default to raw output for academic honesty
        if 'clean_output' not in kwargs:
            kwargs['clean_output'] = False
        return generate(self.model, self.tokenizer, prompt,
                       max_tokens=max_tokens, device=self.device, **kwargs)

    def chat(self, message: str, max_tokens: int = 128):
        """Simple chat interface - returns RAW response text."""
        prompt = f"Human: {message}\n\nAssistant:"
        full_response, generated_part = self.generate_raw(prompt, max_tokens=max_tokens)
        return generated_part.strip()

    def debug(self, prompt: str, max_tokens: int = 20):
        """Quick debug - shows calibration analysis."""
        return self.analyze_calibrations(prompt, max_tokens)

    def analyze_calibrations(self, prompt: str, max_tokens: int = 20) -> Dict[str, Any]:
        """Analyze the calibration patterns for a given prompt.

        Returns detailed information about how calibrations affect generation.
        """
        full_response, generated_part, debug_info = self.generate_raw(
            prompt, max_tokens=max_tokens, return_debug_info=True
        )

        if not debug_info:
            return {"error": "No debug info available"}

        # Calculate statistics
        base_logits = [step["base_logit"] for step in debug_info]
        calibration_logits = [step["calibration_logit"] for step in debug_info]

        import numpy as np

        analysis = {
            "prompt": prompt,
            "generated_text": generated_part,
            "token_count": len(debug_info),
            "base_logit_stats": {
                "mean": float(np.mean(base_logits)),
                "std": float(np.std(base_logits)),
                "min": float(np.min(base_logits)),
                "max": float(np.max(base_logits))
            },
            "calibration_logit_stats": {
                "mean": float(np.mean(calibration_logits)),
                "std": float(np.std(calibration_logits)),
                "min": float(np.min(calibration_logits)),
                "max": float(np.max(calibration_logits))
            },
            "magnitude_ratio": float(np.mean(np.abs(base_logits)) / np.mean(np.abs(calibration_logits))) if np.mean(np.abs(calibration_logits)) > 0 else float('inf'),
            "calibration_scale": self.model.calibration_scale,
            "token_details": debug_info
        }

        return analysis

    def compare_with_base(self, prompt: str, max_tokens: int = 20) -> Dict[str, Any]:
        """Compare PALC output with base model output."""
        # Generate with PALC
        palc_response, palc_generated, palc_debug = self.generate_raw(
            prompt, max_tokens=max_tokens, return_debug_info=True
        )

        # Generate with base model only (temporarily set calibration_scale to 0)
        original_scale = self.model.calibration_scale
        self.model.calibration_scale = 0.0

        base_response, base_generated = self.generate_raw(prompt, max_tokens=max_tokens)

        # Restore original scale
        self.model.calibration_scale = original_scale

        return {
            "prompt": prompt,
            "palc_output": palc_generated,
            "base_output": base_generated,
            "same_output": palc_generated.strip() == base_generated.strip(),
            "calibration_scale": original_scale,
            "debug_info": palc_debug
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the PALC model."""
        return {
            "model_info": self.model.get_system_info(),
            "calibration_scale": self.model.calibration_scale,
            "device": self.device,
            "vocab_size": self.tokenizer.vocab_size
        }


def load_palc(model_path: str, model_name: str = "argsearch/llama-7b-sft-float32",
              device: str = "cuda", bottleneck: int = 64, calibration_scale: float = 5.0) -> PalcInference:
    """
    Load PALC model for inference - simplified interface.

    Args:
        model_path: Path to PALC checkpoint (.pt file)
        model_name: Base model name (e.g., "argsearch/llama-7b-sft-float32")
        device: Device to run on ("cuda" or "cpu")
        bottleneck: Bottleneck dimension (must match training)
        calibration_scale: Scaling factor for calibrations

    Returns:
        Ready-to-use PALC inference object

    Example:
        palc = load_palc("path/to/checkpoint.pt")
        response = palc.generate("Hello, how are you?")
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create and load model
    model = create_palc_for_inference(
        model_name=model_name,
        device=device,
        torch_dtype=torch.bfloat16,
        bottleneck=bottleneck,
        calibration_scale=calibration_scale
    )
    model.load_weights(model_path)

    return PalcInference(model, tokenizer, device)

# Keep old function for backward compatibility
create_inference_wrapper = load_palc
load_loma = load_palc  # Backward compatibility alias