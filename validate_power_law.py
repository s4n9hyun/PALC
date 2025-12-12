#!/usr/bin/env python3
"""
Script to validate the power-law assumption (Eq 12) for ICLR 2026 rebuttal.
Performs SVD on the bottleneck composite matrix M = W_up × W_down and plots
singular values vs. rank on a log-log scale.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================
checkpoint_path = '/Users/huios/palc/research/palc/outputs/main_run_20250922_005203/final_palc.pt'

# =============================================================================
# Step 1: Load checkpoint and inspect keys
# =============================================================================
print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*80)
print("CHECKPOINT KEYS (top-level):")
print("="*80)
for key in checkpoint.keys():
    print(f"  - {key}")
print("="*80 + "\n")

# If the checkpoint has nested structure, let's also check the model state dict
if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
    state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
    print(f"\nKeys inside '{state_dict_key}':")
    print("="*80)
    for key in checkpoint[state_dict_key].keys():
        print(f"  - {key}")
    print("="*80 + "\n")

# =============================================================================
# Step 2: Define weight keys (EDIT THESE AFTER INSPECTING OUTPUT ABOVE)
# =============================================================================
W_DOWN_KEY = 'model.calibration_module.w_down.weight'  # Edit this based on printed keys
W_UP_KEY = 'model.calibration_module.w_up.weight'      # Edit this based on printed keys

# =============================================================================
# Step 3: Extract weights from checkpoint
# =============================================================================
print("Extracting weights from checkpoint...")

# Try to extract from common checkpoint structures
if W_DOWN_KEY in checkpoint:
    W_down = checkpoint[W_DOWN_KEY]
    W_up = checkpoint[W_UP_KEY]
elif 'model_state_dict' in checkpoint and W_DOWN_KEY in checkpoint['model_state_dict']:
    W_down = checkpoint['model_state_dict'][W_DOWN_KEY]
    W_up = checkpoint['model_state_dict'][W_UP_KEY]
elif 'state_dict' in checkpoint and W_DOWN_KEY in checkpoint['state_dict']:
    W_down = checkpoint['state_dict'][W_DOWN_KEY]
    W_up = checkpoint['state_dict'][W_UP_KEY]
else:
    raise KeyError(f"Could not find weights with keys: {W_DOWN_KEY}, {W_UP_KEY}")

print(f"W_down shape: {W_down.shape}")
print(f"W_up shape: {W_up.shape}")

# =============================================================================
# Step 4: Calculate composite matrix M = W_up × W_down
# =============================================================================
print("\nCalculating composite matrix M = W_up × W_down...")
M = torch.matmul(W_up, W_down)
print(f"M shape: {M.shape}")

# =============================================================================
# Step 5: Perform SVD
# =============================================================================
print("\nPerforming SVD...")
U, S, Vh = torch.linalg.svd(M, full_matrices=False)
print(f"Number of singular values: {len(S)}")
print(f"Largest singular value: {S[0].item():.6f}")
print(f"Smallest singular value: {S[-1].item():.6e}")

# =============================================================================
# Step 6: Sort singular values (should already be sorted, but ensure it)
# =============================================================================
S_sorted = torch.sort(S, descending=True)[0]
S_numpy = S_sorted.cpu().numpy()

# =============================================================================
# Step 7: Create log-log plot
# =============================================================================
print("\nGenerating log-log plot...")
ranks = np.arange(1, len(S_numpy) + 1)

plt.figure(figsize=(10, 7))
plt.loglog(ranks, S_numpy, 'b-', linewidth=2, marker='o', markersize=3, alpha=0.7)
plt.title("SVD of Bottleneck Matrix (B=256) - Log-Log Scale", fontsize=14, fontweight='bold')
plt.xlabel("Rank (i)", fontsize=12)
plt.ylabel("Singular Value (σ_i)", fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

# Save the plot
output_path = 'rebuttal_svd_power_law.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_path}")

# Also save as PDF for publication quality
output_path_pdf = 'rebuttal_svd_power_law.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"✓ Plot also saved as PDF: {output_path_pdf}")

plt.show()

print("\n" + "="*80)
print("DONE! Check the plots to validate the power-law assumption.")
print("Expected: straight line on log-log scale indicates power-law decay.")
print("="*80)
