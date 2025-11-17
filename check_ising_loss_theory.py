"""
Simple script to show theoretical ising loss values for validation temperatures.

The ising loss in isingPixelCNN.py is computed as:
    loss = log p_model(x) + E(x)/T

The expected value of this loss should be:
    E[loss] = E[log p_model(x) + beta * E(x)] = -log Z

where Z is the true partition function of the Ising model.

This script calculates:
1. True log Z for each validation temperature
2. Theoretical loss value (-log Z)
3. Loss per site (-log Z / L^2)
"""

import torch
import numpy as np
from utils import isingLogzTr

# Parameters matching isingPixelCNN.py validation
L = 16  # Lattice length
T0 = 2.269  # base T for Ising dist
factorLst = [0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.8]  # multiple factors for T0
TList = [term * T0 for term in factorLst]

print("=" * 80)
print("Theoretical Ising Loss Values - Validation Range")
print("=" * 80)
print(f"Lattice size: {L}x{L} = {L**2} spins")
print(f"Base temperature T0: {T0}")
print(f"Number of validation points: {len(TList)}")
print("=" * 80)
print()

# Calculate theoretical values
print(f"{'Factor':>8} {'T':>10} {'Beta':>10} {'log Z':>15} {'-log Z':>15} {'-log Z / L^2':>15}")
print("-" * 90)

results = []
for idx, (factor, T) in enumerate(zip(factorLst, TList)):
    beta = 1.0 / T
    beta_tensor = torch.tensor(beta)

    # Calculate true log partition function
    log_Z = isingLogzTr(n=L, j=1.0, beta=beta_tensor).item()

    # Theoretical loss = -log Z
    theoretical_loss = -log_Z

    # Loss per site (this is what's reported in isingPixelCNN.py line 168)
    loss_per_site = theoretical_loss / (L ** 2)

    results.append({
        'factor': factor,
        'T': T,
        'beta': beta,
        'log_Z': log_Z,
        'theoretical_loss': theoretical_loss,
        'loss_per_site': loss_per_site
    })

    print(f"{factor:8.1f} {T:10.4f} {beta:10.6f} {log_Z:15.6f} {theoretical_loss:15.6f} {loss_per_site:15.6f}")

print("=" * 90)
print()

# Show what should be reported in training
print("Expected Training Output (matching isingPixelCNN.py format):")
print("-" * 90)
print("In isingPixelCNN.py line 165, the code divides by 16*16*0.45.")
print("Note: The 0.45 factor seems arbitrary and doesn't match theory.")
print()
print(f"{'Factor':>8} {'Reported loss':>15} {'True -log Z':>15} {'Error (should→0)':>18}")
print("-" * 90)
for r in results:
    # This matches line 168: lossLst[idx].item()/16/16/0.45
    # But the theoretical value should just be -log Z, not divided by any constant
    reported = r['loss_per_site'] / 0.45
    true_neg_log_z = r['theoretical_loss']
    print(f"{r['factor']:8.1f} {reported:15.6f} {true_neg_log_z:15.6f} {'loss + log_Z':>18}")

print("=" * 90)
print()
print("Summary:")
print("-" * 90)
print("The theoretical loss = -log Z for each temperature.")
print("In training (isingPixelCNN.py:165), error is computed as:")
print("  error = lossLst[idx].item() + isingExactloss[idx]")
print("  where isingExactloss[idx] = isingLogzTr(...) = log Z")
print()
print("So the error should be:")
print("  error = estimated_loss + log_Z → 0 as training converges")
print("=" * 90)
