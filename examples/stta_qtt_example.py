"""
Example: STTA-to-QTT Conversion.
Demonstrates how to use Streaming Tensor Train Approximation to build 
a Quantized Tensor Train (QTT) representation of a high-dimensional function.
"""

import numpy as np
import tinytt as tt
from tinytt.streaming import StreamingTT
import tinytt._backend as tn

def f(x):
    """A sample function to approximate."""
    return np.exp(-x**2) * np.sin(5 * x)

def run_stta_qtt_demo():
    print("--- STTA-to-QTT Demo ---")
    
    # Parameters for QTT
    # We want to represent a 1D function on a grid of 2^L points.
    L = 10 
    grid_size = 2**L
    shape_qtt = [2] * L
    target_rank = 4
    
    print(f"Targeting a grid of {grid_size} points using {L} QTT modes.")
    
    # Initialize StreamingTT with the QTT shape
    stt = StreamingTT(shape=shape_qtt, ranks=target_rank, oversampling=5)
    
    # Define the grid on [0, 1]
    x_grid = np.linspace(0, 1, grid_size)
    
    # In tinyTT's binary representation, the 'last' axis of the shape [2, 2, ..., 2]
    # corresponds to the least significant bit (finest division).
    # This means index=0 and index=1 along the last axis correspond to EVEN and ODD points.
    # By streaming these interleaved sets, we build the QTT in one pass.
    
    print("Streaming EVEN points (index=0 along last axis)...")
    y_even = f(x_grid[0::2])
    stt.update(y_even.reshape([2]*(L-1)), index=0, axis=-1)
    
    print("Streaming ODD points (index=1 along last axis)...")
    y_odd = f(x_grid[1::2])
    stt.update(y_odd.reshape([2]*(L-1)), index=1, axis=-1)
    
    # Finalize the QTT approximation
    print("Finalizing STTA...")
    qtt_obj = stt.finalize()
    
    print(f"QTT Construction Complete. Ranks: {qtt_obj.R}")
    
    # Verification
    y_full = f(x_grid)
    y_approx = qtt_obj.full().numpy().flatten()
    
    rel_error = np.linalg.norm(y_full - y_approx) / np.linalg.norm(y_full)
    print(f"Relative Reconstruction Error: {rel_error:.2e}")
    
    # Show some values
    print("\nComparison at selected points:")
    indices = [0, grid_size//4, grid_size//2, 3*grid_size//4, -1]
    print(f"{'Index':>8} | {'True':>10} | {'Approx':>10} | {'Error':>10}")
    for idx in indices:
        t_val = y_full[idx]
        a_val = y_approx[idx]
        print(f"{idx:>8} | {t_val:10.6f} | {a_val:10.6f} | {abs(t_val-a_val):10.2e}")

if __name__ == "__main__":
    run_stta_qtt_demo()
