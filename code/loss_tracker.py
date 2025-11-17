#!/usr/bin/env python3
"""
Loss Tracker for TP/FP Training Loss Visualization
Reproduces Figure 3 from "Denoising Implicit Feedback for Recommendation"
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

class LossTracker:
    """
    Tracks training loss for TP and FP interactions separately
    """
    def __init__(self):
        self.history = {
            'iteration': [],
            'tp_loss': [],
            'fp_loss': [],
            'mean_loss': []
        }

    def record(self, iteration, tp_loss, fp_loss, mean_loss):
        """Record loss values for current iteration"""
        self.history['iteration'].append(iteration)
        self.history['tp_loss'].append(tp_loss)
        self.history['fp_loss'].append(fp_loss)
        self.history['mean_loss'].append(mean_loss)

    def get_history(self):
        """Return loss history as numpy arrays"""
        return {
            'iteration': np.array(self.history['iteration']),
            'tp_loss': np.array(self.history['tp_loss']),
            'fp_loss': np.array(self.history['fp_loss']),
            'mean_loss': np.array(self.history['mean_loss'])
        }

    def save(self, filepath):
        """Save tracking data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f)

    def load(self, filepath):
        """Load tracking data from JSON file"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)

    def smooth(self, values, window_size=50):
        """Apply moving average smoothing"""
        if len(values) < window_size:
            return values

        weights = np.ones(window_size) / window_size
        return np.convolve(values, weights, mode='valid')

    def plot(self, output_path=None, smooth_window=50, max_iter_full=None, max_iter_early=1000):
        """
        Generate Figure 3 style visualization

        Args:
            output_path: Path to save the figure
            smooth_window: Window size for smoothing (set to 1 for no smoothing)
            max_iter_full: Maximum iteration for full plot (None = all)
            max_iter_early: Maximum iteration for early stage plot
        """
        history = self.get_history()

        # Check if we have enough data
        if len(history['iteration']) < 10:
            print(f"⚠️  Warning: Only {len(history['iteration'])} iterations tracked. Need at least 10 for meaningful visualization.")
            print("Skipping plot generation.")
            return None

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Determine iteration range for full plot
        if max_iter_full is None:
            max_iter_full = history['iteration'][-1]

        # Filter data for full plot
        mask_full = history['iteration'] <= max_iter_full
        iter_full = history['iteration'][mask_full]

        # Adaptive smoothing window (limit to at most 1/10 of data length)
        effective_smooth_window = min(smooth_window, max(1, len(iter_full) // 10))

        # Apply smoothing if needed
        if effective_smooth_window > 1 and len(iter_full) > effective_smooth_window:
            tp_full = self.smooth(history['tp_loss'][mask_full], effective_smooth_window)
            fp_full = self.smooth(history['fp_loss'][mask_full], effective_smooth_window)
            mean_full = self.smooth(history['mean_loss'][mask_full], effective_smooth_window)
            # Adjust iteration array for smoothed data
            iter_full_smooth = iter_full[effective_smooth_window-1:]
        else:
            tp_full = history['tp_loss'][mask_full]
            fp_full = history['fp_loss'][mask_full]
            mean_full = history['mean_loss'][mask_full]
            iter_full_smooth = iter_full

        # Plot (a): Whole training process
        ax1.plot(iter_full_smooth, mean_full, color='#5B8DBE', linewidth=2,
                label='Mean of positive interactions', alpha=0.8)
        ax1.plot(iter_full_smooth, fp_full, color='#E07B39', linewidth=2,
                label='False-positive interactions', alpha=0.7)
        ax1.plot(iter_full_smooth, tp_full, color='#64A338', linewidth=2,
                label='True-positive interactions', alpha=0.7)

        # Add annotations for subplot (a)
        # Large-loss samples annotation (early stage)
        ax1.text(100, 0.28, 'Large-loss samples', fontsize=12,
                color='#E07B39', weight='bold')

        # All memorized annotation (late stage)
        memorized_iter = int(max_iter_full * 0.7)
        ax1.text(memorized_iter, 0.14, 'All memorized', fontsize=12,
                color='#5B8DBE', weight='bold')

        ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax1.set_title('(a) Whole training process', fontsize=16, fontweight='bold', pad=15)
        ax1.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 0.7)

        # Filter data for early stage plot
        mask_early = history['iteration'] <= max_iter_early
        iter_early = history['iteration'][mask_early]

        # Adaptive smoothing for early stage
        effective_smooth_window_early = min(smooth_window, max(1, len(iter_early) // 10))

        # Apply smoothing for early stage
        if effective_smooth_window_early > 1 and len(iter_early) > effective_smooth_window_early:
            tp_early = self.smooth(history['tp_loss'][mask_early], effective_smooth_window_early)
            fp_early = self.smooth(history['fp_loss'][mask_early], effective_smooth_window_early)
            mean_early = self.smooth(history['mean_loss'][mask_early], effective_smooth_window_early)
            iter_early_smooth = iter_early[effective_smooth_window_early-1:]
        else:
            tp_early = history['tp_loss'][mask_early]
            fp_early = history['fp_loss'][mask_early]
            mean_early = history['mean_loss'][mask_early]
            iter_early_smooth = iter_early

        # Plot (b): Early training stages
        ax2.plot(iter_early_smooth, mean_early, color='#5B8DBE', linewidth=2,
                label='Mean of positive interactions', alpha=0.8)
        ax2.plot(iter_early_smooth, fp_early, color='#E07B39', linewidth=2,
                label='False-positive interactions', alpha=0.7)
        ax2.plot(iter_early_smooth, tp_early, color='#64A338', linewidth=2,
                label='True-positive interactions', alpha=0.7)

        # Add annotations for subplot (b)
        # Large-loss samples annotation
        ax2.text(100, 0.30, 'Large-loss samples', fontsize=12,
                color='#E07B39', weight='bold')

        # Small-loss samples annotation
        small_loss_iter = int(max_iter_early * 0.7)
        ax2.text(small_loss_iter, 0.05, 'Small-loss samples', fontsize=12,
                color='#64A338', weight='bold')

        ax2.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax2.set_title('(b) Early training stages', fontsize=16, fontweight='bold', pad=15)
        ax2.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 0.7)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure saved to: {output_path}")

        return fig


def moving_average(data, window_size):
    """Simple moving average for smoothing"""
    if len(data) < window_size:
        return data
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')
