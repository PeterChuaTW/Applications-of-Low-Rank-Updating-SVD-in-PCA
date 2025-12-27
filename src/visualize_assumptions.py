"""
Visualization Tools for Noise Analysis and Assumption Validation.

This module provides plotting functions to visually assess whether
residuals satisfy the Gaussian white noise assumptions required by
the Gavish-Donoho optimal threshold method.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def plot_qq_plot(residuals_flat, save_path=None):
    """
    Create a Q-Q (Quantile-Quantile) plot to assess normality.
    
    A Q-Q plot compares the quantiles of the data against theoretical
    quantiles from a normal distribution. If data is normal, points
    should lie approximately on the diagonal line.
    
    Parameters:
    -----------
    residuals_flat : array, shape (n,)
        Flattened residual values
    save_path : str, optional
        Path to save the figure. If None, displays interactively.
    """
    # Remove NaN/Inf
    residuals_clean = residuals_flat[np.isfinite(residuals_flat)]
    
    # Sample if too large for performance
    if len(residuals_clean) > 10000:
        residuals_sample = np.random.choice(residuals_clean, 10000, replace=False)
    else:
        residuals_sample = residuals_clean
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Generate Q-Q plot
    stats.probplot(residuals_sample, dist="norm", plot=ax)
    
    ax.set_title('Q-Q Plot: Residuals vs Normal Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Sample Quantiles', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add text annotation
    text = f'n = {len(residuals_sample):,}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved Q-Q plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_histogram(residuals_flat, save_path=None):
    """
    Plot histogram of residuals with fitted normal distribution overlay.
    
    Parameters:
    -----------
    residuals_flat : array, shape (n,)
        Flattened residual values
    save_path : str, optional
        Path to save the figure. If None, displays interactively.
    """
    # Remove NaN/Inf
    residuals_clean = residuals_flat[np.isfinite(residuals_flat)]
    
    # Sample if too large
    if len(residuals_clean) > 50000:
        residuals_sample = np.random.choice(residuals_clean, 50000, replace=False)
    else:
        residuals_sample = residuals_clean
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    n_bins = min(100, int(np.sqrt(len(residuals_sample))))
    counts, bins, patches = ax.hist(residuals_sample, bins=n_bins, 
                                      density=True, alpha=0.7, 
                                      color='skyblue', edgecolor='black')
    
    # Fit and plot normal distribution
    mu, sigma = np.mean(residuals_sample), np.std(residuals_sample)
    x = np.linspace(residuals_sample.min(), residuals_sample.max(), 1000)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
            label=f'Normal fit\n($\mu={mu:.4f}$, $\sigma={sigma:.4f}$)')
    
    ax.set_title('Histogram of Residuals with Normal Distribution Overlay', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Residual Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics box
    skew = stats.skew(residuals_sample)
    kurt = stats.kurtosis(residuals_sample)
    stats_text = f'n = {len(residuals_sample):,}\nSkewness = {skew:.4f}\nKurtosis = {kurt:.4f}'
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved histogram to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_acf(residuals_flat, max_lags=50, save_path=None):
    """
    Plot Autocorrelation Function (ACF) to assess independence.
    
    Parameters:
    -----------
    residuals_flat : array, shape (n,)
        Flattened residual values
    max_lags : int, default=50
        Maximum number of lags to plot
    save_path : str, optional
        Path to save the figure. If None, displays interactively.
    """
    # Remove NaN/Inf
    residuals_clean = residuals_flat[np.isfinite(residuals_flat)]
    
    # Sample if too large
    if len(residuals_clean) > 10000:
        residuals_sample = np.random.choice(residuals_clean, 10000, replace=False)
    else:
        residuals_sample = residuals_clean
    
    # Compute ACF manually
    mean = np.mean(residuals_sample)
    var = np.var(residuals_sample)
    
    acf = np.zeros(max_lags + 1)
    acf[0] = 1.0
    
    for lag in range(1, max_lags + 1):
        if len(residuals_sample) > lag:
            c = np.mean((residuals_sample[:-lag] - mean) * (residuals_sample[lag:] - mean))
            acf[lag] = c / var if var > 0 else 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot ACF as stem plot
    lags = np.arange(max_lags + 1)
    markerline, stemlines, baseline = ax.stem(lags, acf, basefmt=' ')
    plt.setp(markerline, 'markerfacecolor', 'blue', 'markersize', 5)
    plt.setp(stemlines, 'color', 'blue', 'linewidth', 1.5)
    
    # Add 95% confidence interval
    n = len(residuals_sample)
    conf_bound = 1.96 / np.sqrt(n)
    ax.axhline(y=conf_bound, color='red', linestyle='--', linewidth=1.5, 
               label=f'95% CI: ±{conf_bound:.4f}')
    ax.axhline(y=-conf_bound, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_title('Autocorrelation Function (ACF) of Residuals', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_xlim(-1, max_lags + 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Count significant lags
    significant = np.sum(np.abs(acf[1:]) > conf_bound)
    text = f'n = {len(residuals_sample):,}\nSignificant lags: {significant}/{max_lags}'
    ax.text(0.98, 0.95, text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved ACF plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_residual_diagnostics(residuals, residuals_flat, max_lags=50, save_dir='output'):
    """
    Create a comprehensive 4-panel diagnostic plot.
    
    Panels:
    1. Q-Q plot (normality check)
    2. Histogram with normal overlay
    3. ACF plot (independence check)
    4. Residual heatmap (first 20x20 residual matrix)
    
    Parameters:
    -----------
    residuals : array, shape (m, n)
        2D residual matrix
    residuals_flat : array, shape (m*n,)
        Flattened residuals
    max_lags : int, default=50
        Maximum lags for ACF plot
    save_dir : str, default='output'
        Directory to save the figure
    """
    # Create output directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Remove NaN/Inf
    residuals_clean = residuals_flat[np.isfinite(residuals_flat)]
    
    # Sample for performance
    if len(residuals_clean) > 10000:
        residuals_sample = np.random.choice(residuals_clean, 10000, replace=False)
    else:
        residuals_sample = residuals_clean
    
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Q-Q Plot
    ax1 = plt.subplot(2, 2, 1)
    stats.probplot(residuals_sample, dist="norm", plot=ax1)
    ax1.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Histogram
    ax2 = plt.subplot(2, 2, 2)
    n_bins = min(50, int(np.sqrt(len(residuals_sample))))
    ax2.hist(residuals_sample, bins=n_bins, density=True, 
             alpha=0.7, color='skyblue', edgecolor='black')
    
    mu, sigma = np.mean(residuals_sample), np.std(residuals_sample)
    x = np.linspace(residuals_sample.min(), residuals_sample.max(), 1000)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
             label=f'Normal($\mu={mu:.3f}$, $\sigma={sigma:.3f}$)')
    
    ax2.set_title('Histogram with Normal Fit', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: ACF Plot
    ax3 = plt.subplot(2, 2, 3)
    
    mean = np.mean(residuals_sample)
    var = np.var(residuals_sample)
    acf = np.zeros(max_lags + 1)
    acf[0] = 1.0
    
    for lag in range(1, max_lags + 1):
        if len(residuals_sample) > lag:
            c = np.mean((residuals_sample[:-lag] - mean) * (residuals_sample[lag:] - mean))
            acf[lag] = c / var if var > 0 else 0
    
    lags = np.arange(max_lags + 1)
    markerline, stemlines, baseline = ax3.stem(lags, acf, basefmt=' ')
    plt.setp(markerline, 'markerfacecolor', 'blue', 'markersize', 4)
    plt.setp(stemlines, 'color', 'blue', 'linewidth', 1)
    
    n = len(residuals_sample)
    conf_bound = 1.96 / np.sqrt(n)
    ax3.axhline(y=conf_bound, color='red', linestyle='--', linewidth=1)
    ax3.axhline(y=-conf_bound, color='red', linestyle='--', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax3.set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('ACF')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Residual heatmap (sample)
    ax4 = plt.subplot(2, 2, 4)
    
    # Take a small sample of the residual matrix for visualization
    sample_size = min(50, residuals.shape[0])
    residual_sample = residuals[:sample_size, :sample_size]
    
    im = ax4.imshow(residual_sample, cmap='RdBu_r', aspect='auto')
    ax4.set_title('Residual Matrix Sample', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Column Index')
    ax4.set_ylabel('Row Index')
    plt.colorbar(im, ax=ax4, label='Residual Value')
    
    # Overall title
    fig.suptitle('Gavish-Donoho Assumption Diagnostics', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = os.path.join(save_dir, 'residual_diagnostics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✅ Saved diagnostic plots to: {save_path}")
    
    plt.close()


def plot_rank_comparison(results, save_dir='output'):
    """
    Create a bar chart comparing different rank selection methods.
    
    Parameters:
    -----------
    results : dict
        Output from compare_rank_selection_methods()
    save_dir : str, default='output'
        Directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    methods = []
    n_components = []
    explained_var = []
    
    for threshold, data in sorted(results['energy_methods'].items()):
        methods.append(f"{int(threshold*100)}% Energy")
        n_components.append(data['n_components'])
        explained_var.append(data['explained_variance'])
    
    methods.append("Gavish-Donoho")
    n_components.append(results['gavish_donoho']['n_components'])
    explained_var.append(results['gavish_donoho']['explained_variance'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart 1: Number of components
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange']
    ax1.bar(methods, n_components, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Components (k)', fontsize=12)
    ax1.set_title('Recommended Rank by Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (m, k) in enumerate(zip(methods, n_components)):
        ax1.text(i, k + max(n_components)*0.02, str(k), 
                ha='center', va='bottom', fontweight='bold')
    
    # Bar chart 2: Explained variance
    ax2.bar(methods, [v*100 for v in explained_var], color=colors, 
            edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Explained Variance (%)', fontsize=12)
    ax2.set_title('Variance Explained by Selected Components', fontsize=14, fontweight='bold')
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (m, var) in enumerate(zip(methods, explained_var)):
        ax2.text(i, var*100 + 1, f"{var*100:.1f}%", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'rank_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✅ Saved rank comparison to: {save_path}")
    
    plt.close()
