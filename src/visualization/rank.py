"""
Visualization Tools for Rank Selection Methods.

This module provides comprehensive visualizations to compare all 6 rank
selection methods on the same scree plot, making it easy to understand
the differences and consensus among methods.
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_scree_with_elbows(singular_values, results, save_path=None):
    """
    Plot scree plot with all elbow points marked.
    
    Parameters:
    -----------
    singular_values : array
        Singular values from SVD
    results : dict
        Output from compare_all_rank_methods()
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    n = len(singular_values)
    x = np.arange(1, n + 1)
    
    # Plot scree curve
    ax.plot(x, singular_values, 'k-', linewidth=2, label='Singular Values', alpha=0.7)
    ax.scatter(x, singular_values, c='gray', s=20, alpha=0.5, zorder=1)
    
    # Color scheme by category
    colors = {
        'energy': 'blue',
        'gavish_donoho': 'green',
        'kneedle': 'orange',
        'l_method': 'purple'
    }
    
    markers = {
        'energy': 's',  # square
        'gavish_donoho': '^',  # triangle
        'kneedle': 'D',  # diamond
        'l_method': 'o'  # circle
    }
    
    sizes = {
        0.90: 80,
        0.95: 120,
        0.99: 80
    }
    
    # Plot elbow points
    plotted_categories = set()
    
    # Energy methods
    for threshold, data in sorted(results['energy_methods'].items()):
        k = data['n_components']
        label = data['method_name'] if 'energy' not in plotted_categories else None
        if label:
            plotted_categories.add('energy')
        
        alpha = 0.9 if threshold == 0.95 else 0.6
        size = sizes.get(threshold, 100)
        
        ax.scatter(k, singular_values[k-1], 
                  color=colors['energy'], 
                  marker=markers['energy'],
                  s=size, 
                  alpha=alpha,
                  edgecolors='black',
                  linewidths=1.5,
                  label=label if threshold == 0.95 else None,
                  zorder=3)
        
        ax.annotate(f"{int(threshold*100)}%", 
                   xy=(k, singular_values[k-1]),
                   xytext=(10, 10 if threshold == 0.95 else -10),
                   textcoords='offset points',
                   fontsize=9,
                   color=colors['energy'],
                   fontweight='bold')
    
    # Gavish-Donoho
    if results['gavish_donoho'].get('n_components'):
        k = results['gavish_donoho']['n_components']
        ax.scatter(k, singular_values[k-1],
                  color=colors['gavish_donoho'],
                  marker=markers['gavish_donoho'],
                  s=150,
                  alpha=0.9,
                  edgecolors='black',
                  linewidths=2,
                  label='Gavish-Donoho',
                  zorder=4)
        ax.annotate('GD',
                   xy=(k, singular_values[k-1]),
                   xytext=(10, -15),
                   textcoords='offset points',
                   fontsize=10,
                   color=colors['gavish_donoho'],
                   fontweight='bold')
    
    # Kneedle
    if results['kneedle'].get('n_components'):
        k = results['kneedle']['n_components']
        ax.scatter(k, singular_values[k-1],
                  color=colors['kneedle'],
                  marker=markers['kneedle'],
                  s=150,
                  alpha=0.9,
                  edgecolors='black',
                  linewidths=2,
                  label='Kneedle',
                  zorder=5)
        ax.annotate('Knee',
                   xy=(k, singular_values[k-1]),
                   xytext=(-25, 10),
                   textcoords='offset points',
                   fontsize=10,
                   color=colors['kneedle'],
                   fontweight='bold')
    
    # L-Method
    if results['l_method'].get('n_components'):
        k = results['l_method']['n_components']
        ax.scatter(k, singular_values[k-1],
                  color=colors['l_method'],
                  marker=markers['l_method'],
                  s=150,
                  alpha=0.9,
                  edgecolors='black',
                  linewidths=2,
                  label='L-Method',
                  zorder=4)
        ax.annotate('L',
                   xy=(k, singular_values[k-1]),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=10,
                   color=colors['l_method'],
                   fontweight='bold')
    
    ax.set_xlabel('Component Index (k)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Singular Value (σₖ)', fontsize=13, fontweight='bold')
    ax.set_title('Scree Plot with Elbow Points from 6 Methods', 
                fontsize=15, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, min(100, n))
    
    # Add log scale for better visualization
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved scree plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_method_comparison_bars(results, save_path=None):
    """
    Create bar chart comparing all methods.
    
    Parameters:
    -----------
    results : dict
        Output from compare_all_rank_methods()
    save_path : str, optional
        Path to save figure
    """
    # Collect data
    methods = []
    k_values = []
    variances = []
    categories = []
    
    # Energy methods
    for threshold in sorted(results['energy_methods'].keys()):
        data = results['energy_methods'][threshold]
        methods.append(data['method_name'])
        k_values.append(data['n_components'])
        variances.append(data['explained_variance'] * 100)
        categories.append('Empirical')
    
    # Gavish-Donoho
    if results['gavish_donoho'].get('n_components'):
        methods.append('Gavish-Donoho')
        k_values.append(results['gavish_donoho']['n_components'])
        variances.append(results['gavish_donoho']['explained_variance'] * 100)
        categories.append('Statistical')
    
    # Kneedle
    if results['kneedle'].get('n_components'):
        methods.append('Kneedle')
        k_values.append(results['kneedle']['n_components'])
        variances.append(results['kneedle']['explained_variance'] * 100)
        categories.append('Geometric')
    
    # L-Method
    if results['l_method'].get('n_components'):
        methods.append('L-Method')
        k_values.append(results['l_method']['n_components'])
        variances.append(results['l_method']['explained_variance'] * 100)
        categories.append('Statistical')
    
    # Colors
    color_map = {
        'Empirical': 'skyblue',
        'Statistical': 'lightgreen',
        'Geometric': 'orange'
    }
    colors = [color_map[cat] for cat in categories]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Number of components
    bars1 = ax1.bar(range(len(methods)), k_values, color=colors, 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Number of Components (k)', fontsize=12, fontweight='bold')
    ax1.set_title('Recommended Rank by Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, k) in enumerate(zip(bars1, k_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(k_values)*0.02,
                f'{k}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Explained variance
    bars2 = ax2.bar(range(len(methods)), variances, color=colors,
                     edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Variance Explained by Selected Components', 
                  fontsize=14, fontweight='bold')
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, 
                label='95% threshold', alpha=0.7)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(80, 100)
    
    # Add value labels
    for i, (bar, var) in enumerate(zip(bars2, variances)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{var:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[cat], edgecolor='black', 
                            label=cat, alpha=0.8)
                      for cat in ['Empirical', 'Statistical', 'Geometric']]
    ax1.legend(handles=legend_elements, loc='upper left', title='Category')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved comparison bars to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_consensus_analysis(results, save_path=None):
    """
    Plot consensus analysis showing agreement between methods.
    
    Parameters:
    -----------
    results : dict
        Output from compare_all_rank_methods()
    save_path : str, optional
        Path to save figure
    """
    # Collect k values
    k_values = []
    method_names = []
    
    for threshold in sorted(results['energy_methods'].keys()):
        k_values.append(results['energy_methods'][threshold]['n_components'])
        method_names.append(f"{int(threshold*100)}% Energy")
    
    if results['gavish_donoho'].get('n_components'):
        k_values.append(results['gavish_donoho']['n_components'])
        method_names.append('GD')
    
    if results['kneedle'].get('n_components'):
        k_values.append(results['kneedle']['n_components'])
        method_names.append('Kneedle')
    
    if results['l_method'].get('n_components'):
        k_values.append(results['l_method']['n_components'])
        method_names.append('L-Method')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot style visualization
    mean_k = np.mean(k_values)
    median_k = np.median(k_values)
    
    ax.scatter(k_values, method_names, s=200, c='blue', alpha=0.6, 
              edgecolors='black', linewidths=2, zorder=3)
    
    # Add mean and median lines
    ax.axvline(mean_k, color='red', linestyle='--', linewidth=2, 
              label=f'Mean = {mean_k:.1f}', alpha=0.7)
    ax.axvline(median_k, color='green', linestyle='--', linewidth=2,
              label=f'Median = {median_k:.0f}', alpha=0.7)
    
    # Shade consensus region (±1 std)
    std_k = np.std(k_values)
    ax.axvspan(mean_k - std_k, mean_k + std_k, alpha=0.2, color='yellow',
              label=f'±1 SD [{mean_k-std_k:.1f}, {mean_k+std_k:.1f}]')
    
    ax.set_xlabel('Recommended k (Number of Components)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')
    ax.set_title('Consensus Analysis: Agreement Between Methods',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved consensus plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_rank_visualizations(singular_values, results, save_dir='output'):
    """
    Generate all rank selection visualizations in one call.
    
    Parameters:
    -----------
    singular_values : array
        Singular values from SVD
    results : dict
        Output from compare_all_rank_methods()
    save_dir : str, default='output'
        Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n  Generating rank selection visualizations...")
    
    # 1. Scree plot with elbows
    plot_scree_with_elbows(
        singular_values, 
        results,
        save_path=os.path.join(save_dir, 'scree_plot_with_elbows.png')
    )
    
    # 2. Method comparison bars
    plot_method_comparison_bars(
        results,
        save_path=os.path.join(save_dir, 'rank_method_comparison.png')
    )
    
    # 3. Consensus analysis
    plot_consensus_analysis(
        results,
        save_path=os.path.join(save_dir, 'rank_consensus.png')
    )
    
    print("  ✅ All rank selection plots generated!\n")
