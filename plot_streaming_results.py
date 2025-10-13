#!/usr/bin/env python3
"""
Comprehensive Plotting Script for Streaming Benchmark Results

This script creates multiple visualizations for the streaming benchmark results:
1. Throughput comparison across strategies and nodes
2. Resource utilization (CPU and Memory)
3. Scaling efficiency analysis
4. Performance trends
5. Comparative analysis charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load streaming benchmark results from CSV and JSON files."""
    results_dir = Path("results/streaming")
    
    # Load CSV data
    csv_file = results_dir / "averaged_results.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Results file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Load detailed JSON results if available
    json_file = results_dir / "detailed_results.json"
    detailed_results = None
    if json_file.exists():
        with open(json_file, 'r') as f:
            detailed_results = json.load(f)
    
    return df, detailed_results

def create_throughput_comparison(df):
    """Create throughput comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Throughput by Strategy and Nodes
    pivot_throughput = df.pivot(index='nodes', columns='strategy', values='avg_throughput')
    
    pivot_throughput.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Throughput Comparison Across Strategies and Nodes', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Throughput (inferences/second)', fontsize=12)
    ax1.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f', rotation=90, padding=3, fontsize=8)
    
    # Plot 2: Line plot showing scaling trends
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].sort_values('nodes')
        ax2.plot(strategy_data['nodes'], strategy_data['avg_throughput'], 
                marker='o', linewidth=2, markersize=8, label=strategy.title())
    
    ax2.set_title('Throughput Scaling Trends', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Nodes', fontsize=12)
    ax2.set_ylabel('Throughput (inferences/second)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Use actual node counts from data for x-axis
    actual_nodes = sorted(df['nodes'].unique())
    ax2.set_xticks(actual_nodes)
    ax2.set_xticklabels(actual_nodes)
    
    plt.tight_layout()
    plt.savefig('results/plots/streaming_throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_resource_utilization(df):
    """Create resource utilization plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: CPU Usage
    pivot_cpu = df.pivot(index='nodes', columns='strategy', values='avg_cpu_percent')
    
    pivot_cpu.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('CPU Utilization Across Strategies and Nodes', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('CPU Usage (%)', fontsize=12)
    ax1.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', rotation=90, padding=3, fontsize=8)
    
    # Plot 2: Memory Usage
    pivot_memory = df.pivot(index='nodes', columns='strategy', values='avg_memory_mb')
    
    pivot_memory.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Memory Utilization Across Strategies and Nodes', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Nodes', fontsize=12)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.0f MB', rotation=90, padding=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/plots/streaming_resource_utilization.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_scaling_efficiency(df):
    """Create scaling efficiency analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calculate scaling efficiency (throughput at N nodes / throughput at 1 node)
    efficiency_data = []
    
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy].sort_values('nodes')
        baseline_throughput = strategy_df.iloc[0]['avg_throughput']
        
        for _, row in strategy_df.iterrows():
            efficiency = row['avg_throughput'] / baseline_throughput
            efficiency_data.append({
                'strategy': strategy,
                'nodes': row['nodes'],
                'efficiency': efficiency,
                'throughput': row['avg_throughput']
            })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    # Plot 1: Scaling Efficiency
    for strategy in efficiency_df['strategy'].unique():
        strategy_data = efficiency_df[efficiency_df['strategy'] == strategy].sort_values('nodes')
        ax1.plot(strategy_data['nodes'], strategy_data['efficiency'], 
                marker='s', linewidth=2, markersize=8, label=strategy.title())
    
    # Add perfect scaling line
    ax1.axline((1, 1), slope=0, color='red', linestyle='--', alpha=0.7, label='Perfect Scaling')
    
    ax1.set_title('Scaling Efficiency Analysis', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Scaling Efficiency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Use actual node counts from data for x-axis
    actual_nodes = sorted(efficiency_df['nodes'].unique())
    ax1.set_xticks(actual_nodes)
    ax1.set_xticklabels(actual_nodes)
    
    # Plot 2: Efficiency vs Throughput scatter
    for strategy in efficiency_df['strategy'].unique():
        strategy_data = efficiency_df[efficiency_df['strategy'] == strategy]
        ax2.scatter(strategy_data['efficiency'], strategy_data['throughput'], 
                   s=100, alpha=0.7, label=strategy.title())
    
    ax2.set_title('Efficiency vs Throughput Trade-off', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Scaling Efficiency', fontsize=12)
    ax2.set_ylabel('Throughput (inferences/second)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/streaming_scaling_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_heatmap(df):
    """Create performance heatmaps."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Throughput heatmap
    pivot_throughput = df.pivot(index='nodes', columns='strategy', values='avg_throughput')
    sns.heatmap(pivot_throughput, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Throughput (inf/s)'})
    ax1.set_title('Throughput Heatmap', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Strategy', fontsize=12)
    ax1.set_ylabel('Number of Nodes', fontsize=12)
    
    # CPU usage heatmap
    pivot_cpu = df.pivot(index='nodes', columns='strategy', values='avg_cpu_percent')
    sns.heatmap(pivot_cpu, annot=True, fmt='.1f', cmap='Blues', ax=ax2, cbar_kws={'label': 'CPU Usage (%)'})
    ax2.set_title('CPU Usage Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Strategy', fontsize=12)
    ax2.set_ylabel('Number of Nodes', fontsize=12)
    
    # Memory usage heatmap
    pivot_memory = df.pivot(index='nodes', columns='strategy', values='avg_memory_mb')
    sns.heatmap(pivot_memory, annot=True, fmt='.0f', cmap='Greens', ax=ax3, cbar_kws={'label': 'Memory (MB)'})
    ax3.set_title('Memory Usage Heatmap', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Strategy', fontsize=12)
    ax3.set_ylabel('Number of Nodes', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/plots/streaming_performance_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_summary(df):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Throughput comparison (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    pivot_throughput = df.pivot(index='nodes', columns='strategy', values='avg_throughput')
    pivot_throughput.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Throughput (inferences/second)', fontsize=12)
    ax1.legend(title='Strategy')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=0)
    
    # 2. Best performer summary (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    best_performers = []
    for nodes in df['nodes'].unique():
        node_data = df[df['nodes'] == nodes]
        best_idx = node_data['avg_throughput'].idxmax()
        best_strategy = node_data.loc[best_idx, 'strategy']
        best_throughput = node_data.loc[best_idx, 'avg_throughput']
        best_performers.append(f"{nodes} nodes: {best_strategy.title()}\n({best_throughput:.1f} inf/s)")
    
    ax2.text(0.1, 0.9, 'Best Performers by Node Count:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    for i, performer in enumerate(best_performers):
        ax2.text(0.1, 0.8 - i*0.15, performer, fontsize=10, transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # 3. Resource utilization (middle row)
    ax3 = fig.add_subplot(gs[1, 0])
    pivot_cpu = df.pivot(index='nodes', columns='strategy', values='avg_cpu_percent')
    pivot_cpu.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('CPU Utilization', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Nodes', fontsize=10)
    ax3.set_ylabel('CPU Usage (%)', fontsize=10)
    ax3.legend(title='Strategy', fontsize=8)
    ax3.tick_params(axis='x', rotation=0, labelsize=8)
    
    ax4 = fig.add_subplot(gs[1, 1])
    pivot_memory = df.pivot(index='nodes', columns='strategy', values='avg_memory_mb')
    pivot_memory.plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_title('Memory Utilization', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Number of Nodes', fontsize=10)
    ax4.set_ylabel('Memory (MB)', fontsize=10)
    ax4.legend(title='Strategy', fontsize=8)
    ax4.tick_params(axis='x', rotation=0, labelsize=8)
    
    # 4. Scaling efficiency (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    efficiency_data = []
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy].sort_values('nodes')
        baseline_throughput = strategy_df.iloc[0]['avg_throughput']
        for _, row in strategy_df.iterrows():
            efficiency = row['avg_throughput'] / baseline_throughput
            efficiency_data.append({
                'strategy': strategy,
                'nodes': row['nodes'],
                'efficiency': efficiency
            })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    for strategy in efficiency_df['strategy'].unique():
        strategy_data = efficiency_df[efficiency_df['strategy'] == strategy].sort_values('nodes')
        ax5.plot(strategy_data['nodes'], strategy_data['efficiency'], 
                marker='o', linewidth=2, markersize=6, label=strategy.title())
    
    ax5.axline((1, 1), slope=0, color='red', linestyle='--', alpha=0.7, label='Perfect Scaling')
    ax5.set_title('Scaling Efficiency', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Number of Nodes', fontsize=10)
    ax5.set_ylabel('Efficiency', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Use actual node counts from data for x-axis
    actual_nodes = sorted(efficiency_df['nodes'].unique())
    ax5.set_xticks(actual_nodes)
    ax5.set_xticklabels(actual_nodes)
    
    # 5. Performance statistics (bottom row)
    ax6 = fig.add_subplot(gs[2, :])
    
    # Create a table with key statistics
    stats_data = []
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        max_throughput = strategy_df['avg_throughput'].max()
        min_throughput = strategy_df['avg_throughput'].min()
        avg_cpu = strategy_df['avg_cpu_percent'].mean()
        avg_memory = strategy_df['avg_memory_mb'].mean()
        
        # Calculate scaling efficiency
        baseline = strategy_df[strategy_df['nodes'] == 1]['avg_throughput'].iloc[0]
        max_efficiency = max_throughput / baseline
        
        stats_data.append([
            strategy.title(),
            f"{max_throughput:.1f}",
            f"{min_throughput:.1f}",
            f"{max_efficiency:.2f}",
            f"{avg_cpu:.1f}%",
            f"{avg_memory:.0f} MB"
        ])
    
    table = ax6.table(cellText=stats_data,
                     colLabels=['Strategy', 'Max Throughput', 'Min Throughput', 'Max Efficiency', 'Avg CPU', 'Avg Memory'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_data) + 1):
        for j in range(6):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax6.set_title('Performance Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    ax6.axis('off')
    
    plt.suptitle('Streaming Benchmark Results - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('results/plots/streaming_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all plots."""
    print("🎨 Creating comprehensive plots for streaming benchmark results...")
    
    # Create plots directory if it doesn't exist
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        df, detailed_results = load_results()
        print(f"✅ Loaded results for {len(df)} configurations")
        print(f"   Strategies: {df['strategy'].unique()}")
        print(f"   Node counts: {sorted(df['nodes'].unique())}")
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    # Generate all plots
    print("\n📊 Generating throughput comparison plots...")
    create_throughput_comparison(df)
    
    print("📊 Generating resource utilization plots...")
    create_resource_utilization(df)
    
    print("📊 Generating scaling efficiency analysis...")
    create_scaling_efficiency(df)
    
    print("📊 Generating performance heatmaps...")
    create_performance_heatmap(df)
    
    print("📊 Generating comprehensive dashboard...")
    create_comprehensive_summary(df)
    
    print("\n🎉 All plots generated successfully!")
    print("📁 Plots saved to: results/plots/")
    print("   - streaming_throughput_comparison.png")
    print("   - streaming_resource_utilization.png")
    print("   - streaming_scaling_efficiency.png")
    print("   - streaming_performance_heatmaps.png")
    print("   - streaming_comprehensive_dashboard.png")

if __name__ == "__main__":
    main()
