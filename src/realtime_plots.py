"""
Real-time Streaming Results Visualization

Creates comprehensive plots for real-time streaming benchmark results.
"""
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RealTimePlotter:
    """Creates visualizations for real-time streaming benchmark results."""
    
    def __init__(self, results_dir: str = "results/realtime"):
        """
        Initialize the plotter.
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load results
        self.summary_df = self._load_summary_data()
        self.detailed_df = self._load_detailed_data()
        
    def _load_summary_data(self) -> pd.DataFrame:
        """Load summary results from CSV."""
        csv_file = self.results_dir / "realtime_matrix_summary.csv"
        if csv_file.exists():
            return pd.read_csv(csv_file)
        else:
            logger.error(f"Summary CSV not found: {csv_file}")
            return pd.DataFrame()
    
    def _load_detailed_data(self) -> pd.DataFrame:
        """Load detailed worker results from CSV."""
        csv_file = self.results_dir / "realtime_worker_details.csv"
        if csv_file.exists():
            return pd.read_csv(csv_file)
        else:
            logger.error(f"Detailed CSV not found: {csv_file}")
            return pd.DataFrame()
    
    def create_all_plots(self):
        """Create all visualization plots."""
        logger.info("Creating real-time streaming visualization plots...")
        
        # 1. Efficiency vs Workers
        self.plot_efficiency_vs_workers()
        
        # 2. Throughput vs Workers
        self.plot_throughput_vs_workers()
        
        # 3. Rate Accuracy vs Workers
        self.plot_rate_accuracy_vs_workers()
        
        # 4. CPU Usage vs Workers
        self.plot_cpu_usage_vs_workers()
        
        # 5. Memory Usage vs Workers
        self.plot_memory_usage_vs_workers()
        
        # 6. Strategy Comparison
        self.plot_strategy_comparison()
        
        # 7. Scalability Analysis
        self.plot_scalability_analysis()
        
        # 8. Worker Performance Details
        self.plot_worker_performance()
        
        # 9. Heatmap of Performance Metrics
        self.plot_performance_heatmap()
        
        logger.info(f"All plots saved to {self.plots_dir}")
    
    def plot_efficiency_vs_workers(self):
        """Plot efficiency vs number of workers."""
        plt.figure(figsize=(10, 6))
        
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            plt.plot(strategy_data['workers'], strategy_data['efficiency'], 
                    marker='o', linewidth=2, markersize=8, label=strategy.title())
        
        plt.xlabel('Number of Workers', fontsize=12)
        plt.ylabel('Efficiency (%)', fontsize=12)
        plt.title('Real-time Streaming Efficiency vs Workers', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks([1, 2, 4])
        
        # Add value annotations
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            for _, row in strategy_data.iterrows():
                plt.annotate(f'{row["efficiency"]:.1f}%', 
                           (row['workers'], row['efficiency']),
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'efficiency_vs_workers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_throughput_vs_workers(self):
        """Plot throughput vs number of workers."""
        plt.figure(figsize=(10, 6))
        
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            plt.plot(strategy_data['workers'], strategy_data['throughput'], 
                    marker='s', linewidth=2, markersize=8, label=strategy.title())
        
        plt.xlabel('Number of Workers', fontsize=12)
        plt.ylabel('Throughput (inf/s)', fontsize=12)
        plt.title('Real-time Streaming Throughput vs Workers', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks([1, 2, 4])
        
        # Add value annotations
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            for _, row in strategy_data.iterrows():
                plt.annotate(f'{row["throughput"]:.0f}', 
                           (row['workers'], row['throughput']),
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'throughput_vs_workers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rate_accuracy_vs_workers(self):
        """Plot rate accuracy vs number of workers."""
        plt.figure(figsize=(10, 6))
        
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            plt.plot(strategy_data['workers'], strategy_data['rate_accuracy'], 
                    marker='^', linewidth=2, markersize=8, label=strategy.title())
        
        plt.xlabel('Number of Workers', fontsize=12)
        plt.ylabel('Rate Accuracy (%)', fontsize=12)
        plt.title('Real-time Streaming Rate Accuracy vs Workers', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks([1, 2, 4])
        # Set y-axis range to show the actual variation in rate accuracy
        min_accuracy = self.summary_df['rate_accuracy'].min()
        max_accuracy = self.summary_df['rate_accuracy'].max()
        y_range = max_accuracy - min_accuracy
        plt.ylim(min_accuracy - 0.1, max_accuracy + 0.1)  # Add small margin
        
        # Add value annotations
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            for _, row in strategy_data.iterrows():
                plt.annotate(f'{row["rate_accuracy"]:.3f}%', 
                           (row['workers'], row['rate_accuracy']),
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'rate_accuracy_vs_workers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cpu_usage_vs_workers(self):
        """Plot CPU usage vs number of workers."""
        plt.figure(figsize=(10, 6))
        
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            plt.plot(strategy_data['workers'], strategy_data['cpu_percent'], 
                    marker='d', linewidth=2, markersize=8, label=strategy.title())
        
        plt.xlabel('Number of Workers', fontsize=12)
        plt.ylabel('CPU Usage (%)', fontsize=12)
        plt.title('Real-time Streaming CPU Usage vs Workers', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks([1, 2, 4])
        
        # Add value annotations
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            for _, row in strategy_data.iterrows():
                plt.annotate(f'{row["cpu_percent"]:.2f}%', 
                           (row['workers'], row['cpu_percent']),
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'cpu_usage_vs_workers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_memory_usage_vs_workers(self):
        """Plot memory usage vs number of workers."""
        plt.figure(figsize=(10, 6))
        
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            plt.plot(strategy_data['workers'], strategy_data['memory_mb'], 
                    marker='v', linewidth=2, markersize=8, label=strategy.title())
        
        plt.xlabel('Number of Workers', fontsize=12)
        plt.ylabel('Memory Usage (MB)', fontsize=12)
        plt.title('Real-time Streaming Memory Usage vs Workers', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks([1, 2, 4])
        
        # Add value annotations
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            for _, row in strategy_data.iterrows():
                plt.annotate(f'{row["memory_mb"]:.0f}MB', 
                           (row['workers'], row['memory_mb']),
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'memory_usage_vs_workers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_strategy_comparison(self):
        """Plot strategy comparison across all metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Efficiency comparison
        efficiency_data = self.summary_df.pivot(index='workers', columns='strategy', values='efficiency')
        efficiency_data.plot(kind='bar', ax=axes[0,0], width=0.8)
        axes[0,0].set_title('Efficiency Comparison', fontweight='bold')
        axes[0,0].set_xlabel('Workers')
        axes[0,0].set_ylabel('Efficiency (%)')
        axes[0,0].legend(title='Strategy')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # Throughput comparison
        throughput_data = self.summary_df.pivot(index='workers', columns='strategy', values='throughput')
        throughput_data.plot(kind='bar', ax=axes[0,1], width=0.8)
        axes[0,1].set_title('Throughput Comparison', fontweight='bold')
        axes[0,1].set_xlabel('Workers')
        axes[0,1].set_ylabel('Throughput (inf/s)')
        axes[0,1].legend(title='Strategy')
        axes[0,1].tick_params(axis='x', rotation=0)
        
        # Rate accuracy comparison
        rate_data = self.summary_df.pivot(index='workers', columns='strategy', values='rate_accuracy')
        rate_data.plot(kind='bar', ax=axes[1,0], width=0.8)
        axes[1,0].set_title('Rate Accuracy Comparison', fontweight='bold')
        axes[1,0].set_xlabel('Workers')
        axes[1,0].set_ylabel('Rate Accuracy (%)')
        axes[1,0].legend(title='Strategy')
        axes[1,0].tick_params(axis='x', rotation=0)
        
        # CPU usage comparison
        cpu_data = self.summary_df.pivot(index='workers', columns='strategy', values='cpu_percent')
        cpu_data.plot(kind='bar', ax=axes[1,1], width=0.8)
        axes[1,1].set_title('CPU Usage Comparison', fontweight='bold')
        axes[1,1].set_xlabel('Workers')
        axes[1,1].set_ylabel('CPU Usage (%)')
        axes[1,1].legend(title='Strategy')
        axes[1,1].tick_params(axis='x', rotation=0)
        
        plt.suptitle('Real-time Streaming Strategy Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scalability_analysis(self):
        """Plot scalability analysis showing improvement from 1 to 4 workers."""
        plt.figure(figsize=(12, 8))
        
        # Calculate scalability metrics
        scalability_data = []
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy].sort_values('workers')
            
            # Calculate improvements
            efficiency_1w = strategy_data[strategy_data['workers'] == 1]['efficiency'].iloc[0]
            efficiency_4w = strategy_data[strategy_data['workers'] == 4]['efficiency'].iloc[0]
            efficiency_improvement = ((efficiency_4w - efficiency_1w) / efficiency_1w) * 100
            
            throughput_1w = strategy_data[strategy_data['workers'] == 1]['throughput'].iloc[0]
            throughput_4w = strategy_data[strategy_data['workers'] == 4]['throughput'].iloc[0]
            throughput_improvement = ((throughput_4w - throughput_1w) / throughput_1w) * 100
            
            scalability_data.append({
                'strategy': strategy,
                'efficiency_improvement': efficiency_improvement,
                'throughput_improvement': throughput_improvement
            })
        
        scalability_df = pd.DataFrame(scalability_data)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Efficiency improvement
        bars1 = ax1.bar(scalability_df['strategy'], scalability_df['efficiency_improvement'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Efficiency Improvement (1→4 Workers)', fontweight='bold')
        ax1.set_ylabel('Improvement (%)')
        ax1.set_xlabel('Strategy')
        
        # Add value labels on bars
        for bar, value in zip(bars1, scalability_df['efficiency_improvement']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Throughput improvement
        bars2 = ax2.bar(scalability_df['strategy'], scalability_df['throughput_improvement'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Throughput Improvement (1→4 Workers)', fontweight='bold')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_xlabel('Strategy')
        
        # Add value labels on bars
        for bar, value in zip(bars2, scalability_df['throughput_improvement']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.suptitle('Real-time Streaming Scalability Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_worker_performance(self):
        """Plot detailed worker performance if available."""
        if self.detailed_df.empty:
            logger.warning("No detailed worker data available for plotting")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create subplots for different worker counts
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, workers in enumerate([1, 2, 4]):
            worker_data = self.detailed_df[self.detailed_df['workers'] == workers]
            
            if not worker_data.empty:
                # Plot processed samples by worker
                for strategy in worker_data['strategy'].unique():
                    strategy_worker_data = worker_data[worker_data['strategy'] == strategy]
                    axes[i].plot(strategy_worker_data['worker_id'], 
                               strategy_worker_data['processed_samples'],
                               marker='o', linewidth=2, markersize=6, 
                               label=strategy.title())
                
                axes[i].set_title(f'{workers} Workers', fontweight='bold')
                axes[i].set_xlabel('Worker ID')
                axes[i].set_ylabel('Processed Samples')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Worker Performance Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'worker_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_heatmap(self):
        """Create a heatmap of performance metrics."""
        plt.figure(figsize=(12, 8))
        
        # Create pivot tables for different metrics
        efficiency_pivot = self.summary_df.pivot(index='strategy', columns='workers', values='efficiency')
        throughput_pivot = self.summary_df.pivot(index='strategy', columns='workers', values='throughput')
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Efficiency heatmap
        sns.heatmap(efficiency_pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=ax1, cbar_kws={'label': 'Efficiency (%)'})
        ax1.set_title('Efficiency Heatmap', fontweight='bold')
        ax1.set_xlabel('Workers')
        ax1.set_ylabel('Strategy')
        
        # Throughput heatmap
        sns.heatmap(throughput_pivot, annot=True, fmt='.0f', cmap='Blues', 
                   ax=ax2, cbar_kws={'label': 'Throughput (inf/s)'})
        ax2.set_title('Throughput Heatmap', fontweight='bold')
        ax2.set_xlabel('Workers')
        ax2.set_ylabel('Strategy')
        
        plt.suptitle('Real-time Streaming Performance Heatmaps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    

def main():
    """Main function to create all plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create real-time streaming visualization plots')
    parser.add_argument('--results-dir', default='results/realtime',
                       help='Directory containing benchmark results')
    
    args = parser.parse_args()
    
    # Create plotter and generate all plots
    plotter = RealTimePlotter(args.results_dir)
    plotter.create_all_plots()
    
    print(f"✅ All visualization plots created successfully!")
    print(f"📁 Plots saved to: {plotter.plots_dir}")
    print(f"📊 Generated plots:")
    print(f"   - efficiency_vs_workers.png")
    print(f"   - throughput_vs_workers.png")
    print(f"   - rate_accuracy_vs_workers.png")
    print(f"   - cpu_usage_vs_workers.png")
    print(f"   - memory_usage_vs_workers.png")
    print(f"   - strategy_comparison.png")
    print(f"   - scalability_analysis.png")
    print(f"   - worker_performance.png")
    print(f"   - performance_heatmap.png")

if __name__ == "__main__":
    main()
