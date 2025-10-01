"""
Comprehensive benchmarking script for comparing data partitioning approaches.

This script runs all three partitioning approaches (uniform, dynamic, sharded)
and provides detailed analysis and comparison of results.
"""
import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from approaches.uniform_partitioning import run_uniform_partitioning
from approaches.dynamic_partitioning import run_dynamic_partitioning
from approaches.sharded_partitioning import run_sharded_partitioning
from utils import get_system_info, print_results_summary, save_results

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Runs comprehensive benchmarks for all partitioning approaches."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.system_info = get_system_info()
        
    def run_single_benchmark(self, approach: str, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Run a single benchmark for the specified approach.
        
        Args:
            approach: Name of the partitioning approach
            config: Configuration for the benchmark
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Running {approach} partitioning benchmark...")
        
        start_time = time.time()
        
        try:
            if approach == "uniform":
                results = run_uniform_partitioning(config)
            elif approach == "dynamic":
                results = run_dynamic_partitioning(config)
            elif approach == "sharded":
                results = run_sharded_partitioning(config)
            else:
                raise ValueError(f"Unknown approach: {approach}")
            
            end_time = time.time()
            results['benchmark_duration'] = end_time - start_time
            results['approach'] = approach
            results['config'] = config
            
            logger.info(f"{approach} benchmark completed in {results['benchmark_duration']:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"{approach} benchmark failed: {str(e)}")
            return {
                'approach': approach,
                'error': str(e),
                'benchmark_duration': time.time() - start_time
            }
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all partitioning approaches and collect results.
        
        Returns:
            Dictionary containing all benchmark results
        """
        logger.info("Starting comprehensive benchmark suite...")
        
        all_results = {
            'system_info': self.system_info,
            'config': self.config,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'approaches': {}
        }
        
        # Run each approach multiple times for statistical significance
        num_runs = self.config.get('num_runs', 3)
        
        for approach in ['uniform', 'dynamic', 'sharded']:
            logger.info(f"Running {approach} partitioning {num_runs} times...")
            
            approach_results = []
            for run in range(num_runs):
                logger.info(f"Run {run + 1}/{num_runs} for {approach}")
                
                # Adjust config for each approach
                approach_config = self.config.copy()
                if approach == 'uniform':
                    approach_config.update({
                        'world_size': self.config.get('num_workers', 4),
                        'batch_size': self.config.get('batch_size', 128),
                        'num_workers': self.config.get('data_loader_workers', 1),
                        'master_addr': self.config.get('master_addr', 'localhost'),
                        'master_port': self.config.get('master_port', '12355')
                    })
                elif approach == 'dynamic':
                    approach_config.update({
                        'num_workers': self.config.get('num_workers', 4),
                        'num_samples': self.config.get('num_samples', 10000),
                        'chunk_size': self.config.get('chunk_size', 128),
                        'ray_address': self.config.get('ray_address')
                    })
                elif approach == 'sharded':
                    approach_config.update({
                        'num_workers': self.config.get('num_workers', 4),
                        'num_samples': self.config.get('num_samples', 10000),
                        'num_shards': self.config.get('num_shards', 4),
                        'dask_scheduler_address': self.config.get('dask_scheduler_address')
                    })
                
                result = self.run_single_benchmark(approach, approach_config)
                approach_results.append(result)
                
                # Small delay between runs
                time.sleep(2)
            
            # Calculate statistics across runs
            if approach_results and 'error' not in approach_results[0]:
                all_results['approaches'][approach] = self.calculate_statistics(approach_results)
            else:
                all_results['approaches'][approach] = {
                    'error': approach_results[0].get('error', 'Unknown error'),
                    'runs': approach_results
                }
        
        return all_results
    
    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics across multiple runs.
        
        Args:
            results: List of results from multiple runs
            
        Returns:
            Dictionary containing statistical summary
        """
        # Extract metrics
        metrics = ['throughput', 'latency_ms', 'avg_cpu_percent', 'avg_memory_mb', 'total_duration']
        
        stats = {
            'runs': results,
            'statistics': {}
        }
        
        for metric in metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                stats['statistics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return stats
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            results: Benchmark results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*80)
        report.append("DATA PARTITIONING BENCHMARK REPORT")
        report.append("="*80)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"System: {results['system_info']['platform']} with {results['system_info']['cpu_count']} CPUs")
        report.append(f"Memory: {results['system_info']['memory_gb']:.1f} GB")
        report.append("")
        
        # Summary table
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 80)
        report.append(f"{'Approach':<15} {'Throughput':<15} {'Latency':<15} {'CPU %':<10} {'Memory MB':<12}")
        report.append("-" * 80)
        
        for approach, data in results['approaches'].items():
            if 'statistics' in data:
                stats = data['statistics']
                throughput = stats.get('throughput', {}).get('mean', 0)
                latency = stats.get('latency_ms', {}).get('mean', 0)
                cpu = stats.get('avg_cpu_percent', {}).get('mean', 0)
                memory = stats.get('avg_memory_mb', {}).get('mean', 0)
                
                report.append(f"{approach:<15} {throughput:<15.2f} {latency:<15.2f} {cpu:<10.1f} {memory:<12.1f}")
            else:
                report.append(f"{approach:<15} {'ERROR':<15} {'ERROR':<15} {'ERROR':<10} {'ERROR':<12}")
        
        report.append("")
        
        # Detailed results
        for approach, data in results['approaches'].items():
            report.append(f"{approach.upper()} PARTITIONING DETAILS")
            report.append("-" * 40)
            
            if 'statistics' in data:
                stats = data['statistics']
                for metric, values in stats.items():
                    report.append(f"{metric}:")
                    report.append(f"  Mean: {values['mean']:.2f}")
                    report.append(f"  Std:  {values['std']:.2f}")
                    report.append(f"  Min:  {values['min']:.2f}")
                    report.append(f"  Max:  {values['max']:.2f}")
                    report.append(f"  Median: {values['median']:.2f}")
                    report.append("")
            else:
                report.append(f"Error: {data.get('error', 'Unknown error')}")
                report.append("")
        
        return "\n".join(report)
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str = "plots"):
        """
        Create visualization plots for benchmark results.
        
        Args:
            results: Benchmark results
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for plotting
        approaches = []
        throughputs = []
        latencies = []
        cpu_usages = []
        memory_usages = []
        
        for approach, data in results['approaches'].items():
            if 'statistics' in data:
                approaches.append(approach)
                stats = data['statistics']
                throughputs.append(stats.get('throughput', {}).get('mean', 0))
                latencies.append(stats.get('latency_ms', {}).get('mean', 0))
                cpu_usages.append(stats.get('avg_cpu_percent', {}).get('mean', 0))
                memory_usages.append(stats.get('avg_memory_mb', {}).get('mean', 0))
        
        if not approaches:
            logger.warning("No valid results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Partitioning Benchmark Results', fontsize=16)
        
        # Throughput comparison
        axes[0, 0].bar(approaches, throughputs, color=['blue', 'green', 'red'])
        axes[0, 0].set_title('Throughput (inferences/sec)')
        axes[0, 0].set_ylabel('Inferences per second')
        
        # Latency comparison
        axes[0, 1].bar(approaches, latencies, color=['blue', 'green', 'red'])
        axes[0, 1].set_title('Latency (ms/batch)')
        axes[0, 1].set_ylabel('Milliseconds per batch')
        
        # CPU usage comparison
        axes[1, 0].bar(approaches, cpu_usages, color=['blue', 'green', 'red'])
        axes[1, 0].set_title('CPU Usage (%)')
        axes[1, 0].set_ylabel('CPU percentage')
        
        # Memory usage comparison
        axes[1, 1].bar(approaches, memory_usages, color=['blue', 'green', 'red'])
        axes[1, 1].set_title('Memory Usage (MB)')
        axes[1, 1].set_ylabel('Memory in MB')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Comprehensive Data Partitioning Benchmark')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples to process')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs per approach')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output file for results')
    parser.add_argument('--report', type=str, default='benchmark_report.txt', help='Output file for report')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    
    args = parser.parse_args()
    
    config = {
        'num_workers': args.num_workers,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'num_runs': args.num_runs,
        'data_loader_workers': 1,
        'master_addr': 'localhost',
        'master_port': '12355',
        'chunk_size': 128,
        'num_shards': 4
    }
    
    try:
        # Run benchmarks
        runner = BenchmarkRunner(config)
        results = runner.run_all_benchmarks()
        
        # Save results
        save_results(results, args.output)
        
        # Generate and save report
        report = runner.generate_report(results)
        with open(args.report, 'w') as f:
            f.write(report)
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK COMPLETED")
        print("="*60)
        print(f"Results saved to: {args.output}")
        print(f"Report saved to: {args.report}")
        
        # Create visualizations if requested
        if args.plots:
            runner.create_visualizations(results)
            print(f"Plots saved to: plots/")
        
        # Print summary to console
        print("\nQUICK SUMMARY:")
        for approach, data in results['approaches'].items():
            if 'statistics' in data:
                stats = data['statistics']
                throughput = stats.get('throughput', {}).get('mean', 0)
                latency = stats.get('latency_ms', {}).get('mean', 0)
                print(f"{approach}: {throughput:.2f} inf/s, {latency:.2f} ms/batch")
            else:
                print(f"{approach}: ERROR - {data.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
