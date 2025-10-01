"""
Real-time Streaming Matrix Benchmark

Comprehensive benchmarking of real-time streaming data partitioning approaches
across different worker configurations and strategies.
"""
import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from approaches.realtime_streaming import run_realtime_streaming_experiment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeMatrixBenchmark:
    """
    Matrix benchmark for real-time streaming data partitioning.
    """
    
    def __init__(self, output_dir: str = "results/realtime"):
        """
        Initialize real-time matrix benchmark.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark configurations
        self.worker_counts = [1, 2, 4]  # Focus on 1, 2, 4 clusters
        self.strategies = ['uniform', 'dynamic', 'sharded']  # 3 approaches
        self.stream_rates = [500]  # Focus on 500 samples/s
        self.max_samples = 10000  # Full dataset for comprehensive results
        
    def run_single_experiment(self, workers: int, strategy: str, 
                            stream_rate: int) -> Dict[str, Any]:
        """
        Run a single real-time streaming experiment.
        
        Args:
            workers: Number of workers
            strategy: Partitioning strategy
            stream_rate: Data generation rate
            
        Returns:
            Dictionary containing experiment results
        """
        config = {
            'num_workers': workers,
            'strategy': strategy,
            'samples_per_second': stream_rate,
            'max_samples': self.max_samples
        }
        
        logger.info(f"Running experiment: {strategy} with {workers} workers at {stream_rate} samples/s")
        
        try:
            results = run_realtime_streaming_experiment(config)
            return results
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            return {
                'strategy': strategy,
                'num_workers': workers,
                'stream_rate': stream_rate,
                'error': str(e),
                'total_processed': 0,
                'efficiency': 0
            }
    
    def run_matrix_benchmark(self) -> Dict[str, Any]:
        """
        Run complete matrix benchmark across all configurations.
        
        Returns:
            Dictionary containing all benchmark results
        """
        logger.info("Starting Real-time Streaming Matrix Benchmark")
        logger.info(f"Worker counts: {self.worker_counts}")
        logger.info(f"Strategies: {self.strategies}")
        logger.info(f"Stream rates: {self.stream_rates}")
        logger.info(f"Max samples per experiment: {self.max_samples}")
        
        all_results = []
        start_time = time.time()
        
        # Run experiments
        for workers in self.worker_counts:
            for strategy in self.strategies:
                for stream_rate in self.stream_rates:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Running: {strategy} | {workers} workers | {stream_rate} samples/s")
                    logger.info(f"{'='*60}")
                    
                    experiment_start = time.time()
                    results = self.run_single_experiment(workers, strategy, stream_rate)
                    experiment_time = time.time() - experiment_start
                    
                    # Add metadata
                    results['experiment_time'] = experiment_time
                    results['stream_rate'] = stream_rate
                    results['timestamp'] = time.time()
                    
                    all_results.append(results)
                    
                    # Save intermediate results
                    self._save_intermediate_results(all_results)
                    
                    logger.info(f"Experiment completed in {experiment_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"\nMatrix benchmark completed in {total_time:.2f}s")
        
        # Compile final results
        final_results = {
            'benchmark_info': {
                'worker_counts': self.worker_counts,
                'strategies': self.strategies,
                'stream_rates': self.stream_rates,
                'max_samples': self.max_samples,
                'total_experiments': len(all_results),
                'total_time': total_time
            },
            'experiments': all_results
        }
        
        # Save final results
        self._save_final_results(final_results)
        
        return final_results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]):
        """Save intermediate results to prevent data loss."""
        intermediate_file = self.output_dir / "intermediate_results.json"
        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final benchmark results."""
        # Save JSON results
        json_file = self.output_dir / "realtime_matrix_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        self._save_csv_summary(results)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _save_csv_summary(self, results: Dict[str, Any]):
        """Save CSV summary of results."""
        experiments = results['experiments']
        
        # Create summary DataFrame
        summary_data = []
        for exp in experiments:
            if 'error' not in exp:
                summary_data.append({
                    'strategy': exp['strategy'],
                    'workers': exp['num_workers'],
                    'stream_rate': exp['stream_rate'],
                    'total_processed': exp['total_processed'],
                    'efficiency': exp['efficiency'],
                    'experiment_time': exp['experiment_time'],
                    'throughput': exp['overall_metrics'].get('throughput', 0),
                    'latency_ms': exp['overall_metrics'].get('latency_ms', 0),
                    'cpu_percent': exp['overall_metrics'].get('avg_cpu_percent', 0),
                    'memory_mb': exp['overall_metrics'].get('avg_memory_mb', 0),
                    'rate_accuracy': exp['stream_metrics'].get('rate_accuracy', 0)
                })
        
        df = pd.DataFrame(summary_data)
        csv_file = self.output_dir / "realtime_matrix_summary.csv"
        df.to_csv(csv_file, index=False)
        
        # Create detailed results CSV
        detailed_data = []
        for exp in experiments:
            if 'error' not in exp:
                for worker_metric in exp['worker_metrics']:
                    detailed_data.append({
                        'strategy': exp['strategy'],
                        'workers': exp['num_workers'],
                        'stream_rate': exp['stream_rate'],
                        'worker_id': worker_metric['worker_id'],
                        'processed_samples': worker_metric['processed_samples'],
                        'queue_size': worker_metric['queue_size']
                    })
        
        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            detailed_csv = self.output_dir / "realtime_worker_details.csv"
            df_detailed.to_csv(detailed_csv, index=False)
    
    def generate_analysis_report(self, results: Dict[str, Any]):
        """Generate analysis report from benchmark results."""
        experiments = results['experiments']
        
        # Filter successful experiments
        successful_experiments = [exp for exp in experiments if 'error' not in exp]
        
        if not successful_experiments:
            logger.warning("No successful experiments to analyze")
            return
        
        # Analysis by strategy
        strategy_analysis = {}
        for strategy in self.strategies:
            strategy_experiments = [exp for exp in successful_experiments 
                                  if exp['strategy'] == strategy]
            
            if strategy_experiments:
                avg_efficiency = np.mean([exp['efficiency'] for exp in strategy_experiments])
                avg_throughput = np.mean([exp['overall_metrics'].get('throughput', 0) 
                                       for exp in strategy_experiments])
                avg_rate_accuracy = np.mean([exp['stream_metrics'].get('rate_accuracy', 0) 
                                          for exp in strategy_experiments])
                
                strategy_analysis[strategy] = {
                    'avg_efficiency': avg_efficiency,
                    'avg_throughput': avg_throughput,
                    'avg_rate_accuracy': avg_rate_accuracy,
                    'experiment_count': len(strategy_experiments)
                }
        
        # Analysis by worker count
        worker_analysis = {}
        for workers in self.worker_counts:
            worker_experiments = [exp for exp in successful_experiments 
                                if exp['num_workers'] == workers]
            
            if worker_experiments:
                avg_efficiency = np.mean([exp['efficiency'] for exp in worker_experiments])
                avg_throughput = np.mean([exp['overall_metrics'].get('throughput', 0) 
                                       for exp in worker_experiments])
                
                worker_analysis[workers] = {
                    'avg_efficiency': avg_efficiency,
                    'avg_throughput': avg_throughput,
                    'experiment_count': len(worker_experiments)
                }
        
        # Save analysis report
        analysis_report = {
            'strategy_analysis': strategy_analysis,
            'worker_analysis': worker_analysis,
            'total_successful_experiments': len(successful_experiments),
            'total_failed_experiments': len(experiments) - len(successful_experiments)
        }
        
        report_file = self.output_dir / "realtime_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(analysis_report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("REAL-TIME STREAMING MATRIX BENCHMARK ANALYSIS")
        print("="*80)
        
        print(f"\nTotal Experiments: {len(experiments)}")
        print(f"Successful: {len(successful_experiments)}")
        print(f"Failed: {len(experiments) - len(successful_experiments)}")
        
        print(f"\nStrategy Analysis:")
        for strategy, analysis in strategy_analysis.items():
            print(f"  {strategy.upper()}:")
            print(f"    Avg Efficiency: {analysis['avg_efficiency']:.2f}%")
            print(f"    Avg Throughput: {analysis['avg_throughput']:.2f} inf/s")
            print(f"    Avg Rate Accuracy: {analysis['avg_rate_accuracy']:.2f}%")
            print(f"    Experiments: {analysis['experiment_count']}")
        
        print(f"\nWorker Count Analysis:")
        for workers, analysis in worker_analysis.items():
            print(f"  {workers} Workers:")
            print(f"    Avg Efficiency: {analysis['avg_efficiency']:.2f}%")
            print(f"    Avg Throughput: {analysis['avg_throughput']:.2f} inf/s")
            print(f"    Experiments: {analysis['experiment_count']}")

def main():
    """Main function for running real-time matrix benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Streaming Matrix Benchmark')
    parser.add_argument('--output-dir', default='results/realtime',
                       help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=5000,
                       help='Maximum samples per experiment')
    parser.add_argument('--stream-rates', nargs='+', type=int, 
                      default=[500],
                      help='Stream rates to test (samples/s)')
    parser.add_argument('--worker-counts', nargs='+', type=int,
                      default=[1, 2, 4],
                      help='Worker counts to test')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = RealTimeMatrixBenchmark(args.output_dir)
    
    # Update configurations if provided
    if args.max_samples:
        benchmark.max_samples = args.max_samples
    if args.stream_rates:
        benchmark.stream_rates = args.stream_rates
    if args.worker_counts:
        benchmark.worker_counts = args.worker_counts
    
    # Run benchmark
    results = benchmark.run_matrix_benchmark()
    
    # Generate analysis
    benchmark.generate_analysis_report(results)
    
    # Generate plots automatically
    print(f"\nGenerating visualization plots...")
    try:
        from realtime_plots import RealTimePlotter
        plotter = RealTimePlotter(results_dir=args.output_dir)
        plotter.create_all_plots()
        print(f"✅ Plots generated successfully in {args.output_dir}/plots/")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {e}")
        print(f"   You can generate plots manually with: python src/realtime_plots.py --results-dir {args.output_dir}")
    
    print(f"\nBenchmark completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
