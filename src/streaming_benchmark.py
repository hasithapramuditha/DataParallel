"""
Comprehensive Streaming Benchmark for DataParallel

This module implements a complete streaming benchmark system that:
1. Creates CIFAR-10 data streams at configurable rates
2. Tests all three partitioning strategies (Uniform, Dynamic, Sharded)
3. Tests across different node configurations (1, 2, 4 nodes)
4. Runs multiple iterations and calculates averages
5. Provides comprehensive performance analysis
"""
import os
import sys
import time
import json
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from approaches.uniform_partitioning import run_uniform_partitioning
from approaches.dynamic_partitioning import run_dynamic_partitioning
from approaches.sharded_partitioning import run_sharded_partitioning
from config import PartitioningConfig
from utils import get_raw_data, MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CIFAR10StreamGenerator:
    """Generates CIFAR-10 data streams at configurable rates."""
    
    def __init__(self, stream_rate: int = 1000, total_samples: int = 60000):
        """
        Initialize CIFAR-10 stream generator.
        
        Args:
            stream_rate: Samples per second
            total_samples: Total number of samples in CIFAR-10 (60000)
        """
        self.stream_rate = stream_rate
        self.total_samples = total_samples
        self.stream_duration = total_samples / stream_rate  # Total time for full stream
        self.samples_per_stream = stream_rate  # Samples in each 1-second stream
        self.num_streams = int(total_samples / stream_rate)  # Number of 1-second streams
        
        logger.info(f"Stream Configuration:")
        logger.info(f"  - Stream Rate: {stream_rate} samples/s")
        logger.info(f"  - Total Samples: {total_samples}")
        logger.info(f"  - Stream Duration: {self.stream_duration:.1f} seconds")
        logger.info(f"  - Samples per Stream: {self.samples_per_stream}")
        logger.info(f"  - Number of Streams: {self.num_streams}")
    
    def generate_stream_batch(self, stream_index: int) -> np.ndarray:
        """
        Generate a batch of data for a specific stream.
        
        Args:
            stream_index: Index of the current stream (0 to num_streams-1)
            
        Returns:
            Batch of CIFAR-10 data for this stream
        """
        # Load all CIFAR-10 data
        all_data = get_raw_data(self.total_samples)
        all_data = np.array(all_data)
        
        # Calculate start and end indices for this stream
        start_idx = stream_index * self.samples_per_stream
        end_idx = min(start_idx + self.samples_per_stream, self.total_samples)
        
        # Extract stream batch
        stream_batch = all_data[start_idx:end_idx]
        
        logger.debug(f"Stream {stream_index}: Generated batch of shape {stream_batch.shape}")
        return stream_batch

class StreamingPartitioningBenchmark:
    """Main benchmark class for streaming partitioning strategies."""
    
    def __init__(self, stream_rate: int = 1000, total_samples: int = 60000, iterations: int = 3, node_counts: List[int] = None):
        """
        Initialize streaming benchmark.
        
        Args:
            stream_rate: Samples per second
            total_samples: Total CIFAR-10 samples
            iterations: Number of iterations for averaging
            node_counts: List of node counts to test (default: [1, 2, 4, 8])
        """
        self.stream_generator = CIFAR10StreamGenerator(stream_rate, total_samples)
        self.iterations = iterations
        self.results = {}
        
        # Define test configurations
        self.strategies = ['uniform', 'dynamic', 'sharded']
        self.node_counts = node_counts if node_counts is not None else [1, 2, 4, 8]
        
        logger.info(f"Benchmark Configuration:")
        logger.info(f"  - Strategies: {self.strategies}")
        logger.info(f"  - Node Counts: {self.node_counts}")
        logger.info(f"  - Iterations: {iterations}")
        logger.info(f"  - Total Tests: {len(self.strategies) * len(self.node_counts) * iterations}")
    
    def _run_uniform_streaming(self, stream_data: np.ndarray, nodes: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run uniform partitioning for streaming data.
        
        Args:
            stream_data: The stream data to process
            nodes: Number of nodes
            config: Configuration object
            
        Returns:
            Results dictionary
        """
        import torch
        from utils import get_model, MetricsCollector
        
        # Load model
        model = get_model()
        model.eval()
        
        # Convert stream data to tensor
        data_tensor = torch.tensor(stream_data, dtype=torch.float32)
        
        # Create metrics collector
        metrics = MetricsCollector()
        metrics.start_timing()
        
        # Process data uniformly across nodes
        batch_size = config['batch_size']
        total_inferences = 0
        
        with torch.no_grad():
            # Split data into batches
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i:i + batch_size]
                
                # Perform inference
                _ = model(batch)
                total_inferences += len(batch)
                
                # Record metrics periodically
                if i % (batch_size * 10) == 0:
                    metrics.record_cpu_usage()
                    metrics.record_memory_usage()
        
        metrics.stop_timing()
        metrics.add_inferences(total_inferences)
        
        result_metrics = metrics.get_metrics()
        
        # Debug logging
        logger.debug(f"Uniform streaming metrics: {result_metrics}")
        logger.debug(f"Total inferences: {total_inferences}")
        logger.debug(f"Duration: {result_metrics.get('total_duration', 0)}")
        
        # For multi-node scenarios, simulate uniform distribution
        if nodes > 1:
            # Simulate uniform distribution across nodes
            node_throughput = result_metrics['throughput'] / nodes
            node_cpu = result_metrics['avg_cpu_percent'] / nodes
            node_memory = result_metrics['avg_memory_mb'] / nodes
            node_inferences = total_inferences // nodes
            
            nodewise = []
            for rank in range(nodes):
                nodewise.append({
                    'rank': rank,
                    'throughput': node_throughput,
                    'latency_ms': result_metrics['latency_ms'],
                    'avg_cpu_percent': node_cpu,
                    'avg_memory_mb': node_memory,
                    'total_duration': result_metrics['total_duration'],
                    'total_inferences': node_inferences
                })
        else:
            nodewise = [{
                'rank': 0,
                'throughput': result_metrics['throughput'],
                'latency_ms': result_metrics['latency_ms'],
                'avg_cpu_percent': result_metrics['avg_cpu_percent'],
                'avg_memory_mb': result_metrics['avg_memory_mb'],
                'total_duration': result_metrics['total_duration'],
                'total_inferences': result_metrics['total_inferences']
            }]
        
        aggregate = {
            'throughput': result_metrics['throughput'],
            'latency_ms': result_metrics['latency_ms'],
            'avg_cpu_percent': result_metrics['avg_cpu_percent'],
            'avg_memory_mb': result_metrics['avg_memory_mb'],
            'total_duration': result_metrics['total_duration'],
            'total_inferences': result_metrics['total_inferences']
        }
        
        return {
            'aggregate': aggregate,
            'nodewise': nodewise
        }
    
    def run_single_test(self, strategy: str, nodes: int, iteration: int) -> Dict[str, Any]:
        """
        Run a single test configuration.
        
        Args:
            strategy: Partitioning strategy ('uniform', 'dynamic', 'sharded')
            nodes: Number of nodes
            iteration: Iteration number
            
        Returns:
            Test results dictionary
        """
        logger.info(f"Running {strategy.upper()} with {nodes} node(s) - Iteration {iteration + 1}")
        
        # Create configuration
        config = PartitioningConfig({
            'num_workers': nodes,
            'num_samples': self.stream_generator.samples_per_stream,
            'batch_size': 128,
            'chunk_size': 32,
            'num_shards': nodes
        })
        
        # Run all streams for this test
        stream_results = []
        total_start_time = time.time()
        
        for stream_idx in range(self.stream_generator.num_streams):
            logger.debug(f"  Processing stream {stream_idx + 1}/{self.stream_generator.num_streams}")
            
            # Generate stream data
            stream_data = self.stream_generator.generate_stream_batch(stream_idx)
            
            # Run partitioning strategy
            stream_start_time = time.time()
            
            try:
                if strategy == 'uniform':
                    # For uniform partitioning in streaming, use a simpler approach
                    result = self._run_uniform_streaming(stream_data, nodes, config.get_uniform_config())
                elif strategy == 'dynamic':
                    result = run_dynamic_partitioning(config.get_dynamic_config())
                elif strategy == 'sharded':
                    result = run_sharded_partitioning(config.get_sharded_config())
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                stream_duration = time.time() - stream_start_time
                
                # Extract metrics - handle both direct results and aggregate/nodewise structure
                if 'aggregate' in result:
                    # Results from _run_uniform_streaming
                    aggregate = result['aggregate']
                    stream_metrics = {
                        'stream_index': stream_idx,
                        'duration': stream_duration,
                        'throughput': aggregate.get('throughput', 0),
                        'total_inferences': aggregate.get('total_inferences', len(stream_data)),
                        'avg_cpu_percent': aggregate.get('avg_cpu_percent', 0),
                        'avg_memory_mb': aggregate.get('avg_memory_mb', 0),
                        'strategy': strategy,
                        'nodes': nodes,
                        'iteration': iteration
                    }
                else:
                    # Direct results from other strategies
                    stream_metrics = {
                        'stream_index': stream_idx,
                        'duration': stream_duration,
                        'throughput': result.get('throughput', 0),
                        'total_inferences': result.get('total_inferences', len(stream_data)),
                        'avg_cpu_percent': result.get('avg_cpu_percent', 0),
                        'avg_memory_mb': result.get('avg_memory_mb', 0),
                        'strategy': strategy,
                        'nodes': nodes,
                        'iteration': iteration
                    }
                
                stream_results.append(stream_metrics)
                
            except Exception as e:
                logger.error(f"Error in stream {stream_idx}: {e}")
                stream_metrics = {
                    'stream_index': stream_idx,
                    'duration': 0,
                    'throughput': 0,
                    'total_inferences': 0,
                    'avg_cpu_percent': 0,
                    'avg_memory_mb': 0,
                    'strategy': strategy,
                    'nodes': nodes,
                    'iteration': iteration,
                    'error': str(e)
                }
                stream_results.append(stream_metrics)
        
        total_duration = time.time() - total_start_time
        
        # Calculate aggregate metrics
        successful_streams = [r for r in stream_results if 'error' not in r]
        failed_streams = [r for r in stream_results if 'error' in r]
        
        if successful_streams:
            avg_throughput = np.mean([r['throughput'] for r in successful_streams])
            avg_cpu = np.mean([r['avg_cpu_percent'] for r in successful_streams])
            avg_memory = np.mean([r['avg_memory_mb'] for r in successful_streams])
            total_inferences = sum([r['total_inferences'] for r in successful_streams])
        else:
            avg_throughput = 0
            avg_cpu = 0
            avg_memory = 0
            total_inferences = 0
        
        test_result = {
            'strategy': strategy,
            'nodes': nodes,
            'iteration': iteration,
            'total_duration': total_duration,
            'avg_throughput': avg_throughput,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_mb': avg_memory,
            'total_inferences': total_inferences,
            'successful_streams': len(successful_streams),
            'failed_streams': len(failed_streams),
            'stream_results': stream_results
        }
        
        logger.info(f"  Completed: {avg_throughput:.2f} inf/s, {len(successful_streams)}/{len(stream_results)} streams successful")
        return test_result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test configurations and return comprehensive results."""
        logger.info("Starting comprehensive streaming benchmark...")
        
        all_results = []
        start_time = time.time()
        
        # Run all combinations
        for strategy in self.strategies:
            for nodes in self.node_counts:
                for iteration in range(self.iterations):
                    result = self.run_single_test(strategy, nodes, iteration)
                    all_results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate averages across iterations
        averaged_results = self._calculate_averages(all_results)
        
        # Generate summary
        summary = {
            'benchmark_config': {
                'stream_rate': self.stream_generator.stream_rate,
                'total_samples': self.stream_generator.total_samples,
                'num_streams': self.stream_generator.num_streams,
                'iterations': self.iterations,
                'strategies': self.strategies,
                'node_counts': self.node_counts
            },
            'execution_time': total_time,
            'detailed_results': all_results,
            'averaged_results': averaged_results,
            'performance_summary': self._generate_performance_summary(averaged_results)
        }
        
        logger.info(f"Benchmark completed in {total_time:.2f} seconds")
        return summary
    
    def _calculate_averages(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate average metrics across iterations for each strategy-node combination."""
        averaged = {}
        
        for strategy in self.strategies:
            averaged[strategy] = {}
            for nodes in self.node_counts:
                # Get all results for this strategy-node combination
                strategy_node_results = [
                    r for r in all_results 
                    if r['strategy'] == strategy and r['nodes'] == nodes
                ]
                
                if strategy_node_results:
                    # Calculate averages
                    avg_throughput = np.mean([r['avg_throughput'] for r in strategy_node_results])
                    avg_cpu = np.mean([r['avg_cpu_percent'] for r in strategy_node_results])
                    avg_memory = np.mean([r['avg_memory_mb'] for r in strategy_node_results])
                    avg_duration = np.mean([r['total_duration'] for r in strategy_node_results])
                    total_inferences = sum([r['total_inferences'] for r in strategy_node_results])
                    successful_streams = sum([r['successful_streams'] for r in strategy_node_results])
                    failed_streams = sum([r['failed_streams'] for r in strategy_node_results])
                    
                    averaged[strategy][nodes] = {
                        'avg_throughput': avg_throughput,
                        'avg_cpu_percent': avg_cpu,
                        'avg_memory_mb': avg_memory,
                        'avg_duration': avg_duration,
                        'total_inferences': total_inferences,
                        'successful_streams': successful_streams,
                        'failed_streams': failed_streams,
                        'success_rate': successful_streams / (successful_streams + failed_streams) if (successful_streams + failed_streams) > 0 else 0
                    }
        
        return averaged
    
    def _generate_performance_summary(self, averaged_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary and rankings."""
        summary = {
            'best_throughput': {},
            'best_efficiency': {},
            'scaling_analysis': {}
        }
        
        # Find best throughput for each node count
        for nodes in self.node_counts:
            best_throughput = 0
            best_strategy = None
            
            for strategy in self.strategies:
                if strategy in averaged_results and nodes in averaged_results[strategy]:
                    throughput = averaged_results[strategy][nodes]['avg_throughput']
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_strategy = strategy
            
            summary['best_throughput'][nodes] = {
                'strategy': best_strategy,
                'throughput': best_throughput
            }
        
        # Calculate scaling efficiency
        for strategy in self.strategies:
            if strategy in averaged_results:
                scaling_data = []
                for nodes in self.node_counts:
                    if nodes in averaged_results[strategy]:
                        scaling_data.append({
                            'nodes': nodes,
                            'throughput': averaged_results[strategy][nodes]['avg_throughput']
                        })
                
                if len(scaling_data) > 1:
                    # Calculate scaling efficiency (linear scaling = 1.0)
                    base_throughput = scaling_data[0]['throughput']
                    scaling_efficiency = []
                    
                    for data in scaling_data[1:]:
                        expected_throughput = base_throughput * data['nodes']
                        actual_throughput = data['throughput']
                        efficiency = actual_throughput / expected_throughput if expected_throughput > 0 else 0
                        scaling_efficiency.append(efficiency)
                    
                    summary['scaling_analysis'][strategy] = {
                        'scaling_efficiency': np.mean(scaling_efficiency),
                        'scaling_data': scaling_data
                    }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results/streaming"):
        """Save benchmark results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        with open(f"{output_dir}/detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save averaged results as CSV
        csv_data = []
        for strategy in self.strategies:
            for nodes in self.node_counts:
                if strategy in results['averaged_results'] and nodes in results['averaged_results'][strategy]:
                    row = results['averaged_results'][strategy][nodes].copy()
                    row['strategy'] = strategy
                    row['nodes'] = nodes
                    csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(f"{output_dir}/averaged_results.csv", index=False)
        
        # Save performance summary
        with open(f"{output_dir}/performance_summary.json", 'w') as f:
            json.dump(results['performance_summary'], f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}/")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of results."""
        print("\n" + "="*80)
        print("STREAMING BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        # Print configuration
        config = results['benchmark_config']
        print(f"\nBenchmark Configuration:")
        print(f"  Stream Rate: {config['stream_rate']} samples/s")
        print(f"  Total Samples: {config['total_samples']}")
        print(f"  Number of Streams: {config['num_streams']}")
        print(f"  Iterations: {config['iterations']}")
        print(f"  Execution Time: {results['execution_time']:.2f} seconds")
        
        # Print results table
        print(f"\nPerformance Results (Average Across {config['iterations']} Iterations):")
        print("-" * 80)
        print(f"{'Strategy':<10} {'Nodes':<6} {'Throughput':<12} {'CPU %':<8} {'Memory MB':<12} {'Success %':<10}")
        print("-" * 80)
        
        for strategy in self.strategies:
            for nodes in self.node_counts:
                if strategy in results['averaged_results'] and nodes in results['averaged_results'][strategy]:
                    data = results['averaged_results'][strategy][nodes]
                    print(f"{strategy:<10} {nodes:<6} {data['avg_throughput']:<12.2f} {data['avg_cpu_percent']:<8.1f} {data['avg_memory_mb']:<12.1f} {data['success_rate']*100:<10.1f}")
        
        # Print best performers
        print(f"\nBest Performers:")
        print("-" * 40)
        for nodes in self.node_counts:
            if nodes in results['performance_summary']['best_throughput']:
                best = results['performance_summary']['best_throughput'][nodes]
                print(f"  {nodes} Node(s): {best['strategy'].upper()} ({best['throughput']:.2f} inf/s)")
        
        # Print scaling analysis
        print(f"\nScaling Efficiency:")
        print("-" * 40)
        for strategy in self.strategies:
            if strategy in results['performance_summary']['scaling_analysis']:
                efficiency = results['performance_summary']['scaling_analysis'][strategy]['scaling_efficiency']
                print(f"  {strategy.upper()}: {efficiency:.2f} (1.0 = perfect linear scaling)")
        
        print("="*80)

def main():
    """Main function for running streaming benchmark."""
    parser = argparse.ArgumentParser(description='Comprehensive Streaming Benchmark for DataParallel')
    parser.add_argument('--stream-rate', type=int, default=1000, help='Stream rate in samples per second')
    parser.add_argument('--total-samples', type=int, default=60000, help='Total CIFAR-10 samples to process')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations for averaging')
    parser.add_argument('--strategies', nargs='+', default=['uniform', 'dynamic', 'sharded'], 
                       help='Partitioning strategies to test')
    parser.add_argument('--nodes', type=str, default='1,2,4', 
                       help='Comma-separated list of node counts to test (default: 1,2,4)')
    parser.add_argument('--output-dir', type=str, default='results/streaming', 
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse nodes argument
    node_counts = [int(x.strip()) for x in args.nodes.split(',')]
    
    # Create and run benchmark
    benchmark = StreamingPartitioningBenchmark(
        stream_rate=args.stream_rate,
        total_samples=args.total_samples,
        iterations=args.iterations,
        node_counts=node_counts
    )
    
    # Override strategies if provided
    benchmark.strategies = args.strategies
    
    # Run benchmark
    results = benchmark.run_all_tests()
    
    # Save and display results
    benchmark.save_results(results, args.output_dir)
    benchmark.print_summary(results)
    
    # Automatically generate plots after benchmark completion
    print("\n🎨 Generating comprehensive plots...")
    try:
        import subprocess
        import sys
        import os
        
        # Get the project root directory (parent of src/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plot_script = os.path.join(project_root, "plot_streaming_results.py")
        
        if os.path.exists(plot_script):
            print("📊 Running plotting script...")
            result = subprocess.run([sys.executable, plot_script], 
                                  cwd=project_root, 
                                  capture_output=True, 
                                  text=True)
            
            if result.returncode == 0:
                print("✅ Plots generated successfully!")
                print("📁 Plots saved to: results/plots/")
            else:
                print("⚠️  Plot generation completed with warnings:")
                print(result.stdout)
                if result.stderr:
                    print("Errors:", result.stderr)
        else:
            print("⚠️  Plotting script not found at:", plot_script)
            
    except Exception as e:
        print(f"⚠️  Could not generate plots automatically: {e}")
        print("💡 You can manually run: python plot_streaming_results.py")
    
    return results

if __name__ == "__main__":
    main()
