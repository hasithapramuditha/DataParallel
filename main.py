#!/usr/bin/env python3
"""
Data Partitioning Experiments - Main Entry Point

This is the main entry point for the data partitioning experiments system.
It provides a unified interface to run all three partitioning approaches with streaming benchmarks.
"""
import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import new configuration system
from config import PartitioningConfig

def main():
    """Main entry point for the data partitioning experiments."""
    parser = argparse.ArgumentParser(
        description='Data Partitioning Experiments - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --streaming-benchmark                    # Run comprehensive streaming benchmark
  python main.py --streaming-benchmark --stream-rate 500  # Custom stream rate
  python main.py --streaming-benchmark --nodes 1,2,4      # Custom node counts
  python main.py --streaming-benchmark --nodes 1,2,4,8,16 # Test up to 16 nodes
  python main.py --approach uniform                       # Run uniform partitioning
  python main.py --approach dynamic                       # Run dynamic partitioning  
  python main.py --approach sharded                       # Run sharded partitioning
  python main.py --approach all                           # Run all approaches
  python main.py --matrix                                 # Run matrix benchmark (local)
  python main.py --cloud-matrix                           # Run matrix benchmark (cloud)
  python main.py --test                                   # Test installation
  python main.py --setup                                  # Setup environment
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='Run comprehensive demo of all approaches')
    parser.add_argument('--approach', choices=['uniform', 'dynamic', 'sharded', 'all'],
                       help='Run specific partitioning approach')
    parser.add_argument('--test', action='store_true',
                       help='Test installation and basic functionality')
    parser.add_argument('--setup', action='store_true',
                       help='Setup environment (install dependencies)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of workers to use (default: 2)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples to process (default: 1000)')
    parser.add_argument('--stream-rate', type=int, default=1000,
                       help='Stream rate for streaming experiments (samples per second)')
    parser.add_argument('--total-samples', type=int, default=60000,
                       help='Total CIFAR-10 samples to process (default: 60000)')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of iterations for averaging (default: 3)')
    parser.add_argument('--nodes', type=str, default='1,2,4,8',
                       help='Comma-separated list of node counts to test (default: 1,2,4,8)')
    parser.add_argument('--streaming-benchmark', action='store_true',
                       help='Run comprehensive streaming benchmark (3 strategies x custom node counts)')
    parser.add_argument('--matrix', action='store_true',
                       help='Run matrix benchmark for sizes 1,2,4 across approaches')
    parser.add_argument('--cloud-matrix', action='store_true',
                       help='Run matrix benchmark on cloud cluster for fair comparison')
    
    args = parser.parse_args()
    
    if args.setup:
        print("🚀 Setting up Data Partitioning Experiments...")
        os.system(f"{sys.executable} scripts/setup_local_venv.py")
        return
    
    if args.test:
        print("🧪 Testing installation...")
        os.system(f"{sys.executable} scripts/test_installation.py")
        return
    
    if args.demo:
        print("🎯 Running comprehensive demo...")
        os.system(f"{sys.executable} examples/demo_all_approaches.py")
        return
    
    if args.matrix:
        print("📊 Running matrix benchmark (1,2,4 workers) across approaches...")
        os.system(f"{sys.executable} src/matrix_benchmark.py --sizes 1,2,4 --num-samples {args.samples} --batch-size 128")
        return
    
    if args.streaming_benchmark:
        # Parse nodes argument
        node_counts = [int(x.strip()) for x in args.nodes.split(',')]
        num_configs = len(node_counts)
        
        print("🚀 Running comprehensive streaming benchmark...")
        print(f"   Stream Rate: {args.stream_rate} samples/s")
        print(f"   Total Samples: {args.total_samples}")
        print(f"   Iterations: {args.iterations}")
        print(f"   Node Counts: {node_counts}")
        print(f"   Testing: 3 strategies x {num_configs} node counts = {3 * num_configs} configurations")
        os.system(f"{sys.executable} src/streaming_benchmark.py --stream-rate {args.stream_rate} --total-samples {args.total_samples} --iterations {args.iterations} --nodes {args.nodes}")
        return
    
    if args.cloud_matrix:
        print("☁️ Running cloud matrix benchmark for fair comparison...")
        os.system(f"{sys.executable} cloud/run_cloud_experiments.py --matrix --sizes 1,2,4 --samples {args.samples} --batch-size 128")
        return

    if args.approach:
        if args.approach == 'all':
            print(f"🏃 Running all partitioning approaches...")
            from approaches.uniform_partitioning import run_uniform_partitioning
            from approaches.dynamic_partitioning import run_dynamic_partitioning
            from approaches.sharded_partitioning import run_sharded_partitioning
            
            # Create configuration
            config = PartitioningConfig({
                'num_workers': args.workers,
                'num_samples': args.samples,
                'batch_size': 128,
                'chunk_size': 32
            })
            
            approaches = [
                ('uniform', run_uniform_partitioning, config.get_uniform_config()),
                ('dynamic', run_dynamic_partitioning, config.get_dynamic_config()),
                ('sharded', run_sharded_partitioning, config.get_sharded_config())
            ]
            
            for name, func, cfg in approaches:
                print(f"\n📊 Running {name} partitioning...")
                try:
                    result = func(cfg)
                    if 'aggregate' in result:
                        print(f"✅ {name}: {result['aggregate']['throughput']:.2f} inf/s")
                    else:
                        print(f"✅ {name}: {result['throughput']:.2f} inf/s")
                except Exception as e:
                    print(f"❌ {name}: {e}")
        else:
            print(f"🏃 Running {args.approach} partitioning...")
            # Use direct approach execution instead of local/run_local_experiments.py
            from approaches.uniform_partitioning import run_uniform_partitioning
            from approaches.dynamic_partitioning import run_dynamic_partitioning
            from approaches.sharded_partitioning import run_sharded_partitioning
            
            config = PartitioningConfig({
                'num_workers': args.workers,
                'num_samples': args.samples,
                'batch_size': 128,
                'chunk_size': 32
            })
            
            try:
                if args.approach == 'uniform':
                    result = run_uniform_partitioning(config.get_uniform_config())
                elif args.approach == 'dynamic':
                    result = run_dynamic_partitioning(config.get_dynamic_config())
                elif args.approach == 'sharded':
                    result = run_sharded_partitioning(config.get_sharded_config())
                
                if 'aggregate' in result:
                    print(f"✅ {args.approach}: {result['aggregate']['throughput']:.2f} inf/s")
                else:
                    print(f"✅ {args.approach}: {result['throughput']:.2f} inf/s")
            except Exception as e:
                print(f"❌ {args.approach}: {e}")
        return
    
    # If no arguments provided, show help
    parser.print_help()

if __name__ == "__main__":
    main()
