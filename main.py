#!/usr/bin/env python3
"""
Data Partitioning Experiments - Main Entry Point

This is the main entry point for the data partitioning experiments system.
It provides a unified interface to run all three partitioning approaches.
"""
import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def main():
    """Main entry point for the data partitioning experiments."""
    parser = argparse.ArgumentParser(
        description='Data Partitioning Experiments - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    # Run comprehensive demo
  python main.py --approach uniform       # Run uniform partitioning
  python main.py --approach dynamic       # Run dynamic partitioning  
  python main.py --approach sharded       # Run sharded partitioning
  python main.py --approach all           # Run all approaches
  python main.py --matrix                 # Run matrix benchmark (local)
  python main.py --cloud-matrix           # Run matrix benchmark (cloud)
  python main.py --test                   # Test installation
  python main.py --setup                  # Setup environment
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='Run comprehensive demo of all approaches')
    parser.add_argument('--approach', choices=['uniform', 'dynamic', 'sharded', 'realtime', 'all'],
                       help='Run specific partitioning approach')
    parser.add_argument('--test', action='store_true',
                       help='Test installation and basic functionality')
    parser.add_argument('--setup', action='store_true',
                       help='Setup environment (install dependencies)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of workers to use (default: 2)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples to process (default: 1000)')
    parser.add_argument('--stream-rate', type=int, default=500,
                       help='Real-time stream rate in samples/s (default: 500)')
    parser.add_argument('--matrix', action='store_true',
                       help='Run matrix benchmark for sizes 1,2,4 across approaches')
    parser.add_argument('--realtime-matrix', action='store_true',
                       help='Run real-time streaming matrix benchmark')
    parser.add_argument('--realtime-plots', action='store_true',
                       help='Generate visualization plots for real-time streaming results')
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
    
    if args.realtime_matrix:
        print("🚀 Running real-time streaming matrix benchmark...")
        os.system(f"{sys.executable} src/realtime_matrix_benchmark.py --max-samples {args.samples} --stream-rates 100 250 500 750 1000")
        return
    
    if args.realtime_plots:
        print("📊 Generating real-time streaming visualization plots...")
        os.system(f"{sys.executable} src/realtime_plots.py --results-dir results/realtime")
        return
    
    if args.cloud_matrix:
        print("☁️ Running cloud matrix benchmark for fair comparison...")
        os.system(f"{sys.executable} cloud/run_cloud_matrix.py --sizes 1,2,4 --samples {args.samples} --batch-size 128")
        return

    if args.approach:
        if args.approach == 'realtime':
            print(f"🚀 Running real-time streaming partitioning...")
            os.system(f"{sys.executable} src/approaches/realtime_streaming.py --workers {args.workers} --samples-per-second {args.stream_rate} --max-samples {args.samples}")
        else:
            print(f"🏃 Running {args.approach} partitioning...")
            os.system(f"{sys.executable} local/run_local_experiments.py --approach {args.approach} --workers {args.workers} --samples {args.samples}")
        return
    
    # If no arguments provided, show help
    parser.print_help()

if __name__ == "__main__":
    main()
