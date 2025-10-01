"""
Local experiment runner for data partitioning approaches.

This script runs all three partitioning approaches locally using
multiple processes to simulate a distributed environment.
"""
import os
import sys
import yaml
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from benchmark import BenchmarkRunner
from utils import print_results_summary

logger = logging.getLogger(__name__)

def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(__file__).parent / config_file
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config

def run_single_approach(approach: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single partitioning approach.
    
    Args:
        approach: Name of the approach to run
        config: Configuration dictionary
        
    Returns:
        Results dictionary
    """
    logger.info(f"Running {approach} partitioning approach...")
    
    # Import the specific approach
    if approach == "uniform":
        from src.approaches.uniform_partitioning import run_uniform_partitioning
        approach_config = {
            'world_size': config.get('num_workers', 4),
            'batch_size': config.get('batch_size', 128),
            'num_workers': config.get('data_loader_workers', 1),
            'master_addr': config.get('master_addr', 'localhost'),
            'master_port': config.get('master_port', '12355')
        }
        results = run_uniform_partitioning(approach_config)
        
    elif approach == "dynamic":
        from src.approaches.dynamic_partitioning import run_dynamic_partitioning
        approach_config = {
            'num_workers': config.get('num_workers', 4),
            'num_samples': config.get('num_samples', 10000),
            'chunk_size': config.get('chunk_size', 128),
            'ray_address': config.get('ray_address')
        }
        results = run_dynamic_partitioning(approach_config)
        
    elif approach == "sharded":
        from src.approaches.sharded_partitioning import run_sharded_partitioning
        approach_config = {
            'num_workers': config.get('num_workers', 4),
            'num_samples': config.get('num_samples', 10000),
            'num_shards': config.get('num_shards', 4),
            'dask_scheduler_address': config.get('dask_scheduler_address')
        }
        results = run_sharded_partitioning(approach_config)
        
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    results['approach'] = approach
    return results

def run_all_approaches(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all partitioning approaches.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing all results
    """
    approaches = ['uniform', 'dynamic', 'sharded']
    all_results = {}
    
    for approach in approaches:
        try:
            logger.info(f"Starting {approach} approach...")
            start_time = time.time()
            
            results = run_single_approach(approach, config)
            results['execution_time'] = time.time() - start_time
            
            all_results[approach] = results
            
            logger.info(f"{approach} approach completed in {results['execution_time']:.2f} seconds")
            
            # Small delay between approaches
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"{approach} approach failed: {str(e)}")
            all_results[approach] = {'error': str(e)}
    
    return all_results

def run_scalability_test(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run scalability test with different numbers of workers.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing scalability results
    """
    worker_counts = [1, 2, 4, 8] if config.get('num_workers', 4) >= 8 else [1, 2, 4]
    scalability_results = {}
    
    for num_workers in worker_counts:
        logger.info(f"Running scalability test with {num_workers} workers...")
        
        # Update config for this test
        test_config = config.copy()
        test_config['num_workers'] = num_workers
        
        # Run all approaches with this worker count
        results = run_all_approaches(test_config)
        scalability_results[f'{num_workers}_workers'] = results
        
        logger.info(f"Scalability test with {num_workers} workers completed")
    
    return scalability_results

def save_results(results: Dict[str, Any], output_file: str):
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        output_file: Output file path
    """
    import json
    
    output_path = Path(__file__).parent.parent / "results" / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

def generate_summary_report(results: Dict[str, Any]) -> str:
    """
    Generate a summary report of the results.
    
    Args:
        results: Results dictionary
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("="*80)
    report.append("LOCAL DATA PARTITIONING EXPERIMENT RESULTS")
    report.append("="*80)
    report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary table
    report.append("PERFORMANCE SUMMARY")
    report.append("-" * 80)
    report.append(f"{'Approach':<15} {'Throughput':<15} {'Latency':<15} {'CPU %':<10} {'Memory MB':<12}")
    report.append("-" * 80)
    
    for approach, data in results.items():
        if isinstance(data, dict) and 'error' not in data:
            throughput = data.get('throughput', 0)
            latency = data.get('latency_ms', 0)
            cpu = data.get('avg_cpu_percent', 0)
            memory = data.get('avg_memory_mb', 0)
            
            report.append(f"{approach:<15} {throughput:<15.2f} {latency:<15.2f} {cpu:<10.1f} {memory:<12.1f}")
        else:
            report.append(f"{approach:<15} {'ERROR':<15} {'ERROR':<15} {'ERROR':<10} {'ERROR':<12}")
    
    report.append("")
    
    # Detailed results
    for approach, data in results.items():
        report.append(f"{approach.upper()} DETAILS")
        report.append("-" * 40)
        
        if isinstance(data, dict) and 'error' not in data:
            for key, value in data.items():
                if key != 'approach':
                    report.append(f"{key}: {value}")
        else:
            report.append(f"Error: {data.get('error', 'Unknown error')}")
        
        report.append("")
    
    return "\n".join(report)

def main():
    """Main function for running local experiments."""
    parser = argparse.ArgumentParser(description='Run local data partitioning experiments')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--approach', type=str, choices=['uniform', 'dynamic', 'sharded', 'all'], 
                       default='all', help='Approach to run')
    parser.add_argument('--scalability', action='store_true', help='Run scalability test')
    parser.add_argument('--output', type=str, default='local_results.json', help='Output file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Run experiments
        if args.scalability:
            logger.info("Running scalability test...")
            results = run_scalability_test(config)
            output_file = f"scalability_{args.output}"
        elif args.approach == 'all':
            logger.info("Running all approaches...")
            results = run_all_approaches(config)
            output_file = args.output
        else:
            logger.info(f"Running {args.approach} approach...")
            results = {args.approach: run_single_approach(args.approach, config)}
            output_file = f"{args.approach}_{args.output}"
        
        # Save results
        save_results(results, output_file)
        
        # Generate and print summary
        summary = generate_summary_report(results)
        print("\n" + summary)
        
        # Save summary to file
        summary_file = Path(__file__).parent.parent / "results" / f"summary_{output_file.replace('.json', '.txt')}"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Results saved to results/{output_file}")
        logger.info(f"Summary saved to results/summary_{output_file.replace('.json', '.txt')}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
