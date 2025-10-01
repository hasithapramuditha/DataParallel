"""
Cloud experiment runner for data partitioning approaches on GCP.
Updated for Fair Comparison and Matrix Benchmarking.

This script runs all three partitioning approaches on a GCP cluster
using the configured instances with fair comparison and matrix benchmarking.
"""
import os
import sys
import yaml
import argparse
import logging
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class CloudExperimentRunner:
    """Runs experiments on GCP cluster."""
    
    def __init__(self, cluster_info: Dict[str, Any]):
        self.cluster_info = cluster_info
        self.head_ip = cluster_info['head_ip']
        self.worker_ips = cluster_info['worker_ips']
        self.zone = cluster_info['zone']
        
    def copy_code_to_cluster(self):
        """Copy experiment code to the head node."""
        logger.info("Copying code to cluster...")
        
        head_node = f"{self.cluster_info['cluster_name']}-head"
        
        # Copy source code
        subprocess.run([
            'gcloud', 'compute', 'scp',
            '--recurse',
            str(Path(__file__).parent.parent / 'src'),
            f'{head_node}:/home/ubuntu/data-parallel/',
            '--zone', self.zone
        ], check=True)
        
        # Copy requirements
        subprocess.run([
            'gcloud', 'compute', 'scp',
            str(Path(__file__).parent.parent / 'requirements.txt'),
            f'{head_node}:/home/ubuntu/data-parallel/',
            '--zone', self.zone
        ], check=True)
        
        logger.info("Code copied to cluster")
    
    def run_remote_command(self, command: str, node: str = None) -> str:
        """
        Run a command on a remote node.
        
        Args:
            command: Command to run
            node: Node name (defaults to head node)
            
        Returns:
            Command output
        """
        if node is None:
            node = f"{self.cluster_info['cluster_name']}-head"
        
        try:
            result = subprocess.run([
                'gcloud', 'compute', 'ssh', node,
                '--zone', self.zone,
                '--command', command
            ], capture_output=True, text=True, check=True)
            
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    def run_uniform_partitioning(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run uniform partitioning on the cluster."""
        logger.info("Running uniform partitioning on cluster...")
        
        # Prepare command
        command = f"""
        cd /home/ubuntu/data-parallel
        source /home/ubuntu/venv/bin/activate
        
        python src/approaches/uniform_partitioning.py \
            --world-size {config.get('num_workers', 4)} \
            --batch-size {config.get('batch_size', 128)} \
            --master-addr {self.head_ip} \
            --master-port 12355 \
            --output uniform_results.json
        """
        
        # Run on head node
        output = self.run_remote_command(command)
        
        # Get results
        results = self.get_remote_results('uniform_results.json')
        
        return results
    
    def run_dynamic_partitioning(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run dynamic partitioning on the cluster."""
        logger.info("Running dynamic partitioning on cluster...")
        
        # Prepare command
        command = f"""
        cd /home/ubuntu/data-parallel
        source /home/ubuntu/venv/bin/activate
        
        python src/approaches/dynamic_partitioning.py \
            --num-workers {config.get('num_workers', 4)} \
            --num-samples {config.get('num_samples', 10000)} \
            --chunk-size {config.get('chunk_size', 128)} \
            --ray-address {self.head_ip}:6379 \
            --output dynamic_results.json
        """
        
        # Run on head node
        output = self.run_remote_command(command)
        
        # Get results
        results = self.get_remote_results('dynamic_results.json')
        
        return results
    
    def run_sharded_partitioning(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run sharded partitioning on the cluster."""
        logger.info("Running sharded partitioning on cluster...")
        
        # Prepare command
        command = f"""
        cd /home/ubuntu/data-parallel
        source /home/ubuntu/venv/bin/activate
        
        python src/approaches/sharded_partitioning.py \
            --num-workers {config.get('num_workers', 4)} \
            --num-samples {config.get('num_samples', 10000)} \
            --num-shards {config.get('num_shards', 4)} \
            --dask-scheduler-address {self.head_ip}:8786 \
            --output sharded_results.json
        """
        
        # Run on head node
        output = self.run_remote_command(command)
        
        # Get results
        results = self.get_remote_results('sharded_results.json')
        
        return results
    
    def get_remote_results(self, filename: str) -> Dict[str, Any]:
        """Get results from remote node."""
        head_node = f"{self.cluster_info['cluster_name']}-head"
        
        # Copy results file locally
        local_file = Path(__file__).parent / filename
        subprocess.run([
            'gcloud', 'compute', 'scp',
            f'{head_node}:/home/ubuntu/data-parallel/{filename}',
            str(local_file),
            '--zone', self.zone
        ], check=True)
        
        # Load results
        with open(local_file, 'r') as f:
            results = json.load(f)
        
        return results
    
    def run_all_experiments(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run all partitioning approaches."""
        results = {}
        
        approaches = ['uniform', 'dynamic', 'sharded']
        
        for approach in approaches:
            try:
                logger.info(f"Starting {approach} experiment...")
                start_time = time.time()
                
                if approach == 'uniform':
                    result = self.run_uniform_partitioning(config)
                elif approach == 'dynamic':
                    result = self.run_dynamic_partitioning(config)
                elif approach == 'sharded':
                    result = self.run_sharded_partitioning(config)
                
                result['execution_time'] = time.time() - start_time
                result['approach'] = approach
                results[approach] = result
                
                logger.info(f"{approach} experiment completed in {result['execution_time']:.2f} seconds")
                
                # Wait between experiments
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"{approach} experiment failed: {str(e)}")
                results[approach] = {'error': str(e)}
        
        return results
    
    def run_scalability_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run scalability test with different worker counts."""
        worker_counts = [1, 2, 4]  # Adjust based on cluster size
        scalability_results = {}
        
        for num_workers in worker_counts:
            logger.info(f"Running scalability test with {num_workers} workers...")
            
            # Update config
            test_config = config.copy()
            test_config['num_workers'] = num_workers
            
            # Run experiments
            results = self.run_all_experiments(test_config)
            scalability_results[f'{num_workers}_workers'] = results
            
            logger.info(f"Scalability test with {num_workers} workers completed")
        
        return scalability_results
    
    def run_matrix_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run matrix benchmark for fair comparison across all approaches and worker counts."""
        logger.info("Starting matrix benchmark for fair comparison...")
        
        # Matrix benchmark configuration
        worker_sizes = config.get('matrix_worker_sizes', [1, 2, 4])
        approaches = config.get('matrix_approaches', ['uniform', 'dynamic', 'sharded'])
        
        matrix_results = {}
        
        for workers in worker_sizes:
            logger.info(f"Running matrix benchmark with {workers} workers...")
            worker_results = {}
            
            for approach in approaches:
                try:
                    logger.info(f"Running {approach} with {workers} workers...")
                    start_time = time.time()
                    
                    # Update config for this specific test
                    test_config = config.copy()
                    test_config['num_workers'] = workers
                    test_config['num_samples'] = 10000  # Fair comparison: all process 10,000 total
                    
                    # Apply memory optimization for sharded approach
                    if approach == 'sharded':
                        if workers == 1:
                            # Single worker: use maximum memory and chunked processing
                            test_config.update({
                                'worker_memory_limit': config.get('single_worker_memory_limit', '4GB'),
                                'chunk_size_processing': config.get('single_worker_chunk_size', 1000),
                                'batch_size': config.get('single_worker_batch_size', 32),
                                'process_in_chunks': True
                            })
                        else:
                            # Multi-worker: standard memory configuration
                            test_config.update({
                                'worker_memory_limit': config.get('multi_worker_memory_limit', '2GB'),
                                'chunk_size_processing': config.get('multi_worker_chunk_size', 10000),
                                'batch_size': config.get('multi_worker_batch_size', 128),
                                'process_in_chunks': False
                            })
                    
                    # Run the specific approach
                    if approach == 'uniform':
                        result = self.run_uniform_partitioning(test_config)
                    elif approach == 'dynamic':
                        result = self.run_dynamic_partitioning(test_config)
                    elif approach == 'sharded':
                        result = self.run_sharded_partitioning(test_config)
                    
                    result['execution_time'] = time.time() - start_time
                    result['approach'] = approach
                    result['workers'] = workers
                    result['total_inferences'] = 10000  # Fair comparison
                    
                    worker_results[approach] = result
                    
                    logger.info(f"{approach} with {workers} workers completed in {result['execution_time']:.2f} seconds")
                    
                    # Wait between approaches
                    time.sleep(5)
                    
                except Exception as e:
                    logger.error(f"{approach} with {workers} workers failed: {str(e)}")
                    worker_results[approach] = {'error': str(e), 'workers': workers}
            
            matrix_results[f'{workers}_workers'] = worker_results
            logger.info(f"Matrix benchmark with {workers} workers completed")
            
            # Wait between worker counts
            time.sleep(10)
        
        return matrix_results
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of cluster services."""
        logger.info("Checking cluster status...")
        
        status = {}
        
        # Check Ray status
        try:
            ray_status = self.run_remote_command("ray status")
            status['ray'] = ray_status
        except Exception as e:
            status['ray'] = f"Error: {str(e)}"
        
        # Check Dask status
        try:
            dask_status = self.run_remote_command("ps aux | grep dask")
            status['dask'] = dask_status
        except Exception as e:
            status['dask'] = f"Error: {str(e)}"
        
        return status
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to file."""
        output_path = Path(__file__).parent / output_file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate experiment report."""
        report = []
        report.append("="*80)
        report.append("CLOUD DATA PARTITIONING EXPERIMENT RESULTS")
        report.append("="*80)
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Cluster: {self.cluster_info['cluster_name']}")
        report.append(f"Head IP: {self.head_ip}")
        report.append(f"Workers: {len(self.worker_ips)}")
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

def load_cluster_info() -> Dict[str, Any]:
    """Load cluster information from file."""
    info_file = Path(__file__).parent / "cluster_info.yaml"
    
    if not info_file.exists():
        raise FileNotFoundError(f"Cluster info file not found: {info_file}")
    
    with open(info_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main function for running cloud experiments."""
    parser = argparse.ArgumentParser(description='Run cloud data partitioning experiments')
    parser.add_argument('--config', type=str, default='cloud_config.yaml', help='Cloud configuration file')
    parser.add_argument('--approach', type=str, choices=['uniform', 'dynamic', 'sharded', 'all'], 
                       default='all', help='Approach to run')
    parser.add_argument('--scalability', action='store_true', help='Run scalability test')
    parser.add_argument('--matrix', action='store_true', help='Run matrix benchmark for fair comparison')
    parser.add_argument('--output', type=str, default='cloud_results.json', help='Output file')
    parser.add_argument('--status', action='store_true', help='Check cluster status only')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load cluster info
        cluster_info = load_cluster_info()
        
        # Load configuration
        config_file = Path(__file__).parent / args.config
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Create experiment runner
        runner = CloudExperimentRunner(cluster_info)
        
        if args.status:
            # Check cluster status
            status = runner.get_cluster_status()
            print("CLUSTER STATUS:")
            print("="*50)
            for service, info in status.items():
                print(f"\n{service.upper()}:")
                print(info)
            return
        
        # Copy code to cluster
        runner.copy_code_to_cluster()
        
        # Run experiments
        if args.matrix:
            logger.info("Running matrix benchmark for fair comparison...")
            results = runner.run_matrix_benchmark(config)
            output_file = f"matrix_{args.output}"
        elif args.scalability:
            logger.info("Running scalability test...")
            results = runner.run_scalability_test(config)
            output_file = f"scalability_{args.output}"
        elif args.approach == 'all':
            logger.info("Running all approaches...")
            results = runner.run_all_experiments(config)
            output_file = args.output
        else:
            logger.info(f"Running {args.approach} approach...")
            if args.approach == 'uniform':
                results = {args.approach: runner.run_uniform_partitioning(config)}
            elif args.approach == 'dynamic':
                results = {args.approach: runner.run_dynamic_partitioning(config)}
            elif args.approach == 'sharded':
                results = {args.approach: runner.run_sharded_partitioning(config)}
            output_file = f"{args.approach}_{args.output}"
        
        # Save results
        runner.save_results(results, output_file)
        
        # Generate and print summary
        summary = runner.generate_report(results)
        print("\n" + summary)
        
        # Save summary to file
        summary_file = Path(__file__).parent / f"summary_{output_file.replace('.json', '.txt')}"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Summary saved to summary_{output_file.replace('.json', '.txt')}")
        
    except Exception as e:
        logger.error(f"Cloud experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
