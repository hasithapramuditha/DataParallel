"""
Sharded Data Partitioning Implementation using Dask.

This approach shards the dataset into fixed parts and distributes them to nodes
using Dask's distributed computing capabilities.
"""
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster, as_completed
import torch
import numpy as np
import time
import psutil
import os
import sys
import argparse
import logging
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model, get_raw_data, create_shards, MetricsCollector

logger = logging.getLogger(__name__)

class ShardedInferenceWorker:
    """Worker class for sharded inference."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.model = get_model()
        self.inference_count = 0
        
    def infer_shard(self, shard_data: np.ndarray, process_in_chunks: bool = False, chunk_size: int = 1000) -> Dict[str, Any]:
        """
        Perform inference on a data shard.
        
        Args:
            shard_data: Numpy array containing the data shard
            process_in_chunks: Whether to process data in smaller chunks
            chunk_size: Size of each chunk for processing
            
        Returns:
            Dictionary containing inference results and metrics
        """
        start_time = time.time()
        total_inferences = 0
        
        if process_in_chunks and len(shard_data) > chunk_size:
            # Process data in smaller chunks to avoid memory issues
            for i in range(0, len(shard_data), chunk_size):
                chunk_data = shard_data[i:i + chunk_size]
                
                # Convert to tensor
                data_tensor = torch.tensor(chunk_data)
                
                # Perform inference
                with torch.no_grad():
                    outputs = self.model(data_tensor)
                
                total_inferences += len(chunk_data)
                
                # Clear memory after each chunk
                del data_tensor, outputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            # Process all data at once
            data_tensor = torch.tensor(shard_data)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(data_tensor)
            
            total_inferences = len(shard_data)
            
            # Clear memory
            del data_tensor, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        inference_time = time.time() - start_time
        self.inference_count += total_inferences
        
        return {
            'worker_id': self.worker_id,
            'shard_size': total_inferences,
            'inference_time': inference_time,
            'total_inferences': self.inference_count,
            # sample with short interval to get a non-zero measurement
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }

class ShardedPartitioningManager:
    """Manages sharded partitioning using Dask."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.cluster = None
        self.workers = []
        self.metrics = MetricsCollector()
        
    def initialize_dask(self):
        """Initialize Dask cluster."""
        if self.config.get('dask_scheduler_address'):
            # Connect to existing cluster
            self.client = Client(self.config['dask_scheduler_address'])
        else:
            # Create local cluster with configurable memory settings
            memory_limit = self.config.get('worker_memory_limit', '1GB')
            memory_target_fraction = self.config.get('worker_memory_target_fraction', 0.7)
            memory_spill_fraction = self.config.get('worker_memory_spill_fraction', 0.8)
            memory_pause_fraction = self.config.get('worker_memory_pause_fraction', 0.9)
            
            self.cluster = LocalCluster(
                n_workers=self.config['num_workers'],
                threads_per_worker=1,
                memory_limit=memory_limit,
                memory_target_fraction=memory_target_fraction,
                memory_spill_fraction=memory_spill_fraction,
                memory_pause_fraction=memory_pause_fraction
            )
            self.client = Client(self.cluster)
        
        logger.info(f"Dask cluster initialized with {len(self.client.scheduler_info()['workers'])} workers")
        logger.info(f"Dashboard available at: {self.client.dashboard_link}")
    
    def create_workers(self):
        """Create worker instances."""
        self.workers = [
            ShardedInferenceWorker(i) 
            for i in range(self.config['num_workers'])
        ]
        logger.info(f"Created {len(self.workers)} inference workers")
    
    def load_and_shard_data(self) -> List[np.ndarray]:
        """Load data and create shards."""
        logger.info(f"Loading {self.config['num_samples']} samples...")
        raw_data = get_raw_data(self.config['num_samples'])
        
        # Convert to numpy array
        data_array = np.array(raw_data)
        logger.info(f"Loaded data shape: {data_array.shape}")
        
        # Create shards
        shards = create_shards(data_array, self.config['num_shards'])
        logger.info(f"Created {len(shards)} shards")
        
        return shards
    
    def run_sharded_inference(self, shards: List[np.ndarray]) -> Dict[str, float]:
        """
        Run sharded inference.
        
        Args:
            shards: List of data shards
            
        Returns:
            Dictionary containing performance metrics
        """
        # Start timing
        self.metrics.start_timing()
        
        # Submit inference tasks for each shard
        futures = []
        process_in_chunks = self.config.get('process_in_chunks', False)
        chunk_size = self.config.get('chunk_size', 1000)
        
        for i, shard in enumerate(shards):
            worker = self.workers[i % len(self.workers)]
            future = self.client.submit(worker.infer_shard, shard, process_in_chunks, chunk_size)
            futures.append(future)
        
        logger.info(f"Submitted {len(futures)} shard inference tasks")
        
        # Collect results
        completed_tasks = 0
        total_inferences = 0
        worker_stats = []
        
        for future in as_completed(futures):
            try:
                result = future.result()
                total_inferences += result['shard_size']
                worker_stats.append(result)
                completed_tasks += 1
                
                if completed_tasks % 5 == 0:
                    logger.info(f"Completed {completed_tasks}/{len(shards)} shards")
                    
            except Exception as e:
                logger.error(f"Shard inference failed: {str(e)}")
        
        # Stop timing
        self.metrics.stop_timing()
        
        # Record final metrics
        self.metrics.add_inferences(total_inferences)
        
        # Calculate aggregated metrics
        if worker_stats:
            avg_cpu = np.mean([stat['cpu_usage'] for stat in worker_stats])
            avg_memory = np.mean([stat['memory_usage'] for stat in worker_stats])
            total_inference_time = sum([stat['inference_time'] for stat in worker_stats])
        else:
            avg_cpu = 0
            avg_memory = 0
            total_inference_time = 0
        
        metrics = self.metrics.get_metrics()
        metrics.update({
            'avg_cpu_percent': avg_cpu,
            'avg_memory_mb': avg_memory,
            'total_inference_time': total_inference_time,
            'completed_tasks': completed_tasks,
            'num_workers': len(self.workers),
            'num_shards': len(shards)
        })
        
        # include nodewise metrics per worker
        nodewise = []
        for stat in worker_stats:
            nodewise.append({
                'worker_id': stat['worker_id'],
                'throughput': stat['shard_size'] / stat['inference_time'] if stat['inference_time'] > 0 else 0,
                'latency_ms': stat['inference_time'] * 1000,
                'avg_cpu_percent': stat['cpu_usage'],
                'avg_memory_mb': stat['memory_usage'],
                'total_duration': stat['inference_time'],
                'total_inferences': stat['shard_size']
            })
        metrics['nodewise'] = nodewise
        return metrics
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get Dask cluster information."""
        return {
            'num_workers': len(self.client.scheduler_info()['workers']),
            'dashboard_link': self.client.dashboard_link,
            'cluster_info': self.client.scheduler_info()
        }
    
    def cleanup(self):
        """Clean up Dask resources."""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()

def run_sharded_partitioning(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Run sharded partitioning benchmark.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing performance metrics
    """
    manager = ShardedPartitioningManager(config)
    
    try:
        # Initialize Dask
        manager.initialize_dask()
        
        # Create workers
        manager.create_workers()
        
        # Load and shard data
        shards = manager.load_and_shard_data()
        
        # Run inference
        logger.info("Starting sharded partitioning inference...")
        results = manager.run_sharded_inference(shards)
        
        # Get cluster info
        cluster_info = manager.get_cluster_info()
        results['cluster_info'] = cluster_info
        
        logger.info(f"Sharded partitioning completed. Total inferences: {results['total_inferences']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Sharded partitioning failed: {str(e)}")
        raise
    finally:
        manager.cleanup()

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Sharded Data Partitioning Benchmark')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of Dask workers')
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples to process')
    parser.add_argument('--num-shards', type=int, default=4, help='Number of data shards')
    parser.add_argument('--dask-scheduler-address', type=str, default=None, help='Dask scheduler address (optional)')
    parser.add_argument('--output', type=str, default='sharded_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    config = {
        'num_workers': args.num_workers,
        'num_samples': args.num_samples,
        'num_shards': args.num_shards,
        'dask_scheduler_address': args.dask_scheduler_address
    }
    
    try:
        results = run_sharded_partitioning(config)
        
        # Save results
        from utils import save_results
        save_results(results, args.output)
        
        print(f"Sharded partitioning benchmark completed. Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
