"""
Sharded Data Partitioning Implementation - Fixed Version.

This approach partitions data into shards and processes them using a simpler
multiprocessing approach to avoid Dask serialization issues with PyTorch models.
"""
import torch
import numpy as np
import time
import psutil
import os
import sys
import argparse
import logging
import multiprocessing as mp
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model, get_raw_data, create_shards, MetricsCollector
from config import PartitioningConfig
from error_handling import retry_on_failure, RetryConfig, safe_execute

logger = logging.getLogger(__name__)

def process_shard_worker(shard_data: np.ndarray, worker_id: int, process_in_chunks: bool = True, chunk_size: int = 128) -> Dict[str, Any]:
    """
    Worker function for processing a single shard.
    This runs in a separate process to avoid serialization issues.
    """
    try:
        # Load model in the worker process
        model = get_model()
        
        start_time = time.time()
        total_inferences = 0
        cpu_usages = []
        memory_usages = []
        
        if process_in_chunks and len(shard_data) > chunk_size:
            # Process data in smaller chunks to avoid memory issues
            for i in range(0, len(shard_data), chunk_size):
                chunk_data = shard_data[i:i + chunk_size]
                
                # Convert to tensor with proper memory management
                data_tensor = torch.tensor(chunk_data, dtype=torch.float32)
                
                # Perform inference
                with torch.no_grad():
                    outputs = model(data_tensor)
                
                total_inferences += len(chunk_data)
                
                # Record metrics
                cpu_usages.append(psutil.cpu_percent(interval=0.01))
                memory_usages.append(psutil.Process().memory_info().rss / 1024 / 1024)
                
                # Clear memory after each chunk
                del data_tensor, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection periodically
                if i % (chunk_size * 10) == 0:
                    import gc
                    gc.collect()
        else:
            # Process all data at once (for small shards)
            data_tensor = torch.tensor(shard_data, dtype=torch.float32)
            
            # Perform inference
            with torch.no_grad():
                outputs = model(data_tensor)
            
            total_inferences = len(shard_data)
            
            # Record metrics
            cpu_usages.append(psutil.cpu_percent(interval=0.1))
            memory_usages.append(psutil.Process().memory_info().rss / 1024 / 1024)
            
            # Clear memory
            del data_tensor, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        inference_time = time.time() - start_time
        
        return {
            'worker_id': worker_id,
            'total_inferences': total_inferences,
            'inference_time': inference_time,
            'throughput': total_inferences / inference_time if inference_time > 0 else 0,
            'avg_cpu_percent': np.mean(cpu_usages) if cpu_usages else 0,
            'avg_memory_mb': np.mean(memory_usages) if memory_usages else 0,
            'max_memory_mb': max(memory_usages) if memory_usages else 0,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {e}")
        return {
            'worker_id': worker_id,
            'total_inferences': 0,
            'inference_time': 0,
            'throughput': 0,
            'avg_cpu_percent': 0,
            'avg_memory_mb': 0,
            'max_memory_mb': 0,
            'status': 'failed',
            'error': str(e)
        }

class ShardedPartitioningManager:
    """
    Manager class for sharded partitioning using multiprocessing.
    """
    
    def __init__(self, num_workers: int = 2, num_shards: int = None):
        """
        Initialize sharded partitioning manager.
        
        Args:
            num_workers: Number of worker processes
            num_shards: Number of data shards (defaults to num_workers)
        """
        self.num_workers = num_workers
        self.num_shards = num_shards or num_workers
        self.metrics = MetricsCollector()
        
    def _optimize_shard_distribution(self, shards: List[np.ndarray]) -> List[Tuple[np.ndarray, int]]:
        """
        Optimize shard distribution to workers based on shard sizes.
        Uses a greedy algorithm to assign largest shards to least loaded workers.
        """
        if len(shards) <= self.num_workers:
            # If we have fewer shards than workers, assign one shard per worker
            assignments = []
            for i, shard in enumerate(shards):
                worker_id = i % self.num_workers
                assignments.append((shard, worker_id))
            return assignments
        
        # Sort shards by size (largest first)
        shard_size_pairs = [(i, len(shard)) for i, shard in enumerate(shards)]
        shard_size_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Track worker loads
        worker_loads = [0] * self.num_workers
        assignments = []
        
        # Assign shards to workers using greedy approach
        for shard_idx, shard_size in shard_size_pairs:
            # Find worker with minimum current load
            min_load_worker = min(range(self.num_workers), key=lambda w: worker_loads[w])
            assignments.append((shards[shard_idx], min_load_worker))
            worker_loads[min_load_worker] += shard_size
        
        return assignments
    
    def _calculate_distribution_efficiency(self, assignments: List[Tuple[np.ndarray, int]]) -> float:
        """
        Calculate how evenly shards are distributed across workers.
        Returns a value between 0 and 1, where 1 is perfectly balanced.
        """
        worker_loads = [0] * self.num_workers
        for shard, worker_id in assignments:
            worker_loads[worker_id] += len(shard)
        
        if not worker_loads or max(worker_loads) == 0:
            return 1.0
        
        # Calculate coefficient of variation (lower is better)
        mean_load = np.mean(worker_loads)
        std_load = np.std(worker_loads)
        cv = std_load / mean_load if mean_load > 0 else 0
        
        # Convert to efficiency (1 - cv, clamped to [0, 1])
        efficiency = max(0, 1 - cv)
        return efficiency
    
    @retry_on_failure(RetryConfig(max_retries=2, base_delay=2.0))
    def run_sharded_inference(self, data: np.ndarray, process_in_chunks: bool = True, chunk_size: int = 128) -> Dict[str, Any]:
        """
        Run sharded inference on the given data.
        
        Args:
            data: Input data array
            process_in_chunks: Whether to process data in chunks
            chunk_size: Size of chunks for processing
            
        Returns:
            Dictionary containing inference results
        """
        logger.info(f"Starting sharded partitioning inference with {self.num_workers} workers...")
        
        # Create shards
        shards = create_shards(data, self.num_shards)
        logger.info(f"Created {len(shards)} shards")
        
        # Optimize shard distribution
        shard_assignments = self._optimize_shard_distribution(shards)
        distribution_efficiency = self._calculate_distribution_efficiency(shard_assignments)
        
        logger.info(f"Optimized shard distribution: {[len(shard) for shard, _ in shard_assignments]}")
        
        # Start timing
        self.metrics.start_timing()
        
        # Process shards using multiprocessing
        results = []
        failed_tasks = []
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all shard processing tasks
                future_to_shard = {}
                for shard_idx, (shard, worker_id) in enumerate(shard_assignments):
                    future = executor.submit(process_shard_worker, shard, worker_id, process_in_chunks, chunk_size)
                    future_to_shard[future] = (shard_idx, worker_id)
                
                # Collect results as they complete
                for future in as_completed(future_to_shard):
                    shard_idx, worker_id = future_to_shard[future]
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        if result['status'] == 'success':
                            results.append(result)
                        else:
                            failed_tasks.append((shard_idx, worker_id, result.get('error', 'Unknown error')))
                    except Exception as e:
                        failed_tasks.append((shard_idx, worker_id, str(e)))
                        logger.error(f"Shard {shard_idx} processing failed: {e}")
        
        except Exception as e:
            logger.error(f"Sharded inference failed: {e}")
            raise
        
        finally:
            self.metrics.stop_timing()
        
        # Aggregate results
        if not results:
            logger.warning("No shards completed successfully")
            return {
                'total_inferences': 0,
                'total_duration': 0,
                'throughput': 0,
                'completed_tasks': 0,
                'failed_tasks': len(failed_tasks),
                'shard_distribution_efficiency': distribution_efficiency,
                'worker_results': [],
                'status': 'failed'
            }
        
        # Calculate aggregate metrics
        total_inferences = sum(r['total_inferences'] for r in results)
        total_duration = self.metrics.get_metrics().get('total_duration', 0)
        throughput = total_inferences / total_duration if total_duration > 0 else 0
        
        # Calculate aggregate resource usage
        avg_cpu = np.mean([r['avg_cpu_percent'] for r in results])
        avg_memory = np.mean([r['avg_memory_mb'] for r in results])
        max_memory = max([r['max_memory_mb'] for r in results])
        
        logger.info(f"Sharded partitioning completed. Total inferences: {total_inferences}")
        if failed_tasks:
            logger.warning(f"Failed tasks: {failed_tasks}")
        
        return {
            'total_inferences': total_inferences,
            'total_duration': total_duration,
            'throughput': throughput,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'completed_tasks': len(results),
            'failed_tasks': len(failed_tasks),
            'shard_distribution_efficiency': distribution_efficiency,
            'worker_results': results,
            'status': 'success'
        }

@retry_on_failure(RetryConfig(max_retries=2, base_delay=2.0))
def run_sharded_partitioning(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run sharded partitioning experiment with given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("Starting sharded partitioning experiment")
    
    # Create manager
    manager = ShardedPartitioningManager(
        num_workers=config.get('num_workers', 2),
        num_shards=config.get('num_shards', config.get('num_workers', 2))
    )
    
    # Load data
    logger.info(f"Loading {config.get('num_samples', 1000)} samples...")
    data_list = get_raw_data(config.get('num_samples', 1000))
    # Convert list of arrays to single numpy array
    data = np.array(data_list)
    logger.info(f"Loaded data shape: {data.shape}")
    
    # Run inference
    results = manager.run_sharded_inference(
        data,
        process_in_chunks=config.get('process_in_chunks', True),
        chunk_size=config.get('chunk_size', 128)
    )
    
    return results

def main():
    """Main function for running sharded partitioning experiments."""
    parser = argparse.ArgumentParser(description='Sharded Data Partitioning')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--shards', type=int, default=None, help='Number of shards')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--chunk-size', type=int, default=128, help='Chunk size for processing')
    parser.add_argument('--no-chunks', action='store_true', help='Disable chunked processing')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run experiment
    config = {
        'num_workers': args.workers,
        'num_shards': args.shards or args.workers,
        'num_samples': args.samples,
        'process_in_chunks': not args.no_chunks,
        'chunk_size': args.chunk_size
    }
    
    print(f"🚀 Starting Sharded Partitioning Experiment")
    print(f"   Workers: {args.workers}")
    print(f"   Shards: {config['num_shards']}")
    print(f"   Samples: {args.samples}")
    print(f"   Chunk Size: {args.chunk_size}")
    print(f"   Chunked Processing: {config['process_in_chunks']}")
    
    results = run_sharded_partitioning(config)
    
    # Print results
    print("\n" + "="*60)
    print("SHARDED PARTITIONING RESULTS")
    print("="*60)
    print(f"Total Inferences: {results['total_inferences']}")
    print(f"Total Duration: {results['total_duration']:.2f}s")
    print(f"Throughput: {results['throughput']:.2f} inf/s")
    print(f"Completed Tasks: {results['completed_tasks']}")
    print(f"Failed Tasks: {results['failed_tasks']}")
    print(f"Distribution Efficiency: {results['shard_distribution_efficiency']:.3f}")
    print(f"Average CPU Usage: {results['avg_cpu_percent']:.1f}%")
    print(f"Average Memory Usage: {results['avg_memory_mb']:.1f} MB")
    print(f"Maximum Memory Usage: {results['max_memory_mb']:.1f} MB")
    
    if results['worker_results']:
        print(f"\nWorker Details:")
        for worker_result in results['worker_results']:
            print(f"  Worker {worker_result['worker_id']}: "
                  f"{worker_result['total_inferences']} inferences, "
                  f"{worker_result['throughput']:.2f} inf/s")

if __name__ == "__main__":
    main()
