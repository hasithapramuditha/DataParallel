"""
Uniform/Static Data Partitioning Implementation using PyTorch DistributedDataParallel.

This approach evenly splits data batches across nodes using PyTorch's DDP.
The model is replicated on each node and data is statically partitioned.
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import argparse
import logging
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_dataset, get_model, MetricsCollector, split_data_uniform

logger = logging.getLogger(__name__)

def setup_distributed(rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "12355"):
    """
    Initialize the distributed process group.
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        master_addr: Address of the master node
        master_port: Port for communication
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize the process group
    dist.init_process_group(
        backend='gloo',  # Use gloo for CPU
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )

def cleanup_distributed():
    """Clean up the distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def inference_worker(rank: int, world_size: int, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Worker function for distributed inference.
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        config: Configuration dictionary
        
    Returns:
        Dictionary containing performance metrics
    """
    try:
        # Setup distributed only if multi-process
        if world_size > 1:
            setup_distributed(rank, world_size, config['master_addr'], config['master_port'])
        
        # Initialize metrics collector
        metrics = MetricsCollector()
        
        # Load model and wrap with DDP only if multi-process
        model = get_model()
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model)
        
        # Load dataset
        dataloader = get_dataset(
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        logger.info(f"Worker {rank}: Starting inference with {len(dataloader)} batches")
        
        # Start timing
        metrics.start_timing()
        
        total_inferences = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            # Record metrics periodically
            if batch_idx % 10 == 0:
                metrics.record_cpu_usage()
                metrics.record_memory_usage()
            
            # For uniform partitioning, we'll use a simpler approach
            # Each worker processes a portion of the batch based on its rank
            batch_size = data.size(0)
            chunk_size = batch_size // world_size
            start_idx = rank * chunk_size
            end_idx = start_idx + chunk_size if rank < world_size - 1 else batch_size
            
            # Get this worker's chunk
            chunk = data[start_idx:end_idx]
            
            # Perform inference
            with torch.no_grad():
                _ = model(chunk)
            
            # Update metrics
            batch_inferences = len(chunk)
            metrics.add_inferences(batch_inferences)
            total_inferences += batch_inferences
            
            if rank == 0 and batch_idx % 50 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Stop timing
        metrics.stop_timing()
        
        # Get final metrics
        worker_metrics = metrics.get_metrics()
        # attach rank id for nodewise reporting
        worker_metrics['rank'] = rank
        
        # For single process mode, just return the metrics
        if world_size == 1:
            logger.info(f"Uniform partitioning completed. Total inferences: {worker_metrics['total_inferences']}")
            return worker_metrics
        
        # For multi-process mode, push worker metrics to shared list if provided
        results_list = config.get('results_list')
        if results_list is not None:
            # multiprocessing Manager list proxy
            results_list.append(worker_metrics)
        return worker_metrics
            
    except Exception as e:
        logger.error(f"Worker {rank} failed: {str(e)}")
        raise
    finally:
        if world_size > 1:
            cleanup_distributed()

def run_uniform_partitioning(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run uniform partitioning benchmark.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing performance metrics
    """
    world_size = config['world_size']
    
    logger.info(f"Starting uniform partitioning with {world_size} workers")
    logger.info(f"Configuration: {config}")
    
    # Use multiprocessing to spawn workers
    if world_size == 1:
        # Single process mode for local testing
        single = inference_worker(0, 1, config)
        return {
            'aggregate': single,
            'nodewise': [single]
        }
    else:
        # Multi-process mode with nodewise metrics aggregation via Manager list
        with mp.Manager() as manager:
            results_list = manager.list()
            config_with_list = dict(config)
            config_with_list['results_list'] = results_list
            mp.spawn(
                inference_worker,
                args=(world_size, config_with_list),
                nprocs=world_size,
                join=True
            )
            nodewise = list(results_list)
            # Aggregate
            if nodewise:
                total_inferences = sum(m['total_inferences'] for m in nodewise)
                total_duration = max(m['total_duration'] for m in nodewise)
                avg_cpu = sum(m['avg_cpu_percent'] for m in nodewise) / len(nodewise)
                avg_memory = sum(m['avg_memory_mb'] for m in nodewise) / len(nodewise)
                aggregate = {
                    'throughput': (total_inferences / total_duration) if total_duration > 0 else 0,
                    'latency_ms': (total_duration * 1000) / (total_inferences / 128) if total_inferences > 0 else 0,
                    'avg_cpu_percent': avg_cpu,
                    'avg_memory_mb': avg_memory,
                    'total_duration': total_duration,
                    'total_inferences': total_inferences
                }
            else:
                aggregate = {
                    'throughput': 0.0,
                    'latency_ms': 0.0,
                    'avg_cpu_percent': 0.0,
                    'avg_memory_mb': 0.0,
                    'total_duration': 0.0,
                    'total_inferences': 0
                }
            return {
                'aggregate': aggregate,
                'nodewise': nodewise
            }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Uniform Data Partitioning Benchmark')
    parser.add_argument('--world-size', type=int, default=4, help='Number of workers')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of data loader workers')
    parser.add_argument('--master-addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master-port', type=str, default='12355', help='Master node port')
    parser.add_argument('--output', type=str, default='uniform_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    config = {
        'world_size': args.world_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'master_addr': args.master_addr,
        'master_port': args.master_port
    }
    
    try:
        results = run_uniform_partitioning(config)
        
        # Save results
        from utils import save_results
        save_results(results, args.output)
        
        print(f"Uniform partitioning benchmark completed. Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
