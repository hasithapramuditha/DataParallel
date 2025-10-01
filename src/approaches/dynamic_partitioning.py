"""
Dynamic/Load-Balanced Data Partitioning Implementation using Ray.

This approach uses Ray to dynamically assign data chunks based on node load.
Ray handles load balancing automatically across the cluster.
"""
import ray
import torch
import numpy as np
import time
import psutil
import os
import sys
import argparse
import logging
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with error handling for Ray workers
try:
    from utils import get_model, get_raw_data, split_data_dynamic, MetricsCollector
except ImportError:
    # For Ray workers, we'll define the functions locally
    import torchvision
    import torchvision.transforms as transforms
    
    def get_model():
        """Load pre-trained ResNet-18 model for inference."""
        model = torchvision.models.resnet18(pretrained=True)
        model.eval()
        return model.to('cpu')
    
    def get_raw_data(num_samples=10000):
        """Get raw CIFAR-10 data as numpy arrays."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        data = []
        for i in range(min(num_samples, len(testset))):
            img, _ = testset[i]
            data.append(img.numpy())
        
        return data
    
    def split_data_dynamic(data, num_workers):
        """Split data for dynamic partitioning."""
        chunk_size = len(data) // num_workers
        chunks = []
        
        for i in range(num_workers):
            start_idx = i * chunk_size
            if i == num_workers - 1:
                end_idx = len(data)
            else:
                end_idx = (i + 1) * chunk_size
            chunks.append(data[start_idx:end_idx])
        
        return chunks
    
    class MetricsCollector:
        """Simple metrics collector for Ray workers."""
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.cpu_usages = []
            self.memory_usages = []
            self.total_inferences = 0
            
        def start_timing(self):
            self.start_time = time.time()
            
        def stop_timing(self):
            self.end_time = time.time()
            
        def record_cpu_usage(self):
            self.cpu_usages.append(psutil.cpu_percent())
            
        def record_memory_usage(self):
            process = psutil.Process(os.getpid())
            self.memory_usages.append(process.memory_info().rss / 1024 / 1024)
            
        def add_inferences(self, count):
            self.total_inferences += count
            
        def get_metrics(self):
            if self.start_time is None or self.end_time is None:
                raise ValueError("Timing not started or stopped")
                
            duration = self.end_time - self.start_time
            throughput = self.total_inferences / duration if duration > 0 else 0
            avg_cpu = np.mean(self.cpu_usages) if self.cpu_usages else 0
            avg_memory = np.mean(self.memory_usages) if self.memory_usages else 0
            
            return {
                'throughput': throughput,
                'latency_ms': (duration * 1000) / (self.total_inferences / 128) if self.total_inferences > 0 else 0,
                'avg_cpu_percent': avg_cpu,
                'avg_memory_mb': avg_memory,
                'total_duration': duration,
                'total_inferences': self.total_inferences
            }

logger = logging.getLogger(__name__)

@ray.remote
class InferenceActor:
    """Ray actor for distributed inference."""
    
    def __init__(self, actor_id: int):
        self.actor_id = actor_id
        # Load model directly in the actor to avoid import issues
        import torch
        import torchvision
        
        # Suppress warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()
        self.model = self.model.to('cpu')
        self.inference_count = 0
        self.start_time = None
        
    def infer_batch(self, data_batch: List[np.ndarray]) -> Dict[str, Any]:
        """
        Perform inference on a batch of data.
        
        Args:
            data_batch: List of numpy arrays representing images
            
        Returns:
            Dictionary containing inference results and metrics
        """
        import torch
        import numpy as np
        import time
        import psutil
        import os
        
        if self.start_time is None:
            self.start_time = time.time()
            
        # Convert to tensor
        data_tensor = torch.tensor(np.array(data_batch))
        
        # Perform inference
        with torch.no_grad():
            start_inference = time.time()
            outputs = self.model(data_tensor)
            inference_time = time.time() - start_inference
        
        self.inference_count += len(data_batch)
        
        return {
            'actor_id': self.actor_id,
            'batch_size': len(data_batch),
            'inference_time': inference_time,
            'total_inferences': self.inference_count,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get actor statistics."""
        return {
            'actor_id': self.actor_id,
            'total_inferences': self.inference_count,
            'uptime': time.time() - self.start_time if self.start_time else 0
        }

class DynamicPartitioningManager:
    """Manages dynamic partitioning using Ray."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.actors = []
        self.metrics = MetricsCollector()
        
    def initialize_ray(self):
        """Initialize Ray cluster."""
        if self.config.get('ray_address'):
            # Connect to existing cluster
            ray.init(address=self.config['ray_address'])
        else:
            # Start local cluster
            ray.init(
                num_cpus=self.config['num_workers'],
                ignore_reinit_error=True
            )
        
        logger.info(f"Ray initialized with {ray.cluster_resources()}")
    
    def create_actors(self):
        """Create Ray actors for inference."""
        self.actors = [
            InferenceActor.remote(i) 
            for i in range(self.config['num_workers'])
        ]
        logger.info(f"Created {len(self.actors)} inference actors")
    
    def load_data(self) -> List[np.ndarray]:
        """Load and prepare data for dynamic partitioning."""
        logger.info(f"Loading {self.config['num_samples']} samples...")
        data = get_raw_data(self.config['num_samples'])
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def create_initial_chunks(self, data: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Create initial data chunks for distribution."""
        chunk_size = self.config['chunk_size']
        chunks = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks of size {chunk_size}")
        return chunks
    
    def run_dynamic_inference(self, data: List[np.ndarray]) -> Dict[str, float]:
        """
        Run dynamic inference with load balancing.
        
        Args:
            data: List of data samples
            
        Returns:
            Dictionary containing performance metrics
        """
        # Create initial chunks
        chunks = self.create_initial_chunks(data)
        
        # Start timing
        self.metrics.start_timing()
        
        # Submit all chunks to actors (Ray handles load balancing)
        futures = []
        for chunk in chunks:
            # Round-robin assignment with Ray's load balancing
            actor = self.actors[len(futures) % len(self.actors)]
            future = actor.infer_batch.remote(chunk)
            futures.append(future)
        
        logger.info(f"Submitted {len(futures)} inference tasks")
        
        # Collect results as they complete
        completed_tasks = 0
        total_inferences = 0
        actor_stats = []
        
        # Use Ray's wait to get results as they complete
        while futures:
            ready, futures = ray.wait(futures, timeout=1.0)
            
            for future in ready:
                try:
                    result = ray.get(future)
                    total_inferences += result['batch_size']
                    actor_stats.append(result)
                    completed_tasks += 1
                    
                    if completed_tasks % 10 == 0:
                        logger.info(f"Completed {completed_tasks}/{len(chunks)} tasks")
                        
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")
        
        # Stop timing
        self.metrics.stop_timing()
        
        # Record final metrics
        self.metrics.add_inferences(total_inferences)
        
        # Calculate aggregated metrics
        if actor_stats:
            avg_cpu = np.mean([stat['cpu_usage'] for stat in actor_stats])
            avg_memory = np.mean([stat['memory_usage'] for stat in actor_stats])
            total_inference_time = sum([stat['inference_time'] for stat in actor_stats])
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
            'num_actors': len(self.actors)
        })
        
        # include nodewise metrics per actor
        nodewise = []
        for stat in actor_stats:
            nodewise.append({
                'actor_id': stat['actor_id'],
                'throughput': stat['batch_size'] / stat['inference_time'] if stat['inference_time'] > 0 else 0,
                'latency_ms': stat['inference_time'] * 1000,
                'avg_cpu_percent': stat['cpu_usage'],
                'avg_memory_mb': stat['memory_usage'],
                'total_duration': stat['inference_time'],
                'total_inferences': stat['batch_size']
            })
        metrics['nodewise'] = nodewise
        return metrics
    
    def get_actor_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics from all actors."""
        stats_futures = [actor.get_stats.remote() for actor in self.actors]
        return ray.get(stats_futures)
    
    def cleanup(self):
        """Clean up Ray resources."""
        ray.shutdown()

def run_dynamic_partitioning(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Run dynamic partitioning benchmark.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing performance metrics
    """
    manager = DynamicPartitioningManager(config)
    
    try:
        # Initialize Ray
        manager.initialize_ray()
        
        # Create actors
        manager.create_actors()
        
        # Load data
        data = manager.load_data()
        
        # Run inference
        logger.info("Starting dynamic partitioning inference...")
        results = manager.run_dynamic_inference(data)
        
        # Get actor statistics (additional info)
        actor_stats = manager.get_actor_statistics()
        results['actor_statistics'] = actor_stats
        
        logger.info(f"Dynamic partitioning completed. Total inferences: {results['total_inferences']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Dynamic partitioning failed: {str(e)}")
        raise
    finally:
        manager.cleanup()

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Dynamic Data Partitioning Benchmark')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of Ray workers')
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples to process')
    parser.add_argument('--chunk-size', type=int, default=128, help='Chunk size for dynamic partitioning')
    parser.add_argument('--ray-address', type=str, default=None, help='Ray cluster address (optional)')
    parser.add_argument('--output', type=str, default='dynamic_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    config = {
        'num_workers': args.num_workers,
        'num_samples': args.num_samples,
        'chunk_size': args.chunk_size,
        'ray_address': args.ray_address
    }
    
    try:
        results = run_dynamic_partitioning(config)
        
        # Save results
        from utils import save_results
        save_results(results, args.output)
        
        print(f"Dynamic partitioning benchmark completed. Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
