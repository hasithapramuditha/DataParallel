"""
Shared utilities for data partitioning experiments on CIFAR-10 with real-time streaming.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import psutil
import os
import threading
import queue
from typing import Tuple, List, Dict, Any, Iterator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages performance metrics during inference."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.cpu_usages = []
        self.memory_usages = []
        self.total_inferences = 0
        
    def start_timing(self):
        """Start timing the experiment."""
        self.start_time = time.time()
        
    def stop_timing(self):
        """Stop timing the experiment."""
        self.end_time = time.time()
        
    def record_cpu_usage(self):
        """Record current CPU usage."""
        self.cpu_usages.append(psutil.cpu_percent())
        
    def record_memory_usage(self):
        """Record current memory usage."""
        process = psutil.Process(os.getpid())
        self.memory_usages.append(process.memory_info().rss / 1024 / 1024)  # MB
        
    def add_inferences(self, count: int):
        """Add to total inference count."""
        self.total_inferences += count
        
    def get_metrics(self) -> Dict[str, float]:
        """Get computed metrics."""
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

def get_dataset(batch_size: int = 128, num_workers: int = 1) -> torch.utils.data.DataLoader:
    """
    Load CIFAR-10 test dataset.
    
    Args:
        batch_size: Batch size for data loader
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader for CIFAR-10 test set
    """
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
    
    return torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

def get_dataset_for_distributed(batch_size: int = 128, num_workers: int = 1, 
                               rank: int = 0, world_size: int = 1) -> torch.utils.data.DataLoader:
    """
    Load CIFAR-10 test dataset with proper distributed sampling for DDP.
    
    Args:
        batch_size: Batch size for data loader
        num_workers: Number of worker processes for data loading
        rank: Rank of the current process
        world_size: Total number of processes
        
    Returns:
        DataLoader for CIFAR-10 test set with DistributedSampler
    """
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
    
    # Create DistributedSampler for proper data distribution
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        testset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    return torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

def get_model() -> torch.nn.Module:
    """
    Load pre-trained ResNet-18 model for inference.
    
    Returns:
        Pre-trained ResNet-18 model in evaluation mode
    """
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()  # Set to inference mode
    return model.to('cpu')  # CPU only

def get_raw_data(num_samples: int = 10000) -> List[np.ndarray]:
    """
    Get raw CIFAR-10 data as numpy arrays for dynamic partitioning.
    
    Args:
        num_samples: Number of samples to load
        
    Returns:
        List of numpy arrays representing images
    """
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

class RealTimeDataStream:
    """
    Real-time data stream generator for CIFAR-10 at 500 samples/s.
    Simulates continuous data flow for real-time inference scenarios.
    """
    
    def __init__(self, samples_per_second: int = 500, max_samples: int = 10000, num_workers: int = 1):
        """
        Initialize real-time data stream.
        
        Args:
            samples_per_second: Total rate of data generation (default: 500)
            max_samples: Maximum number of samples to generate
            num_workers: Number of workers to distribute load across
        """
        self.samples_per_second = samples_per_second
        self.max_samples = max_samples
        self.num_workers = num_workers
        # Each worker should get samples_per_second / num_workers rate
        self.sample_interval = 1.0 / samples_per_second
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load CIFAR-10 dataset
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=self.transform
        )
        
        self.current_index = 0
        self.generated_samples = 0
        self.start_time = None
        
    def start_stream(self):
        """Start the data stream."""
        self.start_time = time.time()
        logger.info(f"Starting real-time stream at {self.samples_per_second} samples/s")
        
    def get_next_sample(self) -> Tuple[np.ndarray, int]:
        """
        Get the next sample from the stream with proper timing.
        
        Returns:
            Tuple of (image_data, label)
        """
        if self.start_time is None:
            self.start_stream()
            
        # Calculate expected time for this sample
        expected_time = self.generated_samples * self.sample_interval
        current_time = time.time() - self.start_time
        
        # Sleep if we're ahead of schedule
        if current_time < expected_time:
            time.sleep(expected_time - current_time)
        
        # Get sample from dataset (cycling through if needed)
        img, label = self.dataset[self.current_index % len(self.dataset)]
        self.current_index += 1
        self.generated_samples += 1
        
        return img.numpy(), label
    
    def is_stream_complete(self) -> bool:
        """Check if stream has generated all required samples."""
        return self.generated_samples >= self.max_samples
    
    def get_stream_metrics(self) -> Dict[str, float]:
        """Get real-time stream metrics."""
        if self.start_time is None:
            return {}
            
        elapsed_time = time.time() - self.start_time
        actual_rate = self.generated_samples / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'target_rate': self.samples_per_second,
            'actual_rate': actual_rate,
            'elapsed_time': elapsed_time,
            'generated_samples': self.generated_samples,
            'rate_accuracy': (actual_rate / self.samples_per_second) * 100
        }

class StreamingDataPartitioner:
    """
    Handles real-time data partitioning for streaming scenarios.
    """
    
    def __init__(self, num_workers: int, partition_strategy: str = "uniform"):
        """
        Initialize streaming data partitioner.
        
        Args:
            num_workers: Number of worker nodes
            partition_strategy: Strategy for partitioning ("uniform", "dynamic", "sharded")
        """
        self.num_workers = num_workers
        self.partition_strategy = partition_strategy
        self.worker_queues = [queue.Queue() for _ in range(num_workers)]
        self.current_worker = 0
        
    def partition_sample(self, sample_data: np.ndarray, label: int) -> List[Tuple[np.ndarray, int]]:
        """
        Partition a single sample according to the strategy.
        
        Args:
            sample_data: Image data
            label: Sample label
            
        Returns:
            List of (data, label) tuples for each worker
        """
        if self.partition_strategy == "uniform":
            # Round-robin assignment
            worker_id = self.current_worker % self.num_workers
            self.current_worker += 1
            return [(sample_data, label) if i == worker_id else (None, None) for i in range(self.num_workers)]
            
        elif self.partition_strategy == "dynamic":
            # All workers get the sample for dynamic load balancing
            return [(sample_data, label) for _ in range(self.num_workers)]
            
        elif self.partition_strategy == "sharded":
            # Fixed sharding based on sample characteristics
            shard_id = hash(sample_data.tobytes()) % self.num_workers
            return [(sample_data, label) if i == shard_id else (None, None) for i in range(self.num_workers)]
        
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")

def create_realtime_stream(samples_per_second: int = 500, max_samples: int = 10000, num_workers: int = 1) -> RealTimeDataStream:
    """
    Create a real-time data stream for CIFAR-10.
    
    Args:
        samples_per_second: Total rate of data generation
        max_samples: Maximum samples to generate
        num_workers: Number of workers to distribute load across
        
    Returns:
        RealTimeDataStream instance
    """
    return RealTimeDataStream(samples_per_second, max_samples, num_workers)

def create_streaming_partitioner(num_workers: int, strategy: str = "uniform") -> StreamingDataPartitioner:
    """
    Create a streaming data partitioner.
    
    Args:
        num_workers: Number of workers
        strategy: Partitioning strategy
        
    Returns:
        StreamingDataPartitioner instance
    """
    return StreamingDataPartitioner(num_workers, strategy)

def split_data_uniform(data: torch.Tensor, world_size: int) -> List[torch.Tensor]:
    """
    Split data uniformly across workers.
    
    Args:
        data: Input tensor to split
        world_size: Number of workers
        
    Returns:
        List of tensors split across workers
    """
    return torch.chunk(data, world_size)

def split_data_dynamic(data: List[np.ndarray], num_workers: int, chunk_size: int = 32) -> List[List[np.ndarray]]:
    """
    Split data for dynamic partitioning with configurable chunk size.
    
    Args:
        data: List of numpy arrays
        num_workers: Number of workers
        chunk_size: Size of each chunk for dynamic load balancing
        
    Returns:
        List of data chunks for dynamic assignment
    """
    if not data:
        return []
    
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunks.append(chunk)
    
    logger.info(f"Split {len(data)} samples into {len(chunks)} chunks of size {chunk_size}")
    return chunks

def create_shards(data: np.ndarray, num_shards: int, min_shard_size: int = 1) -> List[np.ndarray]:
    """
    Create data shards for sharded partitioning with improved distribution.
    
    Args:
        data: Input data array
        num_shards: Number of shards to create
        min_shard_size: Minimum size for each shard
        
    Returns:
        List of data shards with balanced sizes
    """
    if len(data) == 0:
        return []
    
    if num_shards <= 0:
        raise ValueError("Number of shards must be positive")
    
    if num_shards > len(data):
        logger.warning(f"Number of shards ({num_shards}) exceeds data size ({len(data)}). Using {len(data)} shards.")
        num_shards = len(data)
    
    # Calculate base shard size and remainder
    base_shard_size = len(data) // num_shards
    remainder = len(data) % num_shards
    
    shards = []
    start_idx = 0
    
    for i in range(num_shards):
        # Add one extra sample to first 'remainder' shards for even distribution
        current_shard_size = base_shard_size + (1 if i < remainder else 0)
        
        # Ensure minimum shard size
        if current_shard_size < min_shard_size and i < num_shards - 1:
            current_shard_size = min_shard_size
        
        end_idx = start_idx + current_shard_size
        shard = data[start_idx:end_idx]
        shards.append(shard)
        start_idx = end_idx
    
    # Log shard distribution
    shard_sizes = [len(shard) for shard in shards]
    logger.info(f"Created {len(shards)} shards with sizes: {shard_sizes}")
    logger.info(f"Shard size statistics - Min: {min(shard_sizes)}, Max: {max(shard_sizes)}, Mean: {np.mean(shard_sizes):.1f}")
    
    return shards

def create_balanced_shards(data: np.ndarray, num_shards: int) -> List[np.ndarray]:
    """
    Create optimally balanced shards using a more sophisticated algorithm.
    
    Args:
        data: Input data array
        num_shards: Number of shards to create
        
    Returns:
        List of optimally balanced data shards
    """
    if len(data) == 0 or num_shards <= 0:
        return []
    
    if num_shards >= len(data):
        # If more shards than data points, create one shard per data point
        return [data[i:i+1] for i in range(len(data))]
    
    # Use a greedy algorithm to balance shard sizes
    shard_sizes = [0] * num_shards
    shard_assignments = [[] for _ in range(num_shards)]
    
    # Sort data indices by some criteria (e.g., data size if applicable)
    data_indices = list(range(len(data)))
    
    # Assign each data point to the shard with minimum current size
    for idx in data_indices:
        min_shard = min(range(num_shards), key=lambda i: shard_sizes[i])
        shard_assignments[min_shard].append(idx)
        shard_sizes[min_shard] += 1
    
    # Create actual shards
    shards = []
    for assignment in shard_assignments:
        if assignment:
            shard = data[assignment]
            shards.append(shard)
        else:
            # Empty shard
            shards.append(np.array([]))
    
    # Log balance statistics
    non_empty_shards = [s for s in shards if len(s) > 0]
    if non_empty_shards:
        shard_sizes = [len(shard) for shard in non_empty_shards]
        balance_ratio = min(shard_sizes) / max(shard_sizes) if max(shard_sizes) > 0 else 1.0
        logger.info(f"Created {len(shards)} shards with balance ratio: {balance_ratio:.3f}")
    
    return shards

def save_results(results: Dict[str, Any], filename: str = "results.json"):
    """
    Save benchmark results to file.
    
    Args:
        results: Dictionary containing benchmark results
        filename: Output filename
    """
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {filename}")

def load_results(filename: str = "results.json") -> Dict[str, Any]:
    """
    Load benchmark results from file.
    
    Args:
        filename: Input filename
        
    Returns:
        Dictionary containing benchmark results
    """
    import json
    with open(filename, 'r') as f:
        return json.load(f)

def print_results_summary(results: Dict[str, Any]):
    """
    Print a summary of benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
    """
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    for approach, metrics in results.items():
        print(f"\n{approach.upper()}:")
        print(f"  Throughput: {metrics['throughput']:.2f} inferences/sec")
        print(f"  Latency: {metrics['latency_ms']:.2f} ms/batch")
        print(f"  CPU Usage: {metrics['avg_cpu_percent']:.1f}%")
        print(f"  Memory Usage: {metrics['avg_memory_mb']:.1f} MB")
        print(f"  Total Duration: {metrics['total_duration']:.2f} seconds")
        print(f"  Total Inferences: {metrics['total_inferences']}")

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for the current node.
    
    Returns:
        Dictionary containing system information
    """
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'platform': os.uname().sysname,
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    }
