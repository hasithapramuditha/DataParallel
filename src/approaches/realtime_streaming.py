"""
Real-time Streaming Data Partitioning Implementation.

This approach handles real-time data streams at 500 samples/s with different
partitioning strategies for continuous inference scenarios.
"""
import torch
import numpy as np
import time
import psutil
import os
import sys
import argparse
import logging
import threading
import queue
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    get_model, 
    create_realtime_stream, 
    create_streaming_partitioner,
    MetricsCollector
)

logger = logging.getLogger(__name__)

class RealTimeInferenceWorker:
    """Worker for real-time inference processing."""
    
    def __init__(self, worker_id: int, model: torch.nn.Module):
        """
        Initialize real-time inference worker.
        
        Args:
            worker_id: Unique worker identifier
            model: Pre-trained model for inference
        """
        self.worker_id = worker_id
        self.model = model
        self.metrics = MetricsCollector()
        self.processed_samples = 0
        self.inference_queue = queue.Queue()
        self.running = False
        
    def start_worker(self):
        """Start the worker thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_inferences)
        self.worker_thread.start()
        logger.info(f"Worker {self.worker_id} started")
        
    def stop_worker(self):
        """Stop the worker thread."""
        self.running = False
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join()
        logger.info(f"Worker {self.worker_id} stopped")
        
    def add_sample(self, sample_data: np.ndarray, label: int):
        """
        Add a sample to the worker's processing queue.
        
        Args:
            sample_data: Image data
            label: Sample label
        """
        if sample_data is not None:
            self.inference_queue.put((sample_data, label))
            
    def _process_inferences(self):
        """Process inference queue continuously."""
        while self.running:
            try:
                # Get sample with timeout to allow checking running status
                sample_data, label = self.inference_queue.get(timeout=0.1)
                
                # Perform inference
                start_time = time.time()
                
                # Convert to tensor and add batch dimension
                input_tensor = torch.from_numpy(sample_data).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                
                inference_time = time.time() - start_time
                
                # Record metrics
                self.metrics.record_cpu_usage()
                self.metrics.record_memory_usage()
                self.metrics.add_inferences(1)
                self.processed_samples += 1
                
                logger.debug(f"Worker {self.worker_id}: Processed sample {self.processed_samples} "
                           f"in {inference_time:.4f}s, prediction: {prediction}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                
    def get_worker_metrics(self) -> Dict[str, Any]:
        """Get worker-specific metrics."""
        return {
            'worker_id': self.worker_id,
            'processed_samples': self.processed_samples,
            'queue_size': self.inference_queue.qsize(),
            'metrics': self.metrics.get_metrics() if self.metrics.start_time else {}
        }

class RealTimeStreamingPartitioner:
    """
    Main class for real-time streaming data partitioning experiments.
    """
    
    def __init__(self, num_workers: int = 2, strategy: str = "uniform"):
        """
        Initialize real-time streaming partitioner.
        
        Args:
            num_workers: Number of worker nodes
            strategy: Partitioning strategy ("uniform", "dynamic", "sharded")
        """
        self.num_workers = num_workers
        self.strategy = strategy
        self.model = get_model()
        self.workers = []
        self.partitioner = create_streaming_partitioner(num_workers, strategy)
        self.metrics = MetricsCollector()
        
    def setup_workers(self):
        """Setup worker nodes."""
        self.workers = []
        for i in range(self.num_workers):
            worker = RealTimeInferenceWorker(i, self.model)
            self.workers.append(worker)
            
    def start_workers(self):
        """Start all worker threads."""
        for worker in self.workers:
            worker.start_worker()
            
    def stop_workers(self):
        """Stop all worker threads."""
        for worker in self.workers:
            worker.stop_worker()
            
    def run_realtime_experiment(self, samples_per_second: int = 500, 
                              max_samples: int = 10000) -> Dict[str, Any]:
        """
        Run real-time streaming experiment.
        
        Args:
            samples_per_second: Rate of data generation
            max_samples: Maximum samples to process
            
        Returns:
            Dictionary containing experiment results
        """
        logger.info(f"Starting real-time streaming experiment: {self.strategy} "
                   f"with {self.num_workers} workers at {samples_per_second} samples/s")
        
        # Setup and start workers
        self.setup_workers()
        self.start_workers()
        
        # Create data stream
        data_stream = create_realtime_stream(samples_per_second, max_samples, self.num_workers)
        data_stream.start_stream()
        
        # Start experiment timing
        self.metrics.start_timing()
        
        try:
            # Process stream
            while not data_stream.is_stream_complete():
                sample_data, label = data_stream.get_next_sample()
                
                # Partition sample according to strategy
                partitioned_samples = self.partitioner.partition_sample(sample_data, label)
                
                # Send to appropriate workers
                for i, (worker_data, worker_label) in enumerate(partitioned_samples):
                    if i < len(self.workers):
                        self.workers[i].add_sample(worker_data, worker_label)
                
                # Record overall metrics
                self.metrics.record_cpu_usage()
                self.metrics.record_memory_usage()
                
        except KeyboardInterrupt:
            logger.info("Experiment interrupted by user")
        finally:
            # Stop workers
            self.stop_workers()
            self.metrics.stop_timing()
            
        # Collect results
        results = self._collect_results(data_stream)
        return results
        
    def _collect_results(self, data_stream) -> Dict[str, Any]:
        """Collect and aggregate results from all workers."""
        # Get stream metrics
        stream_metrics = data_stream.get_stream_metrics()
        
        # Get worker metrics
        worker_metrics = []
        total_processed = 0
        
        for worker in self.workers:
            worker_metric = worker.get_worker_metrics()
            worker_metrics.append(worker_metric)
            total_processed += worker_metric['processed_samples']
        
        # Calculate aggregate metrics
        self.metrics.add_inferences(total_processed)
        overall_metrics = self.metrics.get_metrics()
        
        return {
            'strategy': self.strategy,
            'num_workers': self.num_workers,
            'stream_metrics': stream_metrics,
            'worker_metrics': worker_metrics,
            'overall_metrics': overall_metrics,
            'total_processed': total_processed,
            'efficiency': (total_processed / stream_metrics.get('generated_samples', 1)) * 100
        }

def run_realtime_streaming_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run real-time streaming experiment with given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing experiment results
    """
    partitioner = RealTimeStreamingPartitioner(
        num_workers=config.get('num_workers', 2),
        strategy=config.get('strategy', 'uniform')
    )
    
    return partitioner.run_realtime_experiment(
        samples_per_second=config.get('samples_per_second', 500),
        max_samples=config.get('max_samples', 10000)
    )

def main():
    """Main function for running real-time streaming experiments."""
    parser = argparse.ArgumentParser(description='Real-time Streaming Data Partitioning')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--strategy', choices=['uniform', 'dynamic', 'sharded'], 
                      default='uniform', help='Partitioning strategy')
    parser.add_argument('--samples-per-second', type=int, default=500, 
                      help='Data generation rate')
    parser.add_argument('--max-samples', type=int, default=10000, 
                      help='Maximum samples to process')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run experiment
    config = {
        'num_workers': args.workers,
        'strategy': args.strategy,
        'samples_per_second': args.samples_per_second,
        'max_samples': args.max_samples
    }
    
    print(f"🚀 Starting Real-time Streaming Experiment")
    print(f"   Strategy: {args.strategy}")
    print(f"   Workers: {args.workers}")
    print(f"   Rate: {args.samples_per_second} samples/s")
    print(f"   Max Samples: {args.max_samples}")
    
    results = run_realtime_streaming_experiment(config)
    
    # Print results
    print("\n" + "="*60)
    print("REAL-TIME STREAMING RESULTS")
    print("="*60)
    print(f"Strategy: {results['strategy']}")
    print(f"Workers: {results['num_workers']}")
    print(f"Total Processed: {results['total_processed']}")
    print(f"Efficiency: {results['efficiency']:.2f}%")
    
    stream_metrics = results['stream_metrics']
    print(f"\nStream Metrics:")
    print(f"  Target Rate: {stream_metrics.get('target_rate', 0)} samples/s")
    print(f"  Actual Rate: {stream_metrics.get('actual_rate', 0):.2f} samples/s")
    print(f"  Rate Accuracy: {stream_metrics.get('rate_accuracy', 0):.2f}%")
    print(f"  Elapsed Time: {stream_metrics.get('elapsed_time', 0):.2f}s")
    
    overall_metrics = results['overall_metrics']
    print(f"\nOverall Metrics:")
    print(f"  Throughput: {overall_metrics.get('throughput', 0):.2f} inf/s")
    print(f"  Latency: {overall_metrics.get('latency_ms', 0):.2f} ms/batch")
    print(f"  CPU Usage: {overall_metrics.get('avg_cpu_percent', 0):.1f}%")
    print(f"  Memory Usage: {overall_metrics.get('avg_memory_mb', 0):.1f} MB")
    
    print(f"\nWorker Details:")
    for worker_metric in results['worker_metrics']:
        print(f"  Worker {worker_metric['worker_id']}: "
              f"{worker_metric['processed_samples']} samples, "
              f"Queue: {worker_metric['queue_size']}")

if __name__ == "__main__":
    main()
