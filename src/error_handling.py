"""
Comprehensive error handling and recovery mechanisms for data partitioning experiments.

This module provides robust error handling, retry logic, and recovery mechanisms
for all partitioning approaches.
"""
import time
import logging
import traceback
from typing import Dict, Any, Callable, Optional, List, Tuple
from functools import wraps
import psutil
import os

logger = logging.getLogger(__name__)

class PartitioningError(Exception):
    """Base exception for partitioning-related errors."""
    pass

class WorkerError(PartitioningError):
    """Error related to worker processes."""
    pass

class ResourceError(PartitioningError):
    """Error related to resource constraints (memory, CPU, etc.)."""
    pass

class NetworkError(PartitioningError):
    """Error related to network communication."""
    pass

class DataError(PartitioningError):
    """Error related to data processing."""
    pass

class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

def retry_on_failure(retry_config: RetryConfig = None, 
                    exceptions: Tuple = (Exception,),
                    on_retry: Optional[Callable] = None):
    """
    Decorator for retrying functions on failure.
    
    Args:
        retry_config: Retry configuration
        exceptions: Tuple of exceptions to retry on
        on_retry: Optional callback function called on each retry
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == retry_config.max_retries:
                        logger.error(f"Function {func.__name__} failed after {retry_config.max_retries} retries: {str(e)}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{retry_config.max_retries + 1}): {str(e)}. Retrying in {delay:.2f}s...")
                    
                    if on_retry:
                        on_retry(attempt, e, delay)
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

class ResourceMonitor:
    """Monitor system resources and detect potential issues."""
    
    def __init__(self, memory_threshold: float = 0.9, cpu_threshold: float = 0.95):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
    
    def check_memory(self) -> Dict[str, Any]:
        """Check memory usage and return status."""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used,
            'free': memory.free,
            'is_critical': memory.percent > (self.memory_threshold * 100)
        }
    
    def check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage and return status."""
        cpu_percent = psutil.cpu_percent(interval=1.0)
        return {
            'percent': cpu_percent,
            'count': psutil.cpu_count(),
            'is_critical': cpu_percent > (self.cpu_threshold * 100)
        }
    
    def check_disk(self, path: str = '.') -> Dict[str, Any]:
        """Check disk usage and return status."""
        disk = psutil.disk_usage(path)
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': (disk.used / disk.total) * 100,
            'is_critical': (disk.used / disk.total) > self.memory_threshold
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'memory': self.check_memory(),
            'cpu': self.check_cpu(),
            'disk': self.check_disk(),
            'timestamp': time.time()
        }

class ErrorRecoveryManager:
    """Manages error recovery strategies for different types of failures."""
    
    def __init__(self, resource_monitor: Optional[ResourceMonitor] = None):
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.recovery_strategies = {
            'memory': self._recover_from_memory_error,
            'cpu': self._recover_from_cpu_error,
            'network': self._recover_from_network_error,
            'worker': self._recover_from_worker_error,
            'data': self._recover_from_data_error
        }
    
    def analyze_error(self, error: Exception) -> str:
        """Analyze error and determine recovery strategy."""
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ['memory', 'oom', 'out of memory']):
            return 'memory'
        elif any(keyword in error_str for keyword in ['cpu', 'timeout', 'slow']):
            return 'cpu'
        elif any(keyword in error_str for keyword in ['network', 'connection', 'socket']):
            return 'network'
        elif any(keyword in error_str for keyword in ['worker', 'process', 'actor']):
            return 'worker'
        elif any(keyword in error_str for keyword in ['data', 'tensor', 'array', 'shape']):
            return 'data'
        else:
            return 'unknown'
    
    def recover_from_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """
        Attempt to recover from an error.
        
        Args:
            error: The exception that occurred
            context: Additional context for recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if context is None:
            context = {}
        
        error_type = self.analyze_error(error)
        logger.info(f"Attempting recovery from {error_type} error: {str(error)}")
        
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {str(recovery_error)}")
                return False
        else:
            logger.warning(f"No recovery strategy for error type: {error_type}")
            return False
    
    def _recover_from_memory_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from memory-related errors."""
        logger.info("Attempting memory error recovery...")
        
        # Check current memory status
        memory_status = self.resource_monitor.check_memory()
        if memory_status['is_critical']:
            logger.warning(f"Memory usage is critical: {memory_status['percent']:.1f}%")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear PyTorch cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Wait a bit for memory to be freed
            time.sleep(2)
            
            # Check if memory improved
            new_memory_status = self.resource_monitor.check_memory()
            if new_memory_status['percent'] < memory_status['percent']:
                logger.info("Memory recovery successful")
                return True
        
        return False
    
    def _recover_from_cpu_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from CPU-related errors."""
        logger.info("Attempting CPU error recovery...")
        
        # Check current CPU status
        cpu_status = self.resource_monitor.check_cpu()
        if cpu_status['is_critical']:
            logger.warning(f"CPU usage is critical: {cpu_status['percent']:.1f}%")
            
            # Wait for CPU to cool down
            time.sleep(5)
            
            # Check if CPU usage improved
            new_cpu_status = self.resource_monitor.check_cpu()
            if new_cpu_status['percent'] < cpu_status['percent']:
                logger.info("CPU recovery successful")
                return True
        
        return False
    
    def _recover_from_network_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from network-related errors."""
        logger.info("Attempting network error recovery...")
        
        # Wait and retry
        time.sleep(2)
        return True
    
    def _recover_from_worker_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from worker-related errors."""
        logger.info("Attempting worker error recovery...")
        
        # Wait for workers to stabilize
        time.sleep(3)
        return True
    
    def _recover_from_data_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from data-related errors."""
        logger.info("Attempting data error recovery...")
        
        # For data errors, we might need to adjust batch sizes or chunk sizes
        if 'context' in context:
            # This would be implemented based on specific data error types
            pass
        
        return False

def safe_execute(func: Callable, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
    """
    Safely execute a function and return result with any exception.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, exception)
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        logger.error(f"Function {func.__name__} failed: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None, e

def validate_environment() -> Dict[str, bool]:
    """
    Validate the execution environment.
    
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    # Check Python version
    import sys
    results['python_version'] = sys.version_info >= (3, 7)
    
    # Check required packages
    required_packages = ['torch', 'numpy', 'psutil']
    for package in required_packages:
        try:
            __import__(package)
            results[f'package_{package}'] = True
        except ImportError:
            results[f'package_{package}'] = False
    
    # Check system resources
    try:
        memory = psutil.virtual_memory()
        results['sufficient_memory'] = memory.available > 1024 * 1024 * 1024  # 1GB
    except Exception:
        results['sufficient_memory'] = False
    
    try:
        cpu_count = psutil.cpu_count()
        results['sufficient_cpu'] = cpu_count >= 2
    except Exception:
        results['sufficient_cpu'] = False
    
    return results

def log_system_info():
    """Log comprehensive system information for debugging."""
    logger.info("=== SYSTEM INFORMATION ===")
    
    # Python version
    import sys
    logger.info(f"Python version: {sys.version}")
    
    # System resources
    try:
        memory = psutil.virtual_memory()
        logger.info(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
        
        cpu_count = psutil.cpu_count()
        logger.info(f"CPU: {cpu_count} cores")
        
        disk = psutil.disk_usage('.')
        logger.info(f"Disk: {disk.total / (1024**3):.1f} GB total, {disk.free / (1024**3):.1f} GB free")
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
    
    # Environment variables
    relevant_env_vars = ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'RAY_ADDRESS']
    for var in relevant_env_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"{var}: {value}")
    
    logger.info("=== END SYSTEM INFORMATION ===")

