"""
Standardized configuration for all data partitioning approaches.

This module provides a unified configuration system that ensures
consistent parameters across all partitioning approaches.
"""
from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)

class PartitioningConfig:
    """Standardized configuration for data partitioning experiments."""
    
    # Common parameters
    DEFAULT_NUM_WORKERS = 4
    DEFAULT_NUM_SAMPLES = 10000
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_CHUNK_SIZE = 32
    
    # Approach-specific defaults
    DEFAULT_NUM_SHARDS = 4
    DEFAULT_MASTER_ADDR = 'localhost'
    DEFAULT_MASTER_PORT = '12355'
    DEFAULT_DATA_LOADER_WORKERS = 1
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with defaults and optional overrides.
        
        Args:
            config_dict: Optional dictionary to override default values
        """
        # Set defaults
        self.num_workers = self.DEFAULT_NUM_WORKERS
        self.num_samples = self.DEFAULT_NUM_SAMPLES
        self.batch_size = self.DEFAULT_BATCH_SIZE
        self.chunk_size = self.DEFAULT_CHUNK_SIZE
        self.num_shards = self.DEFAULT_NUM_SHARDS
        self.master_addr = self.DEFAULT_MASTER_ADDR
        self.master_port = self.DEFAULT_MASTER_PORT
        self.data_loader_workers = self.DEFAULT_DATA_LOADER_WORKERS
        
        # Optional parameters
        self.ray_address = None
        self.process_in_chunks = True
        
        # Apply overrides if provided
        if config_dict:
            self.update_from_dict(config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def get_uniform_config(self) -> Dict[str, Any]:
        """Get configuration for uniform partitioning."""
        return {
            'world_size': self.num_workers,
            'batch_size': self.batch_size,
            'num_workers': self.data_loader_workers,
            'master_addr': self.master_addr,
            'master_port': self.master_port
        }
    
    def get_dynamic_config(self) -> Dict[str, Any]:
        """Get configuration for dynamic partitioning."""
        return {
            'num_workers': self.num_workers,
            'num_samples': self.num_samples,
            'chunk_size': self.chunk_size,
            'ray_address': self.ray_address
        }
    
    def get_sharded_config(self) -> Dict[str, Any]:
        """Get configuration for sharded partitioning (multiprocessing-based)."""
        return {
            'num_workers': self.num_workers,
            'num_samples': self.num_samples,
            'num_shards': self.num_shards,
            'process_in_chunks': self.process_in_chunks,
            'chunk_size': self.chunk_size
        }
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        if self.num_workers <= 0:
            errors.append("num_workers must be positive")
        
        if self.num_samples <= 0:
            errors.append("num_samples must be positive")
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.num_shards <= 0:
            errors.append("num_shards must be positive")
        
        if self.data_loader_workers < 0:
            errors.append("data_loader_workers must be non-negative")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'num_workers': self.num_workers,
            'num_samples': self.num_samples,
            'batch_size': self.batch_size,
            'chunk_size': self.chunk_size,
            'num_shards': self.num_shards,
            'master_addr': self.master_addr,
            'master_port': self.master_port,
            'data_loader_workers': self.data_loader_workers,
            'ray_address': self.ray_address,
            'process_in_chunks': self.process_in_chunks
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"PartitioningConfig(num_workers={self.num_workers}, num_samples={self.num_samples}, batch_size={self.batch_size})"

def create_config_from_args(args) -> PartitioningConfig:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        PartitioningConfig instance
    """
    config_dict = {}
    
    # Map common argument names to config parameters
    if hasattr(args, 'num_workers'):
        config_dict['num_workers'] = args.num_workers
    if hasattr(args, 'workers'):
        config_dict['num_workers'] = args.workers
    if hasattr(args, 'world_size'):
        config_dict['num_workers'] = args.world_size
    
    if hasattr(args, 'num_samples'):
        config_dict['num_samples'] = args.num_samples
    if hasattr(args, 'samples'):
        config_dict['num_samples'] = args.samples
    
    if hasattr(args, 'batch_size'):
        config_dict['batch_size'] = args.batch_size
    
    if hasattr(args, 'chunk_size'):
        config_dict['chunk_size'] = args.chunk_size
    
    if hasattr(args, 'num_shards'):
        config_dict['num_shards'] = args.num_shards
    
    if hasattr(args, 'master_addr'):
        config_dict['master_addr'] = args.master_addr
    
    if hasattr(args, 'master_port'):
        config_dict['master_port'] = args.master_port
    
    if hasattr(args, 'ray_address'):
        config_dict['ray_address'] = args.ray_address
    
    return PartitioningConfig(config_dict)

def load_config_from_file(config_file: str) -> PartitioningConfig:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        PartitioningConfig instance
    """
    import json
    import yaml
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        elif config_file.endswith('.json'):
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file}")
    
    return PartitioningConfig(config_dict)

def save_config_to_file(config: PartitioningConfig, config_file: str):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: PartitioningConfig instance
        config_file: Path to output configuration file
    """
    import json
    import yaml
    
    config_dict = config.to_dict()
    
    with open(config_file, 'w') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif config_file.endswith('.json'):
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file}")
    
    logger.info(f"Configuration saved to {config_file}")
