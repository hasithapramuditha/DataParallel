# DataParallel - Comprehensive Data Partitioning Framework

## 🎯 **Project Overview**

DataParallel is a comprehensive framework for benchmarking and comparing different data partitioning strategies for distributed machine learning inference. It implements three distinct partitioning approaches using different distributed computing frameworks to optimize performance across various scenarios.

### **Key Features**
- 🚀 **Three Partitioning Strategies**: Uniform (PyTorch DDP), Dynamic (Ray), Sharded (Multiprocessing)
- 📊 **Comprehensive Benchmarking**: Matrix benchmarks and streaming benchmarks across different worker configurations
- 🔧 **Unified Configuration**: Standardized configuration system across all approaches
- 🛡️ **Robust Error Handling**: Retry mechanisms and error recovery
- 📈 **Performance Monitoring**: Detailed metrics collection and visualization
- ☁️ **Cloud Support**: GCP deployment capabilities
- 🎨 **Advanced Plotting**: Comprehensive visualization tools for results analysis

## 🏗️ **Architecture**

### **Core Components**

```
DataParallel/
├── main.py                          # Main entry point (175 lines)
├── plot_streaming_results.py        # Comprehensive plotting utility (384 lines)
├── src/
│   ├── approaches/                  # Partitioning implementations
│   │   ├── uniform_partitioning.py  # PyTorch DDP (241 lines)
│   │   ├── dynamic_partitioning.py  # Ray with load balancing (476 lines)
│   │   └── sharded_partitioning.py  # Multiprocessing (376 lines)
│   ├── config.py                    # Unified configuration system (230 lines)
│   ├── error_handling.py            # Error recovery and retry mechanisms (364 lines)
│   ├── utils.py                     # Shared utilities and data loading (538 lines)
│   ├── streaming_benchmark.py       # Streaming benchmark framework (585 lines)
│   └── matrix_benchmark.py          # Matrix benchmarking across configurations (193 lines)
├── cloud/                           # Cloud deployment (GCP)
│   ├── run_cloud_experiments.py     # Cloud experiments (486 lines)
│   ├── setup_gcp_cluster.py         # GCP cluster setup (451 lines)
│   ├── cloud_config.yaml            # Cloud experiment configuration
│   └── gcp_config.yaml              # GCP cluster configuration
├── scripts/                         # Setup and testing scripts
│   ├── setup_local_venv.py          # Environment setup (332 lines)
│   └── test_installation.py         # Installation test (267 lines)
├── data/                            # CIFAR-10 dataset
├── results/                         # Benchmark results and plots
└── Documentation files
```

## 🚀 **Partitioning Strategies**

### **1. Uniform Partitioning (PyTorch DDP)**
- **Framework**: PyTorch DistributedDataParallel
- **Strategy**: Even data distribution across workers
- **Best For**: Homogeneous workloads, balanced data
- **Key Features**:
  - Automatic data distribution via DistributedSampler
  - Synchronous gradient updates
  - Built-in PyTorch optimizations

### **2. Dynamic Partitioning (Ray)**
- **Framework**: Ray distributed computing
- **Strategy**: Real-time load balancing based on worker performance
- **Best For**: Heterogeneous workloads, varying data complexity
- **Key Features**:
  - Real-time actor performance tracking
  - Intelligent task assignment
  - Exponential moving average for performance metrics
  - Load balancing efficiency calculation

### **3. Sharded Partitioning (Multiprocessing)**
- **Framework**: Python multiprocessing
- **Strategy**: Fixed data sharding with optimized distribution
- **Best For**: Large datasets, memory-constrained environments
- **Key Features**:
  - Greedy algorithm for optimal shard distribution
  - Enhanced memory management
  - Chunked processing for large datasets
  - Process isolation for stability

## 🌊 **Streaming Benchmark System**

### **3. Comprehensive Streaming Benchmark**
- **Framework**: Custom streaming benchmark system
- **Strategy**: CIFAR-10 data streams with configurable rates
- **Best For**: Real-time performance evaluation, streaming scenarios
- **Key Features**:
  - Configurable stream rates (1000+ samples/s)
  - Customizable node counts (1,2,4,8,16+ nodes)
  - Multiple iterations for statistical significance
  - Comprehensive performance analysis
  - Real-time metrics collection
  - **Uniform partitioning optimized for streaming scenarios**
  - **Dynamic load balancing with Ray actors**
  - **Sharded processing with multiprocessing**

## 📊 **Performance Metrics**

### **Collected Metrics**
- **Throughput**: Inferences per second
- **Latency**: Average inference time per batch
- **CPU Usage**: Average CPU utilization
- **Memory Usage**: Peak and average memory consumption
- **Load Balancing Efficiency**: Distribution quality (Dynamic)
- **Shard Distribution Efficiency**: Balance ratio (Sharded)

### **Benchmark Types**
1. **Individual Approach Testing**: Single approach performance
2. **Matrix Benchmarking**: Cross-approach comparison across worker counts
3. **Streaming Benchmark**: Comprehensive streaming evaluation with customizable node counts
4. **Cloud Benchmarking**: Distributed cluster performance

### **Streaming Benchmark Features**
- **Real-time Data Streams**: CIFAR-10 data processed as continuous streams
- **Configurable Parameters**: Stream rate, total samples, iterations
- **Statistical Analysis**: Multiple iterations with comprehensive averaging
- **Performance Rankings**: Best performer identification by node count
- **Scaling Analysis**: Efficiency calculations across node configurations
- **Automatic Plotting**: Comprehensive visualizations generated automatically after benchmark completion

## 🛠️ **Installation & Setup**

### **Prerequisites**
- Python 3.8+
- PyTorch 2.0+
- Ray 2.2+
- Dask 2023.5+
- Additional dependencies in `requirements.txt`

### **Quick Start**
```bash
# Clone and setup
git clone <repository>
cd DataParallel

# Install dependencies
pip install -r requirements.txt

# Test installation
python main.py --test

# Run all approaches
python main.py --approach all --workers 2 --samples 1000
```

### **Using Scripts**
```bash
# Linux/Mac
./run.sh --approach all --workers 4 --samples 2000

# Windows
run.bat --approach all --workers 4 --samples 2000
```

## 🎮 **Usage Examples**

### **Basic Usage**
```bash
# Test individual approaches
python main.py --approach uniform --workers 2 --samples 1000
python main.py --approach dynamic --workers 4 --samples 2000
python main.py --approach sharded --workers 2 --samples 1000

# Run all approaches for comparison
python main.py --approach all --workers 2 --samples 1000
```

### **Matrix Benchmarking**
```bash
# Comprehensive benchmarking across worker configurations
python main.py --matrix --samples 1000

# Custom worker configurations
python src/matrix_benchmark.py --sizes 1,2,4,8 --num-samples 2000 --batch-size 128
```

### **Streaming Benchmark**
```bash
# Comprehensive streaming benchmark (3 strategies x default node counts: 1,2,4,8)
python main.py --streaming-benchmark

# Custom streaming configuration with specific node counts
python main.py --streaming-benchmark --nodes 1,2,4 --stream-rate 1000 --total-samples 4000 --iterations 3

# Test up to 16 nodes
python main.py --streaming-benchmark --nodes 1,2,4,8,16 --stream-rate 1000 --total-samples 8000 --iterations 1

# Different stream rates and sample sizes
python main.py --streaming-benchmark --stream-rate 500 --total-samples 30000 --iterations 5
```

### **Cloud Deployment**
```bash
# Setup GCP cluster
python cloud/setup_gcp_cluster.py

# Run cloud experiments
python main.py --cloud-matrix --samples 10000
```

## 🔧 **Configuration System**

### **Unified Configuration**
All approaches use the `PartitioningConfig` class for consistent parameter management:

```python
from src.config import PartitioningConfig

config = PartitioningConfig({
    'num_workers': 4,
    'num_samples': 10000,
    'batch_size': 128,
    'chunk_size': 32,
    'num_shards': 4
})

# Get approach-specific configurations
uniform_config = config.get_uniform_config()
dynamic_config = config.get_dynamic_config()
sharded_config = config.get_sharded_config()
```

### **Configuration Parameters**
- **Common**: `num_workers`, `num_samples`, `batch_size`, `chunk_size`
- **Uniform**: `world_size`, `master_addr`, `master_port`
- **Dynamic**: `ray_address`, load balancing parameters
- **Sharded**: `num_shards`, `process_in_chunks`
- **Real-time**: `samples_per_second`, streaming parameters

## 📈 **Results & Visualization**

### **Advanced Plotting System**
The project includes a comprehensive plotting system (`plot_streaming_results.py`) that generates multiple visualizations:

```bash
# Generate all plots from streaming benchmark results
python plot_streaming_results.py

# Plots are automatically generated after streaming benchmark completion
python main.py --streaming-benchmark  # Plots generated automatically
```

**Generated Visualizations:**
- **Throughput Comparison**: Bar charts and line plots showing performance across strategies and nodes
- **Resource Utilization**: CPU and memory usage analysis
- **Scaling Efficiency**: Efficiency analysis and trade-off visualizations
- **Performance Heatmaps**: Color-coded performance matrices
- **Comprehensive Dashboard**: Multi-panel analysis with statistics table

### **Output Structure**
```
results/
├── matrix/
│   ├── combined_results.csv      # All benchmark results
│   └── speedup_efficiency.csv    # Performance scaling analysis
├── plots/
│   ├── aggregate_1w.png          # 1 worker performance
│   ├── aggregate_2w.png          # 2 worker performance
│   ├── aggregate_4w.png          # 4 worker performance
│   ├── streaming_throughput_comparison.png      # Throughput comparison
│   ├── streaming_resource_utilization.png       # Resource utilization
│   ├── streaming_scaling_efficiency.png         # Scaling efficiency
│   ├── streaming_performance_heatmaps.png       # Performance heatmaps
│   └── streaming_comprehensive_dashboard.png    # Comprehensive dashboard
└── streaming/
    ├── averaged_results.csv      # Streaming benchmark results
    ├── detailed_results.json     # Detailed streaming metrics
    └── performance_summary.json  # Performance summary
```

### **Performance Analysis**
- **Throughput Comparison**: Direct performance comparison
- **Scaling Efficiency**: How well approaches scale with workers
- **Resource Utilization**: CPU and memory usage patterns
- **Load Balancing**: Distribution quality metrics

## 🛡️ **Error Handling & Recovery**

### **Robust Error Management**
- **Retry Mechanisms**: Exponential backoff for transient failures
- **Error Classification**: Specific handling for different error types
- **Recovery Strategies**: Automatic recovery from common failures
- **System Validation**: Health checks and resource monitoring

### **Error Types Handled**
- **Network Issues**: Connection timeouts, communication failures
- **Resource Exhaustion**: Memory limits, CPU overload
- **Framework Errors**: Ray/Dask/PyTorch specific issues
- **Data Issues**: Corrupted data, format problems

## ☁️ **Cloud Deployment**

### **GCP Support**
- **Cluster Setup**: Automated GCP cluster creation
- **Distributed Execution**: Multi-node benchmarking
- **Resource Management**: Automatic scaling and optimization
- **Cost Optimization**: Efficient resource utilization

### **Cloud Features**
- **Auto-scaling**: Dynamic worker allocation
- **Fault Tolerance**: Automatic recovery from node failures
- **Monitoring**: Cloud-native monitoring and logging
- **Cost Tracking**: Resource usage and cost analysis

## 🧪 **Testing & Validation**

### **Test Suite**
```bash
# Comprehensive testing
python main.py --test

# Individual component testing
python scripts/test_installation.py
```

### **Test Coverage**
- **Package Imports**: All dependencies verified
- **Basic Functionality**: Core components tested
- **Dataset Loading**: CIFAR-10 data validation
- **Model Loading**: ResNet-18 inference testing
- **Distributed Frameworks**: Ray and Dask initialization
- **Project Structure**: File organization validation

## 📊 **Performance Benchmarks**

### **Typical Performance Results**
| Approach | Workers | Throughput | Use Case |
|----------|---------|------------|----------|
| **Uniform** | 2 | 300+ inf/s | Balanced workloads |
| **Dynamic** | 4 | 200+ inf/s | Variable complexity |
| **Sharded** | 2 | 150+ inf/s | Large datasets |

### **Streaming Benchmark Results**
| Strategy | 1 Node | 2 Nodes | 4 Nodes | Best For |
|----------|--------|---------|---------|----------|
| **Uniform** | 353.47 inf/s | 367.22 inf/s | 371.35 inf/s | **Best overall performance** |
| **Dynamic** | 119.26 inf/s | 186.09 inf/s | 265.19 inf/s | Load balancing |
| **Sharded** | 149.87 inf/s | 156.15 inf/s | 138.19 inf/s | Memory efficiency |

### **Scaling Characteristics**
- **Uniform**: Consistent high performance across all node counts (0.39 efficiency)
- **Dynamic**: Best scaling efficiency with load balancing (0.67 efficiency)
- **Sharded**: Stable performance across different node counts (0.38 efficiency)
- **Real-time**: Consistent throughput under load

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Installation Problems**
```bash
# Verify Python version
python --version  # Should be 3.8+

# Check dependencies
pip list | grep -E "(torch|ray|dask)"

# Reinstall if needed
pip install -r requirements.txt --force-reinstall
```

#### **Performance Issues**
- **Low Throughput**: Check worker count and batch size
- **Memory Errors**: Reduce batch size or enable chunking
- **Connection Issues**: Verify network and firewall settings

#### **Framework-Specific Issues**
- **Ray**: Check Ray dashboard for actor status
- **PyTorch DDP**: Verify master address and port availability
- **Multiprocessing**: Check system resource limits

### **Debug Mode**
```bash
# Enable verbose logging
export PYTHONPATH=$PWD/src:$PYTHONPATH
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from approaches.uniform_partitioning import run_uniform_partitioning
# ... run your code
"
```

## 🤝 **Contributing**

### **Development Setup**
```bash
# Clone repository
git clone <repository>
cd DataParallel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

### **Code Standards**
- **Python**: Follow PEP 8 style guidelines
- **Documentation**: Comprehensive docstrings for all functions
- **Testing**: Unit tests for new features
- **Type Hints**: Use type annotations throughout

### **Adding New Approaches**
1. Create new file in `src/approaches/`
2. Implement required interface methods
3. Add configuration support in `config.py`
4. Update `main.py` and `matrix_benchmark.py`
5. Add comprehensive tests

## 📚 **API Reference**

### **Core Functions**

#### **PartitioningConfig**
```python
class PartitioningConfig:
    def __init__(self, config_dict: Dict[str, Any])
    def validate(self) -> bool
    def get_uniform_config(self) -> Dict[str, Any]
    def get_dynamic_config(self) -> Dict[str, Any]
    def get_sharded_config(self) -> Dict[str, Any]
```

#### **Approach Functions**
```python
def run_uniform_partitioning(config: Dict[str, Any]) -> Dict[str, Any]
def run_dynamic_partitioning(config: Dict[str, Any]) -> Dict[str, Any]
def run_sharded_partitioning(config: Dict[str, Any]) -> Dict[str, Any]
def run_realtime_streaming_experiment(config: PartitioningConfig) -> Dict[str, Any]
```

#### **Benchmarking**
```python
class BenchmarkRunner:
    def __init__(self, config: PartitioningConfig)
    def run_all_benchmarks(self) -> Dict[str, Any]
    def run_matrix_benchmark(self) -> Dict[str, Any]
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **PyTorch Team**: For the excellent distributed computing framework
- **Ray Team**: For the powerful distributed computing platform
- **Dask Team**: For the flexible parallel computing library
- **CIFAR-10 Dataset**: For providing the benchmark dataset

## 📞 **Support**

For questions, issues, or contributions:
- **Issues**: Create an issue on the repository
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check this file and inline code documentation

---

**DataParallel** - Empowering distributed machine learning with intelligent data partitioning strategies.
