# DataParallel - Distributed Data Partitioning Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Ray](https://img.shields.io/badge/Ray-2.2%2B-orange.svg)](https://ray.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive framework for benchmarking and comparing different data partitioning strategies for distributed machine learning inference.

## 🚀 **Quick Start**

```bash
# Install dependencies
pip install -r requirements.txt

# Test installation
python main.py --test

# Run all approaches
python main.py --approach all --workers 2 --samples 1000
```

## 📊 **Three Partitioning Strategies**

- **Uniform Partitioning**: PyTorch DDP with even data distribution
- **Dynamic Partitioning**: Ray with real-time load balancing  
- **Sharded Partitioning**: Multiprocessing with optimized shard distribution

## 🌊 **Streaming Benchmark System**

- **Comprehensive Testing**: 3 strategies × customizable node counts (1,2,4,8,16+)
- **CIFAR-10 Streams**: Configurable stream rates (1000+ samples/s)
- **Statistical Analysis**: Multiple iterations with averaging
- **Performance Metrics**: Throughput, CPU, memory, and scaling efficiency
- **Automatic Plotting**: Comprehensive visualizations generated automatically after benchmark completion

## 🎯 **Key Features**

- 🚀 **Comprehensive Benchmarking**: Matrix benchmarks across worker configurations
- 🔧 **Unified Configuration**: Standardized configuration system
- 🛡️ **Robust Error Handling**: Retry mechanisms and error recovery
- 📈 **Performance Monitoring**: Detailed metrics and visualization
- ☁️ **Cloud Support**: GCP deployment capabilities

## 📚 **Documentation**

**📖 [Complete Project Documentation](PROJECT_DOCUMENTATION.md)**

The comprehensive documentation includes:
- Detailed architecture overview
- Complete API reference
- Usage examples and tutorials
- Performance benchmarks
- Troubleshooting guide
- Contributing guidelines

## 🎮 **Usage Examples**

```bash
# Individual approaches
python main.py --approach uniform --workers 2 --samples 1000
python main.py --approach dynamic --workers 4 --samples 2000
python main.py --approach sharded --workers 2 --samples 1000

# Matrix benchmarking
python main.py --matrix --samples 1000

# Streaming benchmark with custom node counts (plots generated automatically)
python main.py --streaming-benchmark --nodes 1,2,4,8 --stream-rate 1000 --total-samples 8000

# Cloud deployment
python main.py --cloud-matrix --samples 10000

# Generate comprehensive plots manually (if needed)
python plot_streaming_results.py
```

## 📊 **Performance Results**

### **Streaming Benchmark Results**
| Strategy | 1 Node | 2 Nodes | 4 Nodes | Best For |
|----------|--------|---------|---------|----------|
| **Uniform** | 353.47 inf/s | 367.22 inf/s | 371.35 inf/s | **Best overall performance** |
| **Dynamic** | 119.26 inf/s | 186.09 inf/s | 265.19 inf/s | Load balancing |
| **Sharded** | 149.87 inf/s | 156.15 inf/s | 138.19 inf/s | Memory efficiency |

### **Scaling Efficiency**
- **Dynamic**: 0.67 (best scaling with load balancing)
- **Uniform**: 0.39 (consistent high performance)
- **Sharded**: 0.38 (stable across node counts)

## 🛠️ **Installation**

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Ray 2.2+
- Additional dependencies in `requirements.txt`

### Setup
```bash
# Clone repository
git clone <repository>
cd DataParallel

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/test_installation.py
```

## 📁 **Project Structure**

```
DataParallel/
├── main.py                          # Main entry point
├── plot_streaming_results.py        # Comprehensive plotting utility
├── src/
│   ├── approaches/                  # Partitioning implementations
│   ├── config.py                    # Unified configuration
│   ├── error_handling.py            # Error recovery
│   ├── utils.py                     # Shared utilities
│   ├── streaming_benchmark.py       # Streaming benchmark framework
│   └── matrix_benchmark.py          # Matrix benchmarking
├── cloud/                           # GCP deployment
├── scripts/                         # Setup and testing
├── data/                            # CIFAR-10 dataset
├── results/                         # Benchmark results and plots
└── Documentation files
```

## 🤝 **Contributing**

We welcome contributions! Please see the [Complete Documentation](PROJECT_DOCUMENTATION.md) for:
- Development setup
- Code standards
- Testing guidelines
- How to add new approaches

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 **Support**

- **Issues**: Create an issue on the repository
- **Documentation**: See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)
- **Questions**: Use GitHub Discussions

---

**DataParallel** - Empowering distributed machine learning with intelligent data partitioning strategies.
