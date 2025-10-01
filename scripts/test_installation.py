"""
Test script to verify the installation and basic functionality.

This script tests if all required packages are installed correctly
and if the basic functionality works.
"""
import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('ray', 'Ray'),
        ('dask', 'Dask'),
        ('dask.distributed', 'Dask Distributed'),
        ('psutil', 'PSUtil'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML')
    ]
    
    failed_imports = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        return False
    else:
        print("\nAll packages imported successfully!")
        return True

def test_basic_functionality():
    """Test basic functionality of the modules."""
    print("\nTesting basic functionality...")
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        # Test utils import
        from utils import get_system_info, MetricsCollector
        print("✓ Utils module imported")
        
        # Test system info
        system_info = get_system_info()
        print(f"✓ System info: {system_info['cpu_count']} CPUs, {system_info['memory_gb']:.1f} GB RAM")
        
        # Test metrics collector
        metrics = MetricsCollector()
        metrics.start_timing()
        metrics.record_cpu_usage()
        metrics.stop_timing()
        print("✓ Metrics collector working")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_dataset_loading():
    """Test if CIFAR-10 dataset can be loaded."""
    print("\nTesting dataset loading...")
    
    try:
        import torch
        import torchvision
        import torchvision.transforms as transforms
        
        # Test dataset loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # This will download the dataset if not present
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
        print(f"✓ CIFAR-10 dataset loaded: {len(dataset)} samples")
        
        # Test data loader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        batch = next(iter(dataloader))
        print(f"✓ Data loader working: batch shape {batch[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
        return False

def test_model_loading():
    """Test if the model can be loaded."""
    print("\nTesting model loading...")
    
    try:
        import torchvision
        
        # Test model loading
        model = torchvision.models.resnet18(pretrained=True)
        model.eval()
        print("✓ ResNet-18 model loaded")
        
        # Test inference
        import torch
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Model inference working: output shape {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False

def test_distributed_frameworks():
    """Test if distributed frameworks can be initialized."""
    print("\nTesting distributed frameworks...")
    
    # Test Ray
    try:
        import ray
        ray.init(ignore_reinit_error=True)
        print("✓ Ray initialized")
        ray.shutdown()
    except Exception as e:
        print(f"✗ Ray initialization failed: {e}")
        return False
    
    # Test Dask
    try:
        from dask.distributed import LocalCluster, Client
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, silence_logs=True)
        client = Client(cluster)
        print("✓ Dask cluster created")
        client.close()
        cluster.close()
    except Exception as e:
        print(f"✗ Dask initialization failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if the project structure is correct."""
    print("\nTesting project structure...")
    
    required_files = [
        'src/utils.py',
        'src/approaches/uniform_partitioning.py',
        'src/approaches/dynamic_partitioning.py',
        'src/approaches/sharded_partitioning.py',
        'src/benchmark.py',
        'local/run_local_experiments.py',
        'local/config.yaml',
        'cloud/setup_gcp_cluster.py',
        'cloud/run_cloud_experiments.py',
        'cloud/gcp_config.yaml',
        'cloud/cloud_config.yaml',
        'scripts/setup_local_venv.py',
        'examples/demo_all_approaches.py',
        'docs/QUICK_START.md',
        'main.py',
        'run.sh',
        'run.bat',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = Path(__file__).parent.parent / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    else:
        print("\nAll required files present!")
        return True

def main():
    """Main test function."""
    print("Data Partitioning Experiments - Installation Test")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Dataset Loading", test_dataset_loading),
        ("Model Loading", test_model_loading),
        ("Distributed Frameworks", test_distributed_frameworks),
        ("Project Structure", test_file_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nYou can now run experiments using:")
        print("  python main.py --demo")
        print("  python main.py --approach all")
        print("  ./run.sh --demo")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTry running:")
        print("  python main.py --setup")
        print("  ./run.sh --setup")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
