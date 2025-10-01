"""
Local setup script with virtual environment creation.

This script creates a virtual environment, installs dependencies,
and sets up the local environment for data partitioning experiments.
"""
import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_virtual_environment(venv_path: str = "venv"):
    """Create a virtual environment."""
    logger.info(f"Creating virtual environment at {venv_path}...")
    
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        logger.info(f"Virtual environment created at {venv_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        return False

def get_python_executable(venv_path: str = "venv"):
    """Get the Python executable path for the virtual environment."""
    if os.name == 'nt':  # Windows
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:  # Unix/Linux/macOS
        return os.path.join(venv_path, "bin", "python")

def get_pip_executable(venv_path: str = "venv"):
    """Get the pip executable path for the virtual environment."""
    if os.name == 'nt':  # Windows
        return os.path.join(venv_path, "Scripts", "pip.exe")
    else:  # Unix/Linux/macOS
        return os.path.join(venv_path, "bin", "pip")

def install_requirements(venv_path: str = "venv"):
    """Install requirements in the virtual environment."""
    logger.info("Installing requirements in virtual environment...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        logger.error(f"Requirements file not found: {requirements_file}")
        return False
    
    pip_executable = get_pip_executable(venv_path)
    
    try:
        # Upgrade pip first
        subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_executable, "install", "-r", str(requirements_file)], check=True)
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    base_dir = Path(__file__).parent
    
    directories = [
        "data",
        "results",
        "logs",
        "plots"
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def test_installation(venv_path: str = "venv"):
    """Test if installation is working."""
    logger.info("Testing installation...")
    
    python_executable = get_python_executable(venv_path)
    
    packages = [
        "torch",
        "torchvision",
        "ray",
        "dask",
        "psutil",
        "matplotlib",
        "pandas",
        "numpy"
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            result = subprocess.run([
                python_executable, "-c", f"import {package}; print('✓ {package}')"
            ], capture_output=True, text=True, check=True)
            print(result.stdout.strip())
        except subprocess.CalledProcessError:
            print(f"✗ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        logger.error(f"Failed to import: {failed_imports}")
        return False
    
    logger.info("All packages imported successfully!")
    return True

def download_dataset(venv_path: str = "venv"):
    """Download CIFAR-10 dataset."""
    logger.info("Downloading CIFAR-10 dataset...")
    
    python_executable = get_python_executable(venv_path)
    
    download_script = """
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"Dataset downloaded: {len(dataset)} samples")
"""
    
    try:
        result = subprocess.run([
            python_executable, "-c", download_script
        ], capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        logger.info("Dataset downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download dataset: {e}")
        return False

def create_activation_script(venv_path: str = "venv"):
    """Create activation script for easy environment activation."""
    base_dir = Path(__file__).parent
    
    if os.name == 'nt':  # Windows
        activate_script = f"""@echo off
echo Activating virtual environment...
call {venv_path}\\Scripts\\activate.bat
echo Virtual environment activated!
echo.
echo You can now run experiments:
echo   python local/run_local_experiments.py
echo   python example_usage.py
echo   python test_installation.py
echo.
cmd /k
"""
        script_path = base_dir / "activate_venv.bat"
    else:  # Unix/Linux/macOS
        activate_script = f"""#!/bin/bash
echo "Activating virtual environment..."
source {venv_path}/bin/activate
echo "Virtual environment activated!"
echo ""
echo "You can now run experiments:"
echo "  python local/run_local_experiments.py"
echo "  python example_usage.py"
echo "  python test_installation.py"
echo ""
exec bash
"""
        script_path = base_dir / "activate_venv.sh"
    
    with open(script_path, 'w') as f:
        f.write(activate_script)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(script_path, 0o755)
    
    logger.info(f"Activation script created: {script_path}")

def run_quick_test(venv_path: str = "venv"):
    """Run a quick test to verify everything is working."""
    logger.info("Running quick test...")
    
    python_executable = get_python_executable(venv_path)
    
    test_script = """
import sys
sys.path.append('src')

from utils import get_system_info, MetricsCollector
import torch
import torchvision

print("Testing basic functionality...")

# Test system info
system_info = get_system_info()
print(f"System: {system_info['cpu_count']} CPUs, {system_info['memory_gb']:.1f} GB RAM")

# Test metrics collector
metrics = MetricsCollector()
metrics.start_timing()
metrics.record_cpu_usage()
metrics.stop_timing()
print("Metrics collector: OK")

# Test model loading
model = torchvision.models.resnet18(pretrained=True)
model.eval()
print("Model loading: OK")

# Test inference
dummy_input = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    output = model(dummy_input)
print(f"Inference test: OK (output shape: {output.shape})")

print("\\n✅ All tests passed! Ready to run experiments.")
"""
    
    try:
        result = subprocess.run([
            python_executable, "-c", test_script
        ], capture_output=True, text=True, check=True)
        print(result.stdout)
        logger.info("Quick test completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Quick test failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup local environment with virtual environment')
    parser.add_argument('--venv-path', type=str, default='venv', help='Virtual environment path')
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset download')
    parser.add_argument('--skip-test', action='store_true', help='Skip quick test')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        print("🚀 Setting up Data Partitioning Experiments with Virtual Environment")
        print("="*70)
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            sys.exit(1)
        
        print(f"Python version: {sys.version}")
        
        # Create virtual environment
        if not create_virtual_environment(args.venv_path):
            sys.exit(1)
        
        # Create directories
        create_directories()
        
        # Install requirements
        if not install_requirements(args.venv_path):
            sys.exit(1)
        
        # Test installation
        if not test_installation(args.venv_path):
            sys.exit(1)
        
        # Download dataset
        if not args.skip_dataset:
            if not download_dataset(args.venv_path):
                logger.warning("Dataset download failed, but continuing...")
        
        # Create activation script
        create_activation_script(args.venv_path)
        
        # Run quick test
        if not args.skip_test:
            if not run_quick_test(args.venv_path):
                logger.warning("Quick test failed, but setup completed")
        
        print("\n" + "="*70)
        print("🎉 Setup completed successfully!")
        print("="*70)
        print(f"Virtual environment created at: {args.venv_path}")
        print("\nTo activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print(f"  {args.venv_path}\\Scripts\\activate")
            print("  or run: activate_venv.bat")
        else:  # Unix/Linux/macOS
            print(f"  source {args.venv_path}/bin/activate")
            print("  or run: ./activate_venv.sh")
        
        print("\nTo run experiments:")
        print("  python local/run_local_experiments.py")
        print("  python example_usage.py")
        print("  python test_installation.py")
        
        print("\nTo test individual approaches:")
        print("  python local/run_local_experiments.py --approach uniform")
        print("  python local/run_local_experiments.py --approach dynamic")
        print("  python local/run_local_experiments.py --approach sharded")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
