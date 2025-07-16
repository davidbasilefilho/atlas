#!/usr/bin/env python3
"""
Setup script for Atlas training environment
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        print(f"   Output: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"   ❌ Python {major}.{minor} is not supported. Please use Python 3.8+")
        return False
    print(f"   ✓ Python {major}.{minor} is compatible")
    return True


def install_torch():
    """Install PyTorch with appropriate CUDA support"""
    print("🔥 Installing PyTorch...")
    
    # Check if NVIDIA GPU is available
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        has_gpu = result.returncode == 0
    except:
        has_gpu = False
    
    if has_gpu:
        print("   🎮 NVIDIA GPU detected, installing CUDA version")
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        print("   💻 No GPU detected, installing CPU version")
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(torch_command, "Installing PyTorch")


def install_requirements():
    """Install other requirements"""
    requirements = [
        "transformers>=4.30.0",
        "datasets>=2.14.0", 
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "accelerate>=0.20.0",
        "tensorboard>=2.13.0",
        "einops>=0.7.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    print("📚 Installing other requirements...")
    for req in requirements:
        if not run_command(f"pip install '{req}'", f"Installing {req.split('>=')[0]}"):
            return False
    return True


def setup_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        "checkpoints",
        "logs", 
        "outputs",
        "plots",
        "data",
        ".cache"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✓ Created {dir_name}/")
    
    return True


def setup_wandb():
    """Setup Weights & Biases"""
    print("📊 Setting up Weights & Biases...")
    print("   ℹ️  You can login to wandb later with: wandb login")
    return True


def verify_installation():
    """Verify the installation works"""
    print("🔍 Verifying installation...")
    
    test_script = """
import torch
import transformers
import wandb
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ Transformers {transformers.__version__}")
print(f"✓ Wandb {wandb.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA devices: {torch.cuda.device_count()}")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print("   Installation verification:")
        for line in result.stdout.strip().split('\n'):
            print(f"   {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Verification failed: {e}")
        print(f"   Output: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False


def main():
    """Main setup function"""
    print("🚀 Atlas Model Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install PyTorch
    if not install_torch():
        print("❌ Failed to install PyTorch")
        sys.exit(1)
    
    # Install other requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Setup directories
    if not setup_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Setup wandb
    setup_wandb()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    print("\n✨ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("   1. Login to wandb: wandb login")
    print("   2. Run demo: python demo.py")
    print("   3. Start training: python main.py --mode train --debug")
    print("   4. For full training: python main.py --mode train")
    

if __name__ == "__main__":
    main()
