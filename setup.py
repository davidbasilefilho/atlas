#!/usr/bin/env python3
"""
Simple setup script for Atlas - use only for manual setup
For package installation, use: uv sync
"""

import os
import sys
from pathlib import Path


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


def main():
    """Main setup function - simplified"""
    print("🚀 Atlas Model Setup (Directories Only)")
    print("=" * 50)
    
    # Only create directories
    setup_directories()
    
    print("\n✨ Directory setup completed!")
    print("\n📋 To install dependencies:")
    print("   uv sync  # Install all dependencies from pyproject.toml")
    print("\n🎯 Next steps:")
    print("   1. Login to wandb: uv run wandb login (optional)")
    print("   2. Run demo: uv run python demo.py") 
    print("   3. Start training: uv run python main.py --mode train --debug")
    

if __name__ == "__main__":
    main()
