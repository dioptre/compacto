#!/usr/bin/env python3
"""
Installation script for CompactifAI dependencies.
Installs required packages without conflicts.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {cmd}")
        print(f"Error: {e.stderr}")
        return False

def install_dependencies():
    """Install dependencies step by step."""
    print("CompactifAI Installation Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install core dependencies
    dependencies = [
        "torch>=2.0.0",
        "numpy>=1.21.0", 
        "tqdm>=4.64.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"python3 -m pip install {dep}"):
            print(f"Failed to install {dep}")
            return False
    
    print("\n✅ Core dependencies installed successfully!")
    
    # Try to install optional dependencies
    optional_deps = [
        "transformers>=4.35.0",
        "tensorly>=0.8.0",
        "accelerate>=0.20.0",
        "datasets>=2.14.0"
    ]
    
    print("\nInstalling optional dependencies (may fail, that's ok)...")
    for dep in optional_deps:
        run_command(f"python3 -m pip install {dep}")
    
    print("\n🎉 Installation completed!")
    print("\nTo test the installation, run:")
    print("  python3 quick_test.py")
    
    return True

if __name__ == "__main__":
    install_dependencies()