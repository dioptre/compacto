from setuptools import setup, find_packages

setup(
    name="compactifai-granite",
    version="0.1.0",
    description="CompactifAI implementation for IBM Granite models using quantum-inspired tensor networks",
    author="Implementation Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "tensorly>=0.8.0",
        "tensornetwork>=0.4.6",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
    ],
    python_requires=">=3.8",
)