from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="n5styleea",
    version="15.3.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced AI-powered trading system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/N5StyleEA",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "torch>=1.10.0",
        "ccxt>=2.0.0",
        "ta>=0.10.0",
        "backtrader>=1.9.76.123",
        "fastapi>=0.75.0",
        "uvicorn>=0.17.0",
        "websockets>=10.0",
        "python-socketio>=5.5.0",
        "python-dotenv>=0.19.0",
        "pydantic>=1.9.0",
        "loguru>=0.6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
        ],
        "docs": [
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
            "sphinx-autoapi>=1.8.0",
            "sphinx-copybutton>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "n5styleea=main_v15_2:main",
        ],
    },
)