# Migration from pip to uv

This project has migrated from pip/virtualenv to **uv** for faster and more reliable dependency management.

## Quick Start

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

### 2. Migration Commands

| Old pip workflow | New uv workflow |
|------------------|-----------------|
| `python -m venv venv && source venv/bin/activate` | `uv venv` |
| `pip install -r requirements.txt` | `uv sync` |
| `pip install -e .` | `uv sync` |
| `pip install package` | `uv add package` |
| `python script.py` | `uv run script.py` |
| `pytest` | `uv run pytest` |

### 3. Choose Your Hardware Configuration

This project supports different hardware configurations via optional dependencies:

```bash
# For CPU-only (good for development)
uv sync --extra cpu

# For NVIDIA GPU with CUDA 12.8
uv sync --extra cu128

# For NVIDIA GPU with CUDA 12.6
uv sync --extra cu126

# For Intel XPU (Intel GPU/Gaudi)
uv sync --extra xpu

# For development with testing tools
uv sync --extra dev

# Combine multiple extras (e.g., CUDA + dev tools)
uv sync --extra cu128 --extra dev
```

### 4. Development Setup

```bash
# Clone and setup (replaces old pip install -e .)
git clone <repo-url>
cd <project>
uv sync --extra cpu --extra dev  # CPU with dev tools
```

### 5. Virtual Environment Management

```bash
# Create virtual environment (like python -m venv)
uv venv                    # creates .venv in current directory
uv venv my_env             # creates virtual environment with custom name
uv venv --python 3.12      # creates venv with specific Python version (requires >=3.12)

# Activate virtual environment (traditional way)
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Use without activation (recommended)
uv run python script.py   # automatically uses project venv
uv run pytest             # runs in project environment

# Install packages in active venv
uv pip install package    # when venv is activated
uv add package            # adds to project (recommended)

# Check venv location
uv venv --seed             # shows venv path and Python version
```

### 6. Running Commands

```bash
# Run tests
uv run pytest                              # searches from project root recursively
uv run pytest test_regr/                   # runs only tests in test_regr directory  
uv run pytest test_regr/example/conll04    # runs specific subfolder tests

# Run Python scripts
uv run python your_script.py

# Install new package
uv add numpy
uv add --group dev pytest  # for dev dependencies
```

## Hardware Configuration Notes

- **CPU**: Pure CPU implementation, good for development and testing
- **CU128**: NVIDIA CUDA 12.8 support for modern GPUs
- **CU126**: NVIDIA CUDA 12.6 support for slightly older GPUs  
- **XPU**: Intel GPU/Gaudi accelerator support
- **Dev**: Testing and development tools (pytest)

Note: Hardware extras are mutually exclusive - you can only install one at a time due to PyTorch compatibility constraints.

## Troubleshooting

**Q: Where's my virtual environment?**  
A: uv manages it automatically. Use `uv run` instead of activating manually.

**Q: How do I see installed packages?**  
A: `uv tree` or `uv pip list`

**Q: Can I still use pip?**  
A: Yes, but `uv add/remove` is recommended for consistency.

**Q: I get conflicts when installing multiple hardware extras**  
A: This is expected - choose only one hardware configuration (cpu, cu128, cu126, or xpu).