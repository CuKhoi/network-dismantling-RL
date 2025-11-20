# network-dismantling-RL

This repository contains several Cython/TensorFlow implementations of the FINDER reinforcement-learning approach for network dismantling.

## Modern GPU support

The original code targeted TensorFlow 1.14, which does not support newer GPUs such as the RTX 4090. All TensorFlow imports now use the TensorFlow 2 `compat.v1` API with `disable_v2_behavior()` so the existing session/placeholder code can run on TensorFlow 2.13 (compatible with CUDA 11.8 and newer drivers). Install dependencies with:

```bash
pip install -r FINDER-ALTHORIGITHMS/requirements.txt
```

Ensure your system provides CUDA 11.8-compatible libraries (the unified `tensorflow` 2.13 wheel expects them) before running training or evaluation.

### Recommended Python/Conda setup

TensorFlow 2.13 supports Python 3.8–3.11; the updated dependency pins work reliably on Python 3.10. To create a conda environment that matches the tested setup:

```bash
conda create -n finder-rl python=3.10 -y
conda activate finder-rl
pip install -r FINDER-ALTHORIGITHMS/requirements.txt
```
