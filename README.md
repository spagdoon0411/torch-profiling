# torch-profiling

## Setup

Each member depends on `torch` behind two mutually exclusive extras. Pick the
one matching your hardware when syncing:

```sh
uv sync --extra apple   # Apple Silicon (plain PyPI wheel, MPS-enabled)
uv sync --extra cuda    # NVIDIA (e.g. 3090), via the pytorch-cu124 index
```
