# BLA-Mem

MVP code lives in:

- Core (OOP + torch): `src/bla_mem/`
- Experiments (adding/parity): `experiments/mvp.py`

Quickstart (cloud):

1) Install:

- `pip install -r requirements.txt`
- `pip install -e .`

2) Run MVP:

- `python -m experiments.mvp --task adding --model bla --seq-len 4096 --chunk 64 --depth 3 --steps 2000`
- `python -m experiments.mvp --task parity --model bla --seq-len 4096 --chunk 64 --depth 3 --steps 2000`

Notes:

- `signatory` installation is environment-dependent (torch/cuda matching). If it fails, the code will raise a clear error when trying to compute signatures.