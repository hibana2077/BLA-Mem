# BLA-Mem

Toy baselines live in:

- Core (OOP + torch + CLI): `src/bla_mem/`

## Quickstart

1) Install:

- `pip install -r requirements.txt`
- `pip install -e .`

2) Train (no tqdm; eval every 200 steps by default):

- `python -m bla_mem.cli train --task parity --model transformer --train-len 128 --test-lens 128 256 512 --steps 2000`
- `python -m bla_mem.cli train --task adding --model sheaf --train-len 128 --test-lens 128 256 512 --steps 2000`

Models:

- `transformer` (baseline)
- `sheaf` (sheaf-gluing)
- `ultrametric` (hierarchical heat-flow)
- `magnus` (Magnus/commutator evolution)
- `cumulant` (Pre-Lie / cumulant scan)
- `jump` (Dirichlet-form jump generator)