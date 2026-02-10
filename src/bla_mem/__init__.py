"""BLA-Mem core package.

This package contains a minimal, torch-first MVP implementation of an
associative memory based on truncated tensor-algebra exp/log and scan.

Cloud note: signature computation uses the optional `signatory` package.
"""

from .model import BLAMem  # noqa: F401
