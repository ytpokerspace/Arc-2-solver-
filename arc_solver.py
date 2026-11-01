#!/usr/bin/env python3
"""Convenience entry point for the ARC-R2 solver.

Run from the repository root:
    python arc_solver.py --demo
    python arc_solver.py --dataset path/to/tasks.json --limit 20

The implementation remains in the tested arc_r2 package; this file provides
a simple, direct Python-script interface without duplicating solver logic.
"""
from __future__ import annotations

from arc_r2.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
