"""
Top-level `python -m tequila` CLI entry point, delegating to the
Tequila Rust/PyO3 (`signals` / `grassmann`) benchmark and verification runner.
"""

from __future__ import annotations

from tequila.simulators.simulator_signals import main

if __name__ == "__main__":
    raise SystemExit(main())
