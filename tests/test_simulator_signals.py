"""
Tequila test suite for the Rust/PyO3 (`signals` / `grassmann`) simulator backend
and CLI benchmark runner (`python -m tequila.simulators.simulator_signals` and
`python -m tequila.simulators.benchmark_signals`).
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import tequila as tq
from tequila.simulators import benchmark_signals, simulator_signals


def test_simulator_signals_gates_expectation_and_vqe():
    U = (
        tq.gates.H(target=0)
        + tq.gates.Ry(angle="a", target=1)
        + tq.gates.CNOT(control=0, target=1)
        + tq.gates.ExpPauli(angle="b", paulistring="X(0)Y(1)")
    )
    H = tq.QubitHamiltonian("0.7*Z(0)Z(1) - 0.3*X(0)X(1)")
    E = tq.ExpectationValue(H=H, U=U)
    vars_d = {"a": 0.45, "b": -0.35}

    e_sig = float(tq.simulate(E, variables=vars_d, backend="signals"))
    e_sym = float(tq.simulate(E, variables=vars_d, backend="symbolic"))
    assert np.isclose(e_sig, e_sym, atol=1e-13)


def test_simulator_signals_benchmarks_and_cli_main(tmp_path):
    res = simulator_signals.run_all_benchmarks(
        suite="quick",
        backends=["signals"],
        qubits=[4, 6],
        chemistry_configs=[(2, 6, 16), (8, 126, 16)],
        repeats=1,
    )
    assert "dense_expectation" in res
    assert "plucker_hcb_chemistry" in res
    assert res["plucker_hcb_chemistry"][-1]["n_qubits"] == 126
    assert res["plucker_hcb_chemistry"][-1]["active_support_size"] == 256

    json_out = tmp_path / "tequila_bench.json"
    md_out = tmp_path / "tequila_bench.md"
    rc = benchmark_signals.main(
        [
            "--suite",
            "chemistry",
            "--backends",
            "signals",
            "--chemistry-configs",
            "2:6:16,8:126:16",
            "--repeats",
            "1",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(md_out),
            "--quiet",
        ]
    )
    assert rc == 0
    assert json_out.exists() and md_out.exists()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(data["plucker_hcb_chemistry"]) == 2

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tequila.simulators.benchmark_signals",
            "-v",
            "--log-level=DEBUG",
            "--lambdify-backend=math",
            "--suite",
            "plucker",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    assert "INFO [tequila.simulators.simulator_signals]" in proc.stdout
    assert "DEBUG [tequila.simulators.simulator_signals]" in proc.stdout
    assert "lambdify_backend=math" in proc.stdout
    assert "Table 5:" in proc.stdout
