"""
Benchmark CLI and programmatic entry point for the Tequila Rust/PyO3 (`signals` / `grassmann`)
quantum simulation backend:

    python -m tequila.simulators.benchmark_signals --quick
    python -m tequila.simulators.simulator_signals --quick
"""

from __future__ import annotations

from tequila.simulators.simulator_signals import (
    DEFAULT_CHEMISTRY_CONFIGS,
    DEFAULT_DENSE_CONFIGS,
    DEFAULT_LAMBDIFY_BACKEND,
    DEFAULT_MEMORY_WALL_QUBITS,
    QUICK_CHEMISTRY_CONFIGS,
    QUICK_DENSE_CONFIGS,
    QUICK_MEMORY_WALL_QUBITS,
    benchmark_dense_expectation_scaling,
    benchmark_dense_memory_wall_limit,
    benchmark_grassmann_plucker_permanent_vs_determinant,
    benchmark_plucker_hcb_chemistry_scaling,
    benchmark_vqe_optimization_and_wavefunction,
    build_arg_parser,
    build_dense_benchmark_circuit_and_hamiltonian,
    build_plucker_hcb_chemistry_circuit_and_hamiltonian,
    configure_logging,
    evaluate_sympy_with_lambdify,
    format_benchmark_markdown_tables,
    get_lambdified_gate_factors,
    logger,
    main,
    run_all_benchmarks,
)

__all__ = [
    "DEFAULT_CHEMISTRY_CONFIGS",
    "DEFAULT_DENSE_CONFIGS",
    "DEFAULT_LAMBDIFY_BACKEND",
    "DEFAULT_MEMORY_WALL_QUBITS",
    "QUICK_CHEMISTRY_CONFIGS",
    "QUICK_DENSE_CONFIGS",
    "QUICK_MEMORY_WALL_QUBITS",
    "benchmark_dense_expectation_scaling",
    "benchmark_dense_memory_wall_limit",
    "benchmark_grassmann_plucker_permanent_vs_determinant",
    "benchmark_plucker_hcb_chemistry_scaling",
    "benchmark_vqe_optimization_and_wavefunction",
    "build_arg_parser",
    "build_dense_benchmark_circuit_and_hamiltonian",
    "build_plucker_hcb_chemistry_circuit_and_hamiltonian",
    "configure_logging",
    "evaluate_sympy_with_lambdify",
    "format_benchmark_markdown_tables",
    "get_lambdified_gate_factors",
    "logger",
    "main",
    "run_all_benchmarks",
]

if __name__ == "__main__":
    raise SystemExit(main())
