"""
Rust/PyO3 (`signals_numerical_adapter`) Grassmannian, Plabic VRC, Madelung/DDF,
and Quantum Causal Inference (`QCI`) Quantum Simulation Backend for Tequila.

Implements:
- `BackendCircuitSignals`: Native Rust statevector, Grassmannian Plücker HCB sector,
  shot sampler, and noisy Kraus + McWeeny cubic-purification density-matrix simulator.
- `BackendExpectationValueSignals`: Rayon-parallelized Rust Pauli Hamiltonian expectation engine.
- `BackendBraKetSignals`: Native Rust overlap `<bra | ket>` and transition matrix element `<bra | O | ket>`.
"""

from __future__ import annotations

import logging
import sys
import typing
from typing import Union

import numpy as np

from tequila import BitNumbering, BitString, TequilaException
from tequila.circuit._gates_impl import (
    ExponentialPauliGateImpl,
    PhaseGateImpl,
    QGateImpl,
    RotationGateImpl,
)
from tequila.circuit.gates import QubitExcitationImpl
from tequila.simulators.simulator_base import (
    BackendBraKet,
    BackendCircuit,
    BackendExpectationValue,
)
from tequila.simulators.simulator_symbolic import (
    BackendCircuitSymbolic,
    DEFAULT_LAMBDIFY_BACKEND,
    evaluate_sympy_with_lambdify,
    get_lambdified_gate_factors,
)
from tequila.utils.bitstrings import reverse_int_bits
from tequila.utils.keymap import KeyMapRegisterToSubregister
from tequila.wavefunction.density_matrix import DensityMatrix
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction

import sympy
import signals_numerical_adapter as sna

logger = logging.getLogger("tequila.simulators.simulator_signals")


def configure_logging(
    log_level: str | int | None = None,
    verbose: bool = False,
    quiet: bool = False,
    stream=None,
) -> logging.Logger:
    """
    Configures `logger` to emit `DEBUG` and `INFO` (and higher) log records to
    `sys.stdout` (or `stream`) according to `--log-level`, `-v`/`--verbose`, and `-q`/`--quiet`.
    """
    if log_level is not None:
        if isinstance(log_level, int):
            level = log_level
        else:
            s = str(log_level).strip().upper()
            level = int(s) if s.isdigit() else getattr(logging, s, logging.INFO)
    elif verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    target_stream = stream if stream is not None else sys.stdout
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(target_stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    return logger


class TequilaSignalsException(TequilaException):
    def __str__(self) -> str:
        return "Error in signals Rust/PyO3 backend: " + self.message


class BackendCircuitSignals(BackendCircuit):
    """
    Compiled Tequila circuit backed by the native Rust/PyO3 `signals_numerical_adapter`
    statevector, Grassmannian Plücker $\\mathrm{Gr}_{\\mathbb{C}}(k,n)$, Complex Plabic VRC,
    and Quantum Causal Inference (`QCI`) engine.
    """

    compiler_arguments = {
        "trotterized": True,
        "swap": False,
        "multitarget": True,
        "controlled_rotation": False,
        "generalized_rotation": True,
        "exponential_pauli": False,
        "controlled_exponential_pauli": False,
        "phase": False,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": False,
        "toffoli": False,
        "phase_to_z": False,
        "cc_max": False,
    }

    numbering = BitNumbering.MSB
    supports_sampling_initialization = True
    supports_generic_initialization = True
    lambdify_backend: str = DEFAULT_LAMBDIFY_BACKEND

    def __init__(
        self,
        abstract_circuit,
        variables=None,
        qubit_map=None,
        noise=None,
        device=None,
        mcweeny_passes: int = 0,
        seed: int = 1337,
        lambdify_backend: str | None = None,
        *args,
        **kwargs,
    ):
        self.measurements = None
        self.mcweeny_passes = int(mcweeny_passes)
        self.seed = int(seed)
        self.lambdify_backend = str(
            lambdify_backend or self.__class__.lambdify_backend or DEFAULT_LAMBDIFY_BACKEND
        )
        self._rust_circuit = None
        self._param_callables: list = []
        self._current_angles: list[float] = []
        super().__init__(
            abstract_circuit=abstract_circuit,
            variables=variables,
            qubit_map=qubit_map,
            noise=noise,
            device=device,
            *args,
            **kwargs,
        )

    @classmethod
    def set_lambdify_backend(cls, lambdify_backend: str = DEFAULT_LAMBDIFY_BACKEND) -> str:
        cls.lambdify_backend = str(lambdify_backend or DEFAULT_LAMBDIFY_BACKEND)
        BackendCircuitSymbolic.set_lambdify_backend(cls.lambdify_backend)
        return cls.lambdify_backend

    def initialize_circuit(self, *args, **kwargs) -> list[dict]:
        return []

    def create_circuit(self, abstract_circuit=None, circuit=None, variables=None, *args, **kwargs):
        if abstract_circuit is None:
            abstract_circuit = self.abstract_circuit

        is_primary = circuit is None and abstract_circuit is self.abstract_circuit
        if is_primary:
            self._rust_circuit = sna.RustFastCircuit(int(self.n_qubits))
            self._param_callables = []
            self._current_angles = []

        result = circuit if circuit is not None else self.initialize_circuit(*args, **kwargs)
        for g in abstract_circuit.gates:
            self._compile_single_gate(
                g,
                result,
                variables=variables,
                rust_circuit=self._rust_circuit if is_primary else None,
            )
        return result

    def update_variables(self, variables) -> None:
        if variables is None:
            return
        if self._rust_circuit is not None:
            self._current_angles = [
                self._resolve_angle(fn, variables) for fn in self._param_callables
            ]
        else:
            self.circuit = self.create_circuit(
                abstract_circuit=self.abstract_circuit, variables=variables
            )

    def _resolve_angle(self, param, variables) -> float:
        if isinstance(param, (int, float, np.number)):
            return float(param)
        if isinstance(param, sympy.Basic):
            return float(
                evaluate_sympy_with_lambdify(
                    param,
                    variables=variables,
                    lambdify_backend=self.lambdify_backend,
                )
            )
        if callable(param):
            raw = param(variables)
            if isinstance(raw, sympy.Basic):
                return float(
                    evaluate_sympy_with_lambdify(
                        raw,
                        variables=variables,
                        lambdify_backend=self.lambdify_backend,
                    )
                )
            return float(raw)
        if variables is not None and param in variables:
            val = variables[param]
            if isinstance(val, sympy.Basic):
                return float(
                    evaluate_sympy_with_lambdify(
                        val,
                        variables=variables,
                        lambdify_backend=self.lambdify_backend,
                    )
                )
            return float(val)
        return float(param)

    def _register_angle(self, param, variables, rust_circuit) -> tuple[float, int | None]:
        val = self._resolve_angle(param, variables)
        if rust_circuit is None:
            return val, None
        if isinstance(param, (int, float, np.number)):
            return val, None
        idx = len(self._param_callables)
        self._param_callables.append(param)
        self._current_angles.append(val)
        return val, idx

    def _mapped_targets(self, gate) -> list[int]:
        return [int(self.qubit(t)) for t in gate.target]

    def _mapped_controls(self, gate) -> list[int]:
        if not gate.is_controlled():
            return []
        return [int(self.qubit(c)) for c in gate.control]

    def _compile_single_gate(
        self, gate, circuit: list[dict], variables=None, rust_circuit=None
    ) -> None:
        if isinstance(gate, QubitExcitationImpl):
            compiled = gate.compile(exponential_pauli=True)
            for sub in compiled.abstract_circuit.gates:
                self._compile_single_gate(
                    sub, circuit, variables=variables, rust_circuit=rust_circuit
                )
            return

        controls = self._mapped_controls(gate)
        if isinstance(gate, ExponentialPauliGateImpl):
            angle, param_idx = self._register_angle(
                gate.parameter, variables, rust_circuit
            )
            paulis = [
                (int(self.qubit(q)), str(op).upper())
                for q, op in gate.paulistring.items()
            ]
            if rust_circuit is not None:
                rust_circuit.add_exp_pauli(paulis, controls, angle, param_idx)
            circuit.append(
                {
                    "kind": "exp_pauli",
                    "paulis": paulis,
                    "angle": angle,
                    "controls": controls,
                }
            )
            return

        targets = self._mapped_targets(gate)
        if isinstance(gate, RotationGateImpl):
            angle, param_idx = self._register_angle(
                gate.parameter, variables, rust_circuit
            )
            axis_upper = str(gate.name).upper()
            if rust_circuit is not None:
                rust_circuit.add_rotation(
                    axis_upper, targets, controls, angle, param_idx
                )
            circuit.append(
                {
                    "kind": "rotation",
                    "axis": axis_upper,
                    "angle": angle,
                    "targets": targets,
                    "controls": controls,
                }
            )
            return

        if isinstance(gate, PhaseGateImpl):
            angle, param_idx = self._register_angle(
                gate.parameter, variables, rust_circuit
            )
            if rust_circuit is not None:
                rust_circuit.add_rotation("PHASE", targets, controls, angle, param_idx)
            circuit.append(
                {
                    "kind": "rotation",
                    "axis": "PHASE",
                    "angle": angle,
                    "targets": targets,
                    "controls": controls,
                }
            )
            return

        name_upper = str(gate.name).upper()
        if name_upper == "SWAP":
            if rust_circuit is not None and len(targets) >= 2:
                rust_circuit.add_swap(targets[0], targets[1], controls)
            circuit.append(
                {
                    "kind": "swap",
                    "targets": targets,
                    "controls": controls,
                }
            )
        else:
            if rust_circuit is not None:
                rust_circuit.add_basic(name_upper, targets, controls)
            circuit.append(
                {
                    "kind": "basic",
                    "name": name_upper,
                    "targets": targets,
                    "controls": controls,
                }
            )

    def add_basic_gate(self, gate, circuit: list[dict], *args, **kwargs) -> None:
        self._compile_single_gate(
            gate, circuit, variables=kwargs.get("variables", {}), rust_circuit=None
        )

    def add_parametrized_gate(self, gate, circuit: list[dict], *args, **kwargs) -> None:
        self._compile_single_gate(
            gate, circuit, variables=kwargs.get("variables", {}), rust_circuit=None
        )

    def add_measurement(self, circuit, target_qubits, *args, **kwargs):
        self.measurements = sorted(target_qubits)
        return circuit

    def _build_initial_state_re_im(
        self, initial_state: Union[int, QubitWaveFunction] = 0
    ) -> list[tuple[float, float]]:
        dim = 1 << self.n_qubits
        if isinstance(initial_state, QubitWaveFunction):
            arr = initial_state.to_array(out_numbering=BitNumbering.MSB)
            return [(float(np.real(c)), float(np.imag(c))) for c in arr]
        if isinstance(initial_state, BitString):
            idx = (
                reverse_int_bits(initial_state.integer, initial_state.nbits)
                if initial_state.numbering != BitNumbering.MSB
                else int(initial_state.integer)
            )
        else:
            idx = int(initial_state)
        if idx == 0:
            return []
        init = [(0.0, 0.0)] * dim
        init[idx] = (1.0, 0.0)
        return init

    def _resolve_initial_state_fast(
        self, initial_state: Union[int, QubitWaveFunction] = 0
    ) -> tuple[int, list[float] | None]:
        if isinstance(initial_state, QubitWaveFunction):
            arr = initial_state.to_array(out_numbering=BitNumbering.MSB)
            flat = np.ascontiguousarray(arr, dtype=np.complex128).view(np.float64)
            return 0, flat.tolist()
        if isinstance(initial_state, BitString):
            idx = (
                reverse_int_bits(initial_state.integer, initial_state.nbits)
                if initial_state.numbering != BitNumbering.MSB
                else int(initial_state.integer)
            )
            return int(idx), None
        return int(initial_state), None

    def simulate_statevector_re_im(
        self,
        variables=None,
        initial_state: Union[int, QubitWaveFunction] = 0,
        circuit_override: list[dict] | None = None,
    ) -> list[tuple[float, float]]:
        self.update_variables(variables)
        if circuit_override is None and self._rust_circuit is not None:
            init_idx, init_flat = self._resolve_initial_state_fast(initial_state)
            flat = self._rust_circuit.simulate_flat_f64(
                self._current_angles, init_idx, init_flat
            )
            it = iter(flat)
            return list(zip(it, it))
        ops = circuit_override if circuit_override is not None else self.circuit
        init_re_im = self._build_initial_state_re_im(initial_state)
        return sna.py_simulate_circuit_msb(
            int(self.n_qubits),
            ops,
            init_re_im if init_re_im else None,
        )

    def do_simulate(
        self, variables, initial_state: Union[int, QubitWaveFunction] = 0, *args, **kwargs
    ) -> QubitWaveFunction:
        self.update_variables(variables)
        if self._rust_circuit is not None:
            init_idx, init_flat = self._resolve_initial_state_fast(initial_state)
            if self.n_qubits > 20 and init_flat is None:
                sparse_triples = self._rust_circuit.simulate_sparse_plucker(
                    self._current_angles, init_idx
                )
                if len(sparse_triples) <= 262144:
                    wfn = QubitWaveFunction(
                        int(self.n_qubits), numbering=self.numbering, dense=False
                    )
                    wfn._state = {
                        int(idx): complex(re, im)
                        for idx, re, im in sparse_triples
                    }
                    return wfn
            flat = self._rust_circuit.simulate_flat_f64(
                self._current_angles, init_idx, init_flat
            )
            arr = (
                np.asarray(flat, dtype=np.float64)
                .view(np.complex128)
                .copy()
            )
            return QubitWaveFunction.from_array(
                array=arr, numbering=self.numbering, copy=False
            )
        re_im = self.simulate_statevector_re_im(
            variables=variables, initial_state=initial_state
        )
        arr = np.array([complex(r, i) for r, i in re_im], dtype=complex)
        return QubitWaveFunction.from_array(array=arr, numbering=self.numbering)

    def do_sample(
        self,
        samples: int,
        circuit,
        noise_model=None,
        initial_state: Union[int, QubitWaveFunction] = 0,
        *args,
        **kwargs,
    ) -> QubitWaveFunction:
        override = None if circuit is self.circuit else circuit
        if noise_model is not None or self.noise is not None:
            active_noise = noise_model if noise_model is not None else self.noise
            dm = self.do_simulate_density(
                variables=kwargs.get("variables", None),
                initial_state=initial_state,
                noise_model=active_noise,
                circuit_override=override,
            )
            diag_probs = np.clip(np.real(np.diag(dm.get_density_as_array())), 0.0, 1.0)
            # DensityMatrix defaults to LSB; convert diagonal probabilities to MSB amplitudes
            msb_re_im = [
                (
                    float(
                        np.sqrt(
                            diag_probs[reverse_int_bits(idx, int(self.n_qubits))]
                        )
                    ),
                    0.0,
                )
                for idx in range(1 << self.n_qubits)
            ]
            sampled_pairs = sna.py_sample_statevector_msb(
                msb_re_im, int(samples), int(self.seed)
            )
        else:
            re_im = self.simulate_statevector_re_im(
                variables=kwargs.get("variables", None),
                initial_state=initial_state,
                circuit_override=override,
            )
            sampled_pairs = sna.py_sample_statevector_msb(
                re_im, int(samples), int(self.seed)
            )

        result = QubitWaveFunction(self.n_qubits, self.numbering)
        for basis_idx, count in sampled_pairs:
            result[int(basis_idx)] = complex(float(count), 0.0)

        if self.measurements is not None and len(self.measurements) < self.n_qubits:
            mapped_target = [self.qubit_map[q].number for q in self.measurements]
            mapped_full = [self.qubit_map[q].number for q in self.abstract_qubits]
            keymap = KeyMapRegisterToSubregister(
                subregister=mapped_target, register=mapped_full
            )
            result = QubitWaveFunction.from_wavefunction(
                result, keymap, n_qubits=len(self.measurements)
            )

        return result

    def do_simulate_density(
        self,
        variables=None,
        initial_state: Union[int, QubitWaveFunction] = 0,
        noise_model=None,
        circuit_override: list[dict] | None = None,
        mcweeny_passes: int | None = None,
        *args,
        **kwargs,
    ) -> DensityMatrix:
        self.update_variables(variables)
        ops = circuit_override if circuit_override is not None else self.circuit
        active_noise = noise_model if noise_model is not None else self.noise
        noise_channels: list[tuple[str, int, float]] = []
        if active_noise is not None and hasattr(active_noise, "noises"):
            for n in active_noise.noises:
                prob = float(n.probs[0]) if len(n.probs) > 0 else 0.0
                for q in range(int(self.n_qubits)):
                    noise_channels.append((str(n.name), int(q), prob))

        passes = (
            int(mcweeny_passes)
            if mcweeny_passes is not None
            else int(self.mcweeny_passes)
        )
        rho_re_im = sna.py_simulate_noisy_density_msb(
            int(self.n_qubits), ops, noise_channels, passes
        )
        dim = 1 << int(self.n_qubits)
        rho_msb = np.array([complex(r, i) for r, i in rho_re_im], dtype=complex).reshape(
            (dim, dim)
        )
        return DensityMatrix.from_array(
            density_matrix=rho_msb,
            n_qubits=int(self.n_qubits),
            numbering=BitNumbering.MSB,
        )

    def analyze_grassmann_qci(
        self,
        variables=None,
        initial_state: Union[int, QubitWaveFunction] = 0,
        soft_evidence_q: float = 0.80,
        prior_p: float = 0.50,
    ) -> dict[str, float]:
        """
        Extracts the Complex Grassmannian Plücker $\\mathrm{Gr}_{\\mathbb{C}}(2,4)$,
        Homogeneous Complex VRC, Madelung/Bohm quantum pressure, QCI spin mutual
        information, and Quantum Jeffrey Plücker kinematic diagnostics in Rust.
        """
        re_im = self.simulate_statevector_re_im(
            variables=variables, initial_state=initial_state
        )
        return dict(
            sna.py_analyze_grassmann_qci_state(
                int(self.n_qubits), re_im, float(soft_evidence_q), float(prior_p)
            )
        )


class BackendExpectationValueSignals(BackendExpectationValue):
    """
    Rayon-parallelized Rust/PyO3 ExpectationValue evaluator for Tequila.
    """

    BackendCircuitType = BackendCircuitSignals
    use_mapping = True

    def initialize_hamiltonian(self, hamiltonians: tuple) -> tuple:
        converted = []
        raw_hams = []
        for H in hamiltonians:
            terms = []
            for ps in H.paulistrings:
                coeff = complex(ps.coeff)
                ops = [
                    (int(self.U.qubit(q)), str(op).upper())
                    for q, op in ps.items()
                ]
                terms.append((float(np.real(coeff)), float(np.imag(coeff)), ops))
            converted.append((H, terms))
            raw_hams.append(terms)
        self._rust_ham_set = sna.RustFastHamiltonianSet(
            int(self.n_qubits), raw_hams
        )
        return tuple(converted)

    def simulate(
        self, variables, initial_state: Union[int, QubitWaveFunction] = 0, *args, **kwargs
    ) -> np.ndarray:
        self.U.update_variables(variables)
        if self.U._rust_circuit is not None and hasattr(self, "_rust_ham_set"):
            init_idx, init_flat = self.U._resolve_initial_state_fast(initial_state)
            vals = self.U._rust_circuit.expectation_compiled(
                self.U._current_angles,
                init_idx,
                init_flat,
                self._rust_ham_set,
            )
            return np.asarray(vals, dtype=float)

        ket_re_im = self.U.simulate_statevector_re_im(
            variables=variables, initial_state=initial_state
        )
        results = []
        for _H_orig, rust_terms in self.H:
            re_val, _im_val = sna.py_expectation_pauli_hamiltonian_msb(
                int(self.n_qubits), ket_re_im, ket_re_im, rust_terms
            )
            results.append(float(re_val))
        return np.asarray(results, dtype=float)


class BackendBraKetSignals(BackendBraKet):
    """
    Native Rust/PyO3 BraKet overlap `<bra | ket>` and transition element `<bra | O | ket>` evaluator.
    """

    BackendCircuitType = BackendCircuitSignals

    def simulate(
        self, variables, initial_state: Union[int, QubitWaveFunction] = 0, *args, **kwargs
    ) -> complex:
        self.update_variables(variables)
        ket_re_im = self.U_ket.simulate_statevector_re_im(
            variables=variables, initial_state=initial_state
        )
        if self.U_bra is self.U_ket:
            bra_re_im = ket_re_im
        else:
            bra_re_im = self.U_bra.simulate_statevector_re_im(
                variables=variables, initial_state=initial_state
            )

        if self.operator is None:
            re_val, im_val = sna.py_expectation_pauli_hamiltonian_msb(
                int(self.n_qubits),
                bra_re_im,
                ket_re_im,
                [(1.0, 0.0, [])],
            )
            return complex(re_val, im_val)

        rust_terms = []
        for ps in self.operator.paulistrings:
            coeff = complex(ps.coeff)
            ops = [
                (int(self.U_ket.qubit(q)), str(op).upper())
                for q, op in ps.items()
            ]
            rust_terms.append((float(np.real(coeff)), float(np.imag(coeff)), ops))

        re_val, im_val = sna.py_expectation_pauli_hamiltonian_msb(
            int(self.n_qubits), bra_re_im, ket_re_im, rust_terms
        )
        return complex(re_val, im_val)


# ==============================================================================
# Reproducible Multi-Backend & High-n Benchmark Suite + CLI (`main` / `__main__`)
# ==============================================================================

DEFAULT_DENSE_CONFIGS: list[tuple[int, int, int]] = [
    (4, 2, 16),
    (6, 3, 30),
    (8, 3, 40),
    (10, 4, 60),
    (12, 4, 100),
    (14, 4, 120),
    (16, 4, 120),
    (18, 4, 120),
    (20, 4, 120),
    (22, 4, 120),
    (24, 4, 120),
]

QUICK_DENSE_CONFIGS: list[tuple[int, int, int]] = [
    (4, 2, 16),
    (6, 3, 24),
    (10, 3, 40),
]

DEFAULT_MEMORY_WALL_QUBITS: list[int] = [26, 27, 28, 29]

QUICK_MEMORY_WALL_QUBITS: list[int] = [12, 14]

DEFAULT_CHEMISTRY_CONFIGS: list[tuple[int, int, int]] = [
    (4, 14, 240),
    (4, 18, 240),
    (5, 22, 240),
    (5, 24, 240),
    (6, 26, 240),
    (6, 28, 240),
    (6, 30, 240),
    (6, 40, 240),
    (6, 60, 240),
    (6, 64, 240),
    (8, 80, 240),
    (8, 100, 240),
    (8, 120, 240),
    (8, 126, 240),
]

QUICK_CHEMISTRY_CONFIGS: list[tuple[int, int, int]] = [
    (2, 6, 40),
    (4, 14, 60),
    (6, 64, 60),
    (8, 126, 60),
]


def build_dense_benchmark_circuit_and_hamiltonian(
    n_qubits: int, n_layers: int, n_pauli_terms: int
):
    """
    Constructs a full-width $n$-qubit parametrized `Ry + CNOT + ExpPauli` circuit
    and an $M$-term Pauli Hamiltonian ($50\\%$ diagonal $Z_i Z_{i+1}$ + $50\\%$ off-diagonal $X_i Y_{i+1}$).
    """
    import tequila as tq

    U = tq.QCircuit()
    for layer in range(n_layers):
        for q in range(n_qubits):
            U += tq.gates.Ry(angle=f"a_{layer}_{q}", target=q)
        for q in range(n_qubits - 1):
            U += tq.gates.CNOT(control=q, target=q + 1)
        pauli_str = "".join([f"X({q})" for q in range(min(4, n_qubits))])
        U += tq.gates.ExpPauli(angle=f"ep_{layer}", paulistring=pauli_str)

    n_half = max(1, n_pauli_terms // 2)
    diag_terms = [
        f"{0.1 * (i + 1)}*Z({i % n_qubits})Z({(i + 1) % n_qubits})"
        for i in range(n_half)
    ]
    off_terms = [
        f"{0.05 * (i + 1)}*X({i % n_qubits})Y({(i + 1) % n_qubits})"
        for i in range(max(1, n_pauli_terms - n_half))
    ]
    H = tq.QubitHamiltonian(" + ".join(diag_terms + off_terms))
    E = tq.ExpectationValue(H=H, U=U)
    variables = {v: 0.23 for v in E.extract_variables()}
    return U, H, E, variables


def build_plucker_hcb_chemistry_circuit_and_hamiltonian(
    n_occ: int, n_orb: int, n_pauli_terms: int = 240
):
    """
    Constructs a full-width $n_{\\mathrm{orb}}$-qubit particle-number-conserving `HCB-SPA` / `MRA-PNO`
    chemistry circuit on $\\mathrm{Gr}_{\\mathbb{C}}(n_{\\mathrm{occ}}, n_{\\mathrm{orb}})$ with active
    Slater-determinant support $2^{n_{\\mathrm{occ}}} \\le \\binom{n_{\\mathrm{orb}}}{n_{\\mathrm{occ}}} \\ll 2^{n_{\\mathrm{orb}}}$.
    """
    import tequila as tq

    U = tq.QCircuit()
    for p in range(n_occ):
        U += tq.gates.X(target=p)
    for p in range(n_occ):
        target_virt = n_occ + p
        if target_virt < n_orb:
            U += tq.gates.Ry(angle=f"th_{p}", target=target_virt)
            U += tq.gates.CNOT(control=target_virt, target=p)
    for q in range(2 * n_occ, n_orb):
        src = q % max(1, 2 * n_occ)
        U += tq.gates.Rz(angle=f"phi_{q}", target=q)
        U += tq.gates.CNOT(control=src, target=q)
        U += tq.gates.CNOT(control=q, target=src)

    n_diag = max(1, int(0.7 * n_pauli_terms))
    n_off = max(2, n_pauli_terms - n_diag)
    H_str = "-1.5 + " + " + ".join(
        [
            f"{-0.2 * (i + 1)}*Z({i % n_orb})Z({(i + 1) % n_orb})"
            for i in range(n_diag)
        ]
        + [
            f"{0.08 * (i + 1)}*X({i % n_orb})X({(i + 1) % n_orb})"
            for i in range(n_off // 2)
        ]
        + [
            f"{0.08 * (i + 1)}*Y({i % n_orb})Y({(i + 1) % n_orb})"
            for i in range(n_off - n_off // 2)
        ]
    )
    H = tq.QubitHamiltonian(H_str)
    E = tq.ExpectationValue(H=H, U=U)
    variables = {v: 0.35 for v in E.extract_variables()}
    return U, H, E, variables


def benchmark_dense_expectation_scaling(
    configs: list[tuple[int, int, int]] | None = None,
    backends: list[str] | None = None,
    repeats: int = 5,
    include_symbolic_max_qubits: int = 6,
    lambdify_backend: str = DEFAULT_LAMBDIFY_BACKEND,
) -> list[dict[str, typing.Any]]:
    """
    Benchmarks Table 1 (Regime A): Full dense $2^n$ parametrized `Ry + CNOT + ExpPauli`
    expectation values across `configs = [(n_qubits, n_layers, n_pauli_terms), ...]`
    and compares `signals` against `qulacs`, `cirq`, `qiskit`, and `symbolic`.
    """
    import time
    import tequila as tq

    BackendCircuitSignals.set_lambdify_backend(lambdify_backend)
    if configs is None:
        configs = list(DEFAULT_DENSE_CONFIGS)

    config_summary = ", ".join(
        f"(n={nq}, L={nl}, M={nt})" for nq, nl, nt in configs
    )
    logger.info(
        "Running dense 2^n expectation benchmark across %d configuration(s) (repeats=%d, lambdify_backend=%s): [%s]",
        len(configs),
        repeats,
        lambdify_backend,
        config_summary,
    )
    rows: list[dict[str, typing.Any]] = []
    for cfg_idx, (n_qubits, n_layers, n_terms) in enumerate(configs, start=1):
        _U, _H, E, vars_d = build_dense_benchmark_circuit_and_hamiltonian(
            n_qubits, n_layers, n_terms
        )
        if backends is not None:
            active_backends = list(backends)
        else:
            active_backends = ["signals"]
            for cand in ("qulacs", "cirq", "qiskit"):
                if cand in tq.INSTALLED_SIMULATORS:
                    if cand in ("cirq", "qiskit") and n_qubits > 16:
                        continue
                    active_backends.append(cand)
            if (
                n_qubits <= include_symbolic_max_qubits
                and "symbolic" in tq.INSTALLED_SIMULATORS
            ):
                active_backends.append("symbolic")

        ram_bytes = 16 * (1 << int(n_qubits))
        logger.info(
            "[Dense %d/%d] Testing n_qubits=%d (2^%d=%d, statevec=%d B, layers=%d, terms=%d) on backends=%s",
            cfg_idx,
            len(configs),
            n_qubits,
            n_qubits,
            1 << n_qubits,
            ram_bytes,
            n_layers,
            n_terms,
            active_backends,
        )
        t_sig: float | None = None
        for b in active_backends:
            if b not in tq.INSTALLED_SIMULATORS:
                logger.debug("Skipping uninstalled backend=%s", b)
                continue
            r_eff = (
                1
                if (b == "symbolic" and n_qubits >= 6) or n_qubits >= 22
                else max(1, int(repeats))
            )
            logger.info(
                "  -> [Dense %d/%d] n=%d (L=%d, M=%d) | running backend='%s' (repeats=%d) ...",
                cfg_idx,
                len(configs),
                n_qubits,
                n_layers,
                n_terms,
                b,
                r_eff,
            )
            t_c0 = time.perf_counter()
            try:
                c_E = tq.compile(E, variables=vars_d, backend=b)
                compile_ms = (time.perf_counter() - t_c0) * 1000.0
                t_eval0 = time.perf_counter()
                val = float(c_E(variables=vars_d))
                first_eval_ms = (time.perf_counter() - t_eval0) * 1000.0
            except Exception as ex:
                logger.info(
                    "  <- [Dense %d/%d] n=%d | backend='%s' | FAILED (%s: %s)",
                    cfg_idx,
                    len(configs),
                    n_qubits,
                    b,
                    type(ex).__name__,
                    ex,
                )
                rows.append(
                    {
                        "regime": "dense_2n",
                        "n_qubits": int(n_qubits),
                        "hilbert_dim": 1 << int(n_qubits),
                        "statevector_bytes": 16 * (1 << int(n_qubits)),
                        "n_layers": int(n_layers),
                        "n_pauli_terms": int(n_terms),
                        "backend": str(b),
                        "status": "FAILED",
                        "compile_ms": None,
                        "median_ms": None,
                        "min_ms": None,
                        "speedup_vs_signals": None,
                        "expectation_value": None,
                        "error": f"{type(ex).__name__}: {ex}",
                    }
                )
                continue

            if r_eff == 1:
                times = [first_eval_ms]
            else:
                times = []
                for _ in range(r_eff):
                    t0 = time.perf_counter()
                    val = float(c_E(variables=vars_d))
                    times.append((time.perf_counter() - t0) * 1000.0)
            med = float(np.median(times))
            min_t = float(np.min(times))
            if b in ("signals", "grassmann") and t_sig is None:
                t_sig = med
            speedup = (med / t_sig) if (t_sig and t_sig > 0.0) else 1.0
            logger.info(
                "  <- [Dense %d/%d] n=%d | backend='%s' | median=%.3f ms (%.2fx vs signals) | <H>=%.6f",
                cfg_idx,
                len(configs),
                n_qubits,
                b,
                med,
                speedup,
                val,
            )
            logger.debug(
                "Dense n=%d | backend=%s | compile=%.3f ms | median=%.3f ms | min=%.3f ms | speedup=%.2fx | <H>=%.6f",
                n_qubits,
                b,
                compile_ms,
                med,
                min_t,
                speedup,
                val,
            )
            rows.append(
                {
                    "regime": "dense_2n",
                    "n_qubits": int(n_qubits),
                    "hilbert_dim": 1 << int(n_qubits),
                    "statevector_bytes": 16 * (1 << int(n_qubits)),
                    "n_layers": int(n_layers),
                    "n_pauli_terms": int(n_terms),
                    "backend": str(b),
                    "status": "OK",
                    "compile_ms": compile_ms,
                    "median_ms": med,
                    "min_ms": min_t,
                    "speedup_vs_signals": speedup,
                    "expectation_value": val,
                    "error": None,
                }
            )
    return rows


def benchmark_dense_memory_wall_limit(
    qubits_list: list[int] | None = None,
    backends: list[str] | None = None,
    max_dense_ram_gb: float = 6.0,
) -> list[dict[str, typing.Any]]:
    """
    Probes the high-$n$ dense $2^n$ statevector hardware RAM limit (e.g. $n = 26, 27, 28, 29$,
    where $n = 28$ requires $4.00\\text{ GiB}$ per complex128 statevector and $n = 29$ requires $8.00\\text{ GiB}$),
    measuring both cold (first allocation + page fault) and warm (persistent `scratch_state` buffer) latencies.
    """
    import time
    import tequila as tq

    if qubits_list is None:
        qubits_list = list(DEFAULT_MEMORY_WALL_QUBITS)
    if backends is None:
        backends = [
            b for b in ("signals", "qulacs") if b in tq.INSTALLED_SIMULATORS
        ]

    mw_summary = ", ".join(
        f"n={nq} ({((1 << int(nq)) * 16) / (1024**3):.4f} GiB)"
        for nq in qubits_list
    )
    logger.info(
        "Probing high-n dense 2^n memory wall across %d configuration(s) [%s] on backends=%s (max_dense_ram_gb=%.2f GiB)",
        len(qubits_list),
        mw_summary,
        backends,
        max_dense_ram_gb,
    )
    rows: list[dict[str, typing.Any]] = []
    for cfg_idx, nq in enumerate(qubits_list, start=1):
        ram_gb = float((1 << int(nq)) * 16) / float(1024**3)
        logger.info(
            "[Memory-Wall %d/%d] Testing n_qubits=%d (2^%d=%d, statevector=%.4f GiB) on backends=%s",
            cfg_idx,
            len(qubits_list),
            nq,
            nq,
            1 << int(nq),
            ram_gb,
            backends,
        )
        U = tq.QCircuit()
        for q in range(nq):
            U += tq.gates.Ry(angle=f"a_{q}", target=q)
        for q in range(nq - 1):
            U += tq.gates.CNOT(control=q, target=q + 1)
        H_str = " + ".join(
            [
                f"0.5*Z({q})Z({(q + 1) % nq}) + 0.25*X({q})Y({(q + 1) % nq})"
                for q in range(min(nq, 16))
            ]
        )
        H = tq.QubitHamiltonian(H_str)
        E = tq.ExpectationValue(H=H, U=U)
        vars_d = {v: 0.23 for v in E.extract_variables()}

        t_sig_warm: float | None = None
        for b in backends:
            if ram_gb > max_dense_ram_gb:
                logger.info(
                    "  -> [Memory-Wall %d/%d] n=%d (%.2f GiB) | backend='%s' | SKIPPED_RAM_LIMIT (> %.2f GiB)",
                    cfg_idx,
                    len(qubits_list),
                    nq,
                    ram_gb,
                    b,
                    max_dense_ram_gb,
                )
                rows.append(
                    {
                        "regime": "dense_memory_wall",
                        "n_qubits": int(nq),
                        "statevector_gb": ram_gb,
                        "backend": str(b),
                        "status": "SKIPPED_RAM_LIMIT",
                        "cold_ms": None,
                        "warm_ms": None,
                        "speedup_vs_signals": None,
                        "expectation_value": None,
                        "error": f"Statevector {ram_gb:.2f} GiB exceeds max_dense_ram_gb={max_dense_ram_gb:.2f} GiB",
                    }
                )
                continue
            logger.info(
                "  -> [Memory-Wall %d/%d] n=%d (%.4f GiB) | running backend='%s' (cold + warm eval) ...",
                cfg_idx,
                len(qubits_list),
                nq,
                ram_gb,
                b,
            )
            try:
                c_E = tq.compile(E, variables=vars_d, backend=b)
                t0 = time.perf_counter()
                val = float(c_E(variables=vars_d))
                cold_ms = (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                val = float(c_E(variables=vars_d))
                warm_ms = (time.perf_counter() - t1) * 1000.0
                if b in ("signals", "grassmann") and t_sig_warm is None:
                    t_sig_warm = warm_ms
                speedup = (
                    (warm_ms / t_sig_warm)
                    if (t_sig_warm and t_sig_warm > 0.0)
                    else 1.0
                )
                logger.info(
                    "  <- [Memory-Wall %d/%d] n=%d (%.4f GiB) | backend='%s' | cold=%.2f ms | warm=%.2f ms (%.2fx vs signals)",
                    cfg_idx,
                    len(qubits_list),
                    nq,
                    ram_gb,
                    b,
                    cold_ms,
                    warm_ms,
                    speedup,
                )
                logger.debug(
                    "Memory-wall n=%d (%.4f GiB) | backend=%s | cold=%.2f ms | warm=%.2f ms | speedup=%.2fx",
                    nq,
                    ram_gb,
                    b,
                    cold_ms,
                    warm_ms,
                    speedup,
                )
                rows.append(
                    {
                        "regime": "dense_memory_wall",
                        "n_qubits": int(nq),
                        "statevector_gb": ram_gb,
                        "backend": str(b),
                        "status": "OK",
                        "cold_ms": cold_ms,
                        "warm_ms": warm_ms,
                        "speedup_vs_signals": speedup,
                        "expectation_value": val,
                        "error": None,
                    }
                )
                del c_E
            except Exception as ex:
                logger.info(
                    "  <- [Memory-Wall %d/%d] n=%d (%.4f GiB) | backend='%s' | FAILED (%s: %s)",
                    cfg_idx,
                    len(qubits_list),
                    nq,
                    ram_gb,
                    b,
                    type(ex).__name__,
                    ex,
                )
                rows.append(
                    {
                        "regime": "dense_memory_wall",
                        "n_qubits": int(nq),
                        "statevector_gb": ram_gb,
                        "backend": str(b),
                        "status": "FAILED",
                        "cold_ms": None,
                        "warm_ms": None,
                        "speedup_vs_signals": None,
                        "expectation_value": None,
                        "error": f"{type(ex).__name__}: {ex}",
                    }
                )
    return rows


def benchmark_plucker_hcb_chemistry_scaling(
    configs: list[tuple[int, int, int]] | None = None,
    backends: list[str] | None = None,
    repeats: int = 5,
    max_dense_qubits_other_backends: int = 26,
) -> list[dict[str, typing.Any]]:
    """
    Benchmarks Table 2 (Regime B): Particle-number-conserving `HCB-SPA` / `MRA-PNO`
    chemistry circuits on $\\mathrm{Gr}_{\\mathbb{C}}(n_{\\mathrm{occ}}, n_{\\mathrm{orb}})$
    from $n = 4$ up to $n = 126$ qubits (FeMoco nitrogenase active-space scale).
    """
    import math
    import time
    import tequila as tq

    if configs is None:
        configs = list(DEFAULT_CHEMISTRY_CONFIGS)

    chem_summary = ", ".join(
        f"Gr({n_occ},{n_orb})[M={nt}]" for n_occ, n_orb, nt in configs
    )
    logger.info(
        "Running particle-conserving Plucker HCB-SPA chemistry benchmark across %d configuration(s) (repeats=%d): [%s]",
        len(configs),
        repeats,
        chem_summary,
    )
    rows: list[dict[str, typing.Any]] = []
    for cfg_idx, (n_occ, n_orb, n_terms) in enumerate(configs, start=1):
        _U, _H, E, vars_d = build_plucker_hcb_chemistry_circuit_and_hamiltonian(
            n_occ, n_orb, n_terms
        )
        if backends is not None:
            active_backends = list(backends)
        else:
            active_backends = ["signals"]
            if (
                n_orb <= max_dense_qubits_other_backends
                and "qulacs" in tq.INSTALLED_SIMULATORS
            ):
                active_backends.append("qulacs")
            if n_orb <= 18:
                for cand in ("cirq", "qiskit"):
                    if cand in tq.INSTALLED_SIMULATORS:
                        active_backends.append(cand)

        plucker_dim = int(math.comb(n_orb, n_occ))
        logger.info(
            "[Chemistry %d/%d] Testing Gr_C(%d,%d) (n_qubits=%d, C(n,k)=%d, terms=%d) on backends=%s",
            cfg_idx,
            len(configs),
            n_occ,
            n_orb,
            n_orb,
            plucker_dim,
            n_terms,
            active_backends,
        )
        t_sig: float | None = None
        for b in active_backends:
            if b not in tq.INSTALLED_SIMULATORS:
                continue
            if b not in ("signals", "grassmann") and n_orb > max_dense_qubits_other_backends:
                logger.info(
                    "  -> [Chemistry %d/%d] Gr(%d,%d) | backend='%s' | SKIPPED_DENSE_WALL (n=%d > %d)",
                    cfg_idx,
                    len(configs),
                    n_occ,
                    n_orb,
                    b,
                    n_orb,
                    max_dense_qubits_other_backends,
                )
                rows.append(
                    {
                        "regime": "plucker_hcb_chemistry",
                        "n_occ": int(n_occ),
                        "n_orb": int(n_orb),
                        "n_qubits": int(n_orb),
                        "hilbert_dim": 1 << int(n_orb),
                        "plucker_sector_dim": plucker_dim,
                        "active_support_size": None,
                        "sparse_bytes": None,
                        "n_pauli_terms": int(n_terms),
                        "backend": str(b),
                        "status": "SKIPPED_DENSE_WALL",
                        "median_ms": None,
                        "speedup_vs_signals": None,
                        "expectation_value": None,
                        "error": f"Dense 2^{n_orb} statevector exceeds {max_dense_qubits_other_backends}-qubit limit",
                    }
                )
                continue
            r_eff = (
                max(1, int(repeats))
                if (b in ("signals", "grassmann") or n_orb <= 20)
                else 1
            )
            logger.info(
                "  -> [Chemistry %d/%d] Gr(%d,%d) (n=%d) | running backend='%s' (repeats=%d) ...",
                cfg_idx,
                len(configs),
                n_occ,
                n_orb,
                n_orb,
                b,
                r_eff,
            )
            try:
                c_E = tq.compile(E, variables=vars_d, backend=b)
                t_eval0 = time.perf_counter()
                val = float(c_E(variables=vars_d))
                first_eval_ms = (time.perf_counter() - t_eval0) * 1000.0
                if r_eff == 1:
                    times = [first_eval_ms]
                else:
                    times = []
                    for _ in range(r_eff):
                        t0 = time.perf_counter()
                        val = float(c_E(variables=vars_d))
                        times.append((time.perf_counter() - t0) * 1000.0)
                med = float(np.median(times))
                if b in ("signals", "grassmann"):
                    if t_sig is None:
                        t_sig = med
                    supp = int(c_E.args[0].U._rust_circuit.last_support_size())
                    sp_bytes = 32 * supp
                else:
                    supp = 1 << int(n_orb)
                    sp_bytes = 16 * supp
                speedup = (med / t_sig) if (t_sig and t_sig > 0.0) else 1.0
                logger.info(
                    "  <- [Chemistry %d/%d] Gr(%d,%d) | backend='%s' | support=%d | median=%.3f ms (%.2fx vs signals) | <H>=%.6f",
                    cfg_idx,
                    len(configs),
                    n_occ,
                    n_orb,
                    b,
                    supp,
                    med,
                    speedup,
                    val,
                )
                logger.debug(
                    "Chemistry Gr(%d,%d) | backend=%s | support=%d | median=%.3f ms | speedup=%.2fx | <H>=%.6f",
                    n_occ,
                    n_orb,
                    b,
                    supp,
                    med,
                    speedup,
                    val,
                )
                rows.append(
                    {
                        "regime": "plucker_hcb_chemistry",
                        "n_occ": int(n_occ),
                        "n_orb": int(n_orb),
                        "n_qubits": int(n_orb),
                        "hilbert_dim": 1 << int(n_orb),
                        "plucker_sector_dim": plucker_dim,
                        "active_support_size": supp,
                        "sparse_bytes": sp_bytes,
                        "n_pauli_terms": int(n_terms),
                        "backend": str(b),
                        "status": "OK",
                        "median_ms": med,
                        "speedup_vs_signals": speedup,
                        "expectation_value": val,
                        "error": None,
                    }
                )
            except Exception as ex:
                logger.info(
                    "  <- [Chemistry %d/%d] Gr(%d,%d) | backend='%s' | FAILED (%s: %s)",
                    cfg_idx,
                    len(configs),
                    n_occ,
                    n_orb,
                    b,
                    type(ex).__name__,
                    ex,
                )
                rows.append(
                    {
                        "regime": "plucker_hcb_chemistry",
                        "n_occ": int(n_occ),
                        "n_orb": int(n_orb),
                        "n_qubits": int(n_orb),
                        "hilbert_dim": 1 << int(n_orb),
                        "plucker_sector_dim": plucker_dim,
                        "active_support_size": None,
                        "sparse_bytes": None,
                        "n_pauli_terms": int(n_terms),
                        "backend": str(b),
                        "status": "FAILED",
                        "median_ms": None,
                        "speedup_vs_signals": None,
                        "expectation_value": None,
                        "error": f"{type(ex).__name__}: {ex}",
                    }
                )
    return rows


def benchmark_vqe_optimization_and_wavefunction(
    backends: list[str] | None = None,
    method: str = "BFGS",
    samples: int = 1000,
    lambdify_backend: str = DEFAULT_LAMBDIFY_BACKEND,
) -> list[dict[str, typing.Any]]:
    """
    Benchmarks Table 3 (Regime C): Full multi-parameter VQE optimization (`tq.minimize`
    with analytical parameter-shift gradients), statevector simulation (`tq.simulate(U)`),
    and shot sampling (`tq.simulate(U, samples=...)`) across Tequila backends.
    """
    import time
    import tequila as tq

    BackendCircuitSignals.set_lambdify_backend(lambdify_backend)
    U_vqe = (
        tq.gates.X(target=0)
        + tq.gates.Ry(angle="a", target=1)
        + tq.gates.CNOT(control=1, target=0)
        + tq.gates.Ry(angle="b", target=2)
        + tq.gates.CNOT(control=2, target=1)
        + tq.gates.Ry(angle="c", target=3)
        + tq.gates.CNOT(control=3, target=2)
    )
    H_vqe = tq.QubitHamiltonian(
        "-0.5 + 0.35*Z(0) - 0.35*Z(1) + 0.22*Z(0)Z(1) + 0.18*X(0)X(1) "
        "+ 0.18*Y(0)Y(1) + 0.15*Z(2)Z(3) + 0.12*X(1)X(2) + 0.12*Y(1)Y(2)"
    )
    E_vqe = tq.ExpectationValue(H=H_vqe, U=U_vqe)
    init_vars = {"a": 0.4, "b": 0.2, "c": -0.3}

    if backends is None:
        backends = [
            b
            for b in ("signals", "qulacs", "cirq", "symbolic", "qiskit")
            if b in tq.INSTALLED_SIMULATORS
        ]

    logger.info(
        "Running VQE %s optimization (n=4, 3 params, 9 Pauli terms), wavefunction, and sampling (%d shots) across %d backend(s): %s",
        method,
        samples,
        len(backends),
        backends,
    )
    rows: list[dict[str, typing.Any]] = []
    t_sig_vqe: float | None = None
    e_sig_opt: float | None = None
    for b_idx, b in enumerate(backends, start=1):
        if b not in tq.INSTALLED_SIMULATORS:
            continue
        logger.info(
            "  -> [VQE %d/%d] Running tq.minimize(method='%s') + simulate + sample on backend='%s' ...",
            b_idx,
            len(backends),
            method,
            b,
        )
        try:
            t0 = time.perf_counter()
            res = tq.minimize(
                objective=E_vqe,
                method=method,
                initial_values=dict(init_vars),
                backend=b,
                silent=True,
            )
            vqe_ms = (time.perf_counter() - t0) * 1000.0
            opt_e = float(res.energy)

            t1 = time.perf_counter()
            _wfn = tq.simulate(U_vqe, variables=init_vars, backend=b)
            wfn_ms = (time.perf_counter() - t1) * 1000.0

            sample_ms: float | None = None
            if b in tq.INSTALLED_SAMPLERS:
                t2 = time.perf_counter()
                _s = tq.simulate(
                    U_vqe, variables=init_vars, samples=int(samples), backend=b
                )
                sample_ms = (time.perf_counter() - t2) * 1000.0

            if b in ("signals", "grassmann") and t_sig_vqe is None:
                t_sig_vqe = vqe_ms
                e_sig_opt = opt_e
            speedup = (
                (vqe_ms / t_sig_vqe) if (t_sig_vqe and t_sig_vqe > 0.0) else 1.0
            )
            e_diff = abs(opt_e - e_sig_opt) if e_sig_opt is not None else 0.0
            logger.info(
                "  <- [VQE %d/%d] backend='%s' | vqe=%.3f ms (%.2fx vs signals) | wfn=%.3f ms | E_opt=%.8f",
                b_idx,
                len(backends),
                b,
                vqe_ms,
                speedup,
                wfn_ms,
                opt_e,
            )
            logger.debug(
                "VQE backend=%s | vqe=%.3f ms | wfn=%.3f ms | sample=%s ms | speedup=%.2fx | E_opt=%.8f",
                b,
                vqe_ms,
                wfn_ms,
                f"{sample_ms:.3f}" if sample_ms is not None else "N/A",
                speedup,
                opt_e,
            )
            rows.append(
                {
                    "regime": "vqe_minimize",
                    "backend": str(b),
                    "method": str(method),
                    "status": "OK",
                    "vqe_ms": vqe_ms,
                    "wfn_ms": wfn_ms,
                    "sample_ms": sample_ms,
                    "speedup_vs_signals": speedup,
                    "optimal_energy": opt_e,
                    "energy_abs_diff": e_diff,
                    "error": None,
                }
            )
        except Exception as ex:
            logger.info(
                "  <- [VQE %d/%d] backend='%s' | FAILED (%s: %s)",
                b_idx,
                len(backends),
                b,
                type(ex).__name__,
                ex,
            )
            rows.append(
                {
                    "regime": "vqe_minimize",
                    "backend": str(b),
                    "method": str(method),
                    "status": "FAILED",
                    "vqe_ms": None,
                    "wfn_ms": None,
                    "sample_ms": None,
                    "speedup_vs_signals": None,
                    "optimal_energy": None,
                    "energy_abs_diff": None,
                    "error": f"{type(ex).__name__}: {ex}",
                }
            )
    return rows


def benchmark_grassmann_plucker_permanent_vs_determinant(
    configs: list[tuple[int, int]] | None = None,
) -> list[dict[str, typing.Any]]:
    """
    Benchmarks and verifies Cauchy–Binet Fermionic Jordan–Wigner determinants ($\\det C[:, I]$,
    `fermionic_jw_sign=True`) vs. Bosonic Hard-Core Boson Ryser permanents ($\\operatorname{perm} C[:, I]$,
    `fermionic_jw_sign=False`) on $\\mathrm{Gr}_{\\mathbb{C}}(n_{\\mathrm{occ}}, n_{\\mathrm{orb}})$.
    """
    import math
    import time

    if configs is None:
        configs = [(2, 4), (2, 6), (3, 6), (2, 10), (2, 12), (4, 16)]

    pl_summary = ", ".join(f"Gr_C({k},{n})" for k, n in configs)
    logger.info(
        "Running Grassmannian Plucker determinant vs. permanent benchmark across %d manifold(s): [%s]",
        len(configs),
        pl_summary,
    )
    rows: list[dict[str, typing.Any]] = []
    for cfg_idx, (n_occ, n_orb) in enumerate(configs, start=1):
        plucker_dim = int(math.comb(n_orb, n_occ))
        hilbert_dim = 1 << int(n_orb)
        logger.info(
            "  -> [Plucker %d/%d] Evaluating Gr_C(%d,%d) (2^%d=%d, C(%d,%d)=%d) det vs. perm ...",
            cfg_idx,
            len(configs),
            n_occ,
            n_orb,
            n_orb,
            hilbert_dim,
            n_orb,
            n_occ,
            plucker_dim,
        )
        rots = [
            (p, min(n_orb - 1, n_occ + p), float(np.pi / (p + 2)))
            for p in range(n_occ)
        ]
        t0 = time.perf_counter()
        det_states = sna.py_simulate_grassmann_plucker_hcb(
            int(n_occ), int(n_orb), rots, True
        )
        det_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        perm_states = sna.py_simulate_grassmann_plucker_hcb(
            int(n_occ), int(n_orb), rots, False
        )
        perm_ms = (time.perf_counter() - t1) * 1000.0

        det_norm = float(sum(r * r + i * i for _, r, i in det_states))
        perm_norm = float(sum(r * r + i * i for _, r, i in perm_states))
        logger.info(
            "  <- [Plucker %d/%d] Gr_C(%d,%d) | compression=%.2fx | det=%.3f ms | perm=%.3f ms",
            cfg_idx,
            len(configs),
            n_occ,
            n_orb,
            float(hilbert_dim) / float(plucker_dim),
            det_ms,
            perm_ms,
        )
        logger.debug(
            "Plucker Gr(%d,%d) | C(n,k)=%d | compression=%.2fx | det=%.3f ms (norm^2=%.6f) | perm=%.3f ms (norm^2=%.6f)",
            n_occ,
            n_orb,
            plucker_dim,
            float(hilbert_dim) / float(plucker_dim),
            det_ms,
            det_norm,
            perm_ms,
            perm_norm,
        )
        rows.append(
            {
                "regime": "plucker_det_vs_perm",
                "n_occ": int(n_occ),
                "n_orb": int(n_orb),
                "hilbert_dim": hilbert_dim,
                "plucker_sector_dim": plucker_dim,
                "compression_ratio": float(hilbert_dim) / float(plucker_dim),
                "det_active_states": len(det_states),
                "perm_active_states": len(perm_states),
                "det_norm_sq": det_norm,
                "perm_norm_sq": perm_norm,
                "det_ms": det_ms,
                "perm_ms": perm_ms,
            }
        )
    return rows


def run_all_benchmarks(
    suite: str = "all",
    quick: bool = False,
    backends: list[str] | None = None,
    qubits: list[int] | None = None,
    chemistry_configs: list[tuple[int, int, int]] | None = None,
    repeats: int = 5,
    max_dense_ram_gb: float = 6.0,
    lambdify_backend: str = DEFAULT_LAMBDIFY_BACKEND,
) -> dict[str, list[dict[str, typing.Any]]]:
    """
    Executes the requested Tequila `signals` / `grassmann` benchmark suite(s) and returns
    a dictionary mapping suite names to lists of benchmark result dictionaries.
    """
    BackendCircuitSignals.set_lambdify_backend(lambdify_backend)
    eff_repeats = 2 if quick else max(1, int(repeats))
    suite_norm = suite.strip().lower()
    if suite_norm == "quick":
        quick = True
        suite_norm = "all"

    results: dict[str, list[dict[str, typing.Any]]] = {}

    if suite_norm in ("all", "dense"):
        if qubits is not None:
            dense_cfgs = [(q, 2 if q <= 6 else 4, 24 if quick else 60) for q in qubits]
        else:
            dense_cfgs = list(QUICK_DENSE_CONFIGS if quick else DEFAULT_DENSE_CONFIGS)
        results["dense_expectation"] = benchmark_dense_expectation_scaling(
            configs=dense_cfgs,
            backends=backends,
            repeats=eff_repeats,
            include_symbolic_max_qubits=4 if quick else 6,
            lambdify_backend=lambdify_backend,
        )

    if suite_norm in ("all", "memory-wall"):
        mw_qubits = (
            list(qubits)
            if qubits is not None
            else list(QUICK_MEMORY_WALL_QUBITS if quick else DEFAULT_MEMORY_WALL_QUBITS)
        )
        results["dense_memory_wall"] = benchmark_dense_memory_wall_limit(
            qubits_list=mw_qubits,
            backends=backends,
            max_dense_ram_gb=max_dense_ram_gb,
        )

    if suite_norm in ("all", "chemistry"):
        chem_cfgs = (
            list(chemistry_configs)
            if chemistry_configs is not None
            else list(QUICK_CHEMISTRY_CONFIGS if quick else DEFAULT_CHEMISTRY_CONFIGS)
        )
        results["plucker_hcb_chemistry"] = benchmark_plucker_hcb_chemistry_scaling(
            configs=chem_cfgs,
            backends=backends,
            repeats=eff_repeats,
        )

    if suite_norm in ("all", "vqe"):
        vqe_backends = (
            list(backends)
            if backends is not None
            else (["signals", "qulacs"] if quick else None)
        )
        results["vqe_optimization"] = benchmark_vqe_optimization_and_wavefunction(
            backends=vqe_backends,
            samples=250 if quick else 1000,
            lambdify_backend=lambdify_backend,
        )

    if suite_norm in ("all", "plucker"):
        pl_cfgs = [(2, 4), (2, 6), (2, 10)] if quick else None
        results["plucker_det_vs_perm"] = (
            benchmark_grassmann_plucker_permanent_vs_determinant(configs=pl_cfgs)
        )

    return results


def format_benchmark_markdown_tables(
    results: dict[str, list[dict[str, typing.Any]]]
) -> str:
    """
    Formats the output of `run_all_benchmarks(...)` as MyST/Markdown tables.
    """
    lines: list[str] = []

    if "dense_expectation" in results:
        lines.append("### Table 1: Dense $2^n$ Parametrized Circuit + Pauli Hamiltonian Scaling")
        lines.append("")
        lines.append(
            "| Qubits $n$ | $2^n$ | Terms $M$ | Backend | Status | Median (ms) | Speedup vs `signals` | $\\langle H \\rangle$ |"
        )
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in results["dense_expectation"]:
            med_s = f"{r['median_ms']:.3f}" if r["median_ms"] is not None else "—"
            spd_s = (
                f"{r['speedup_vs_signals']:.2f}x"
                if r["speedup_vs_signals"] is not None
                else "—"
            )
            ev_s = (
                f"{r['expectation_value']:.6f}"
                if r["expectation_value"] is not None
                else str(r.get("error", "FAILED"))
            )
            lines.append(
                f"| {r['n_qubits']} | {r['hilbert_dim']} | {r['n_pauli_terms']} | `{r['backend']}` | {r['status']} | {med_s} | {spd_s} | {ev_s} |"
            )
        lines.append("")

    if "dense_memory_wall" in results:
        lines.append("### Table 2: High-$n$ Dense $2^n$ Statevector Hardware RAM Ceiling")
        lines.append("")
        lines.append(
            "| Qubits $n$ | Statevector (GiB) | Backend | Status | Cold (ms) | Warm (ms) | Speedup vs `signals` | $\\langle H \\rangle$ |"
        )
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in results["dense_memory_wall"]:
            c_s = f"{r['cold_ms']:.2f}" if r["cold_ms"] is not None else "—"
            w_s = f"{r['warm_ms']:.2f}" if r["warm_ms"] is not None else "—"
            spd_s = (
                f"{r['speedup_vs_signals']:.2f}x"
                if r["speedup_vs_signals"] is not None
                else "—"
            )
            ev_s = (
                f"{r['expectation_value']:.6f}"
                if r["expectation_value"] is not None
                else "—"
            )
            lines.append(
                f"| {r['n_qubits']} | {r['statevector_gb']:.4f} | `{r['backend']}` | {r['status']} | {c_s} | {w_s} | {spd_s} | {ev_s} |"
            )
        lines.append("")

    if "plucker_hcb_chemistry" in results:
        lines.append(
            "### Table 3: Particle-Number-Conserving `HCB-SPA` / `MRA-PNO` Chemistry Scaling ($n \\le 126$ Qubits)"
        )
        lines.append("")
        lines.append(
            "| Active Space | Qubits $n$ | Support $\\|\\mathrm{supp}\\|$ | Backend | Status | Median (ms) | Speedup vs `signals` | $\\langle H \\rangle$ |"
        )
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in results["plucker_hcb_chemistry"]:
            med_s = f"{r['median_ms']:.3f}" if r["median_ms"] is not None else "—"
            spd_s = (
                f"{r['speedup_vs_signals']:.2f}x"
                if r["speedup_vs_signals"] is not None
                else "—"
            )
            ev_s = (
                f"{r['expectation_value']:.6f}"
                if r["expectation_value"] is not None
                else "—"
            )
            supp_s = (
                str(r["active_support_size"])
                if r["active_support_size"] is not None
                else "—"
            )
            lines.append(
                f"| $\\mathrm{{Gr}}_{{\\mathbb{{C}}}}({r['n_occ']},{r['n_orb']})$ | {r['n_qubits']} | {supp_s} | `{r['backend']}` | {r['status']} | {med_s} | {spd_s} | {ev_s} |"
            )
        lines.append("")

    if "vqe_optimization" in results:
        lines.append("### Table 4: Full VQE BFGS Optimization (`tq.minimize`) & Wavefunction Simulation")
        lines.append("")
        lines.append(
            "| Backend | Status | VQE `tq.minimize` (ms) | Wavefunction (ms) | Sampling (ms) | Speedup vs `signals` | $E_{\\mathrm{opt}}$ (Ha) |"
        )
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in results["vqe_optimization"]:
            v_s = f"{r['vqe_ms']:.3f}" if r["vqe_ms"] is not None else "—"
            w_s = f"{r['wfn_ms']:.3f}" if r["wfn_ms"] is not None else "—"
            s_s = f"{r['sample_ms']:.3f}" if r["sample_ms"] is not None else "—"
            spd_s = (
                f"{r['speedup_vs_signals']:.2f}x"
                if r["speedup_vs_signals"] is not None
                else "—"
            )
            e_s = (
                f"{r['optimal_energy']:.8f}"
                if r["optimal_energy"] is not None
                else "—"
            )
            lines.append(
                f"| `{r['backend']}` | {r['status']} | {v_s} | {w_s} | {s_s} | {spd_s} | {e_s} |"
            )
        lines.append("")

    if "plucker_det_vs_perm" in results:
        lines.append(
            "### Table 5: Grassmannian Plücker Fermionic Determinant ($\\det$) vs. Bosonic Permanent ($\\operatorname{perm}$)"
        )
        lines.append("")
        lines.append(
            "| Manifold | $2^{n_{\\mathrm{orb}}}$ | $\\binom{n_{\\mathrm{orb}}}{n_{\\mathrm{occ}}}$ | Compression | $\\det$ Norm$^2$ | $\\det$ (ms) | $\\operatorname{perm}$ (ms) |"
        )
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in results["plucker_det_vs_perm"]:
            lines.append(
                f"| $\\mathrm{{Gr}}_{{\\mathbb{{C}}}}({r['n_occ']},{r['n_orb']})$ | {r['hilbert_dim']} | {r['plucker_sector_dim']} | {r['compression_ratio']:.2f}x | {r['det_norm_sq']:.6f} | {r['det_ms']:.3f} | {r['perm_ms']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines)


def build_arg_parser():
    """
    Builds the `argparse.ArgumentParser` for running the Tequila `signals` / `grassmann`
    benchmark CLI (`python -m tequila.simulators.simulator_signals`).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m tequila.simulators.simulator_signals",
        description=(
            "Run head-to-head quantum simulation and high-n scaling benchmarks "
            "for the Tequila Rust/PyO3 ('signals' / 'grassmann') backend against "
            "qulacs, cirq, qiskit, and symbolic."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=["all", "quick", "dense", "memory-wall", "chemistry", "vqe", "plucker"],
        default="all",
        help="Benchmark suite to run (default: all).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast CI-friendly subset of qubit sizes and repetitions.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default=None,
        help="Comma-separated list of Tequila backends to benchmark (e.g. 'signals,qulacs,cirq,qiskit').",
    )
    parser.add_argument(
        "--qubits",
        type=str,
        default=None,
        help="Comma-separated list of qubit counts for dense / memory-wall suites (e.g. '4,6,10,14').",
    )
    parser.add_argument(
        "--chemistry-configs",
        type=str,
        default=None,
        help="Comma-separated list of n_occ:n_orb[:n_terms] triples for the chemistry suite (e.g. '4:14:120,6:64:120,8:126:120').",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed repetitions per configuration (default: 5).",
    )
    parser.add_argument(
        "--max-dense-ram-gb",
        type=float,
        default=6.0,
        help="Maximum single-buffer dense statevector size in GiB before skipping (default: 6.0).",
    )
    parser.add_argument(
        "--lambdify-backend",
        type=str,
        default=DEFAULT_LAMBDIFY_BACKEND,
        help=(
            "SymPy lambdify backend module (`modules=` argument to sympy.lambdify, "
            "e.g. 'numpy', 'math', 'cmath', 'mpmath', 'scipy'; default: 'numpy')."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write full benchmark results as JSON.",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="Optional path to write formatted Markdown benchmark tables.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging to stdout.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress printing Markdown tables to stdout and set default log level to WARNING.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
        ],
        help="Explicit logging level for stdout logger (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    return parser


def _parse_chemistry_configs(raw: str | None) -> list[tuple[int, int, int]] | None:
    if not raw:
        return None
    out: list[tuple[int, int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [int(p.strip()) for p in item.split(":")]
        if len(parts) == 2:
            out.append((parts[0], parts[1], 120))
        elif len(parts) >= 3:
            out.append((parts[0], parts[1], parts[2]))
        else:
            raise ValueError(
                f"Invalid --chemistry-configs entry '{item}'; expected n_occ:n_orb[:n_terms]"
            )
    return out


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for `python -m tequila.simulators.simulator_signals` (and aliases
    `python -m tequila.simulators.benchmark_signals` / `python -m tequila`).
    Returns 0 on success.
    """
    import json
    from pathlib import Path

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    configure_logging(
        log_level=args.log_level,
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        stream=sys.stdout,
    )
    BackendCircuitSignals.set_lambdify_backend(args.lambdify_backend)
    logger.info(
        "Starting Tequila signals/grassmann benchmark CLI (suite=%s, quick=%s, repeats=%d, lambdify_backend=%s)",
        args.suite,
        bool(args.quick),
        int(args.repeats),
        args.lambdify_backend,
    )
    logger.debug("Parsed CLI arguments: %s", vars(args))

    backends = (
        [b.strip() for b in args.backends.split(",") if b.strip()]
        if args.backends
        else None
    )
    qubits = (
        [int(q.strip()) for q in args.qubits.split(",") if q.strip()]
        if args.qubits
        else None
    )
    chem_configs = _parse_chemistry_configs(args.chemistry_configs)

    results = run_all_benchmarks(
        suite=args.suite,
        quick=bool(args.quick),
        backends=backends,
        qubits=qubits,
        chemistry_configs=chem_configs,
        repeats=int(args.repeats),
        max_dense_ram_gb=float(args.max_dense_ram_gb),
        lambdify_backend=str(args.lambdify_backend),
    )
    md_text = format_benchmark_markdown_tables(results)

    if not args.quiet:
        print(md_text)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote JSON benchmark results to %s", out_path)

    if args.markdown_out:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_text, encoding="utf-8")
        logger.info("Wrote Markdown benchmark tables to %s", md_path)

    logger.info(
        "Completed Tequila benchmark suite(s): %s",
        ", ".join(results.keys()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

