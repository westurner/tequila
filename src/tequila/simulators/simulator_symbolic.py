from tequila.simulators.simulator_base import BackendExpectationValue, BackendCircuit
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila.circuit.circuit import QCircuit
from tequila.circuit.gates import QGate
from tequila import BitString
import numpy
import copy
import sympy
import typing

"""
Simple Symbolic Simulator for debugging purposes, accelerated with sympy.lambdify
"""

DEFAULT_LAMBDIFY_BACKEND: str = "numpy"

_THETA_SYM, _QT_SYM = sympy.symbols("theta qt")
_SYMPY_GATE_FACTOR_EXPRS: dict[str, tuple[sympy.Expr, sympy.Expr]] = {
    "H": (
        sympy.Integer(-1) ** _QT_SYM * sympy.sqrt(sympy.Rational(1, 2)),
        sympy.sqrt(sympy.Rational(1, 2)),
    ),
    "X": (sympy.Integer(0), sympy.Integer(1)),
    "CNOT": (sympy.Integer(0), sympy.Integer(1)),
    "Y": (sympy.Integer(0), sympy.I * sympy.Integer(-1) ** _QT_SYM),
    "Z": (sympy.Integer(-1) ** _QT_SYM, sympy.Integer(0)),
    "RX": (
        sympy.cos(sympy.Rational(1, 2) * _THETA_SYM),
        -sympy.sin(sympy.Rational(1, 2) * _THETA_SYM) * sympy.I,
    ),
    "RY": (
        sympy.cos(-sympy.Rational(1, 2) * _THETA_SYM),
        sympy.sin(-sympy.Rational(1, 2) * _THETA_SYM)
        * sympy.Integer(-1) ** (_QT_SYM + 1),
    ),
    "RZ": (
        sympy.exp(
            -sympy.Rational(1, 2)
            * _THETA_SYM
            * sympy.I
            * sympy.Integer(-1) ** _QT_SYM
        ),
        sympy.Integer(0),
    ),
}
_LAMBDIFIED_GATE_FACTOR_CACHE: dict[tuple[str, str], typing.Callable] = {}


def _resolve_lambdify_modules(lambdify_backend: str | list[str] | None = None):
    if lambdify_backend is None:
        return DEFAULT_LAMBDIFY_BACKEND
    if isinstance(lambdify_backend, (list, tuple)):
        return list(lambdify_backend)
    b = str(lambdify_backend).strip()
    if not b:
        return DEFAULT_LAMBDIFY_BACKEND
    if "," in b:
        return [part.strip() for part in b.split(",") if part.strip()]
    if b.lower() in ("math", "cmath"):
        return ["cmath", "math"]
    return b


def get_lambdified_gate_factors(
    gate_name: str, lambdify_backend: str = DEFAULT_LAMBDIFY_BACKEND
) -> typing.Callable:
    """
    Compiles and caches the SymPy gate factor expressions `(fac1, fac2)` for `gate_name`
    using `sympy.lambdify((theta, qt), exprs, modules=...)`.
    """
    g_upper = "H" if gate_name == "H" else gate_name.upper()
    backend_str = str(lambdify_backend or DEFAULT_LAMBDIFY_BACKEND)
    key = (g_upper, backend_str)
    fn = _LAMBDIFIED_GATE_FACTOR_CACHE.get(key)
    if fn is None:
        if g_upper not in _SYMPY_GATE_FACTOR_EXPRS:
            raise Exception("Gate is not known to simulators, " + str(gate_name))
        exprs = _SYMPY_GATE_FACTOR_EXPRS[g_upper]
        modules = _resolve_lambdify_modules(backend_str)
        fn = sympy.lambdify((_THETA_SYM, _QT_SYM), exprs, modules=modules)
        _LAMBDIFIED_GATE_FACTOR_CACHE[key] = fn
    return fn


def evaluate_sympy_with_lambdify(
    expr,
    variables: dict | None = None,
    lambdify_backend: str = DEFAULT_LAMBDIFY_BACKEND,
):
    """
    Evaluates a `sympy.Basic` expression using `sympy.lambdify(..., modules=...)`
    when all free symbols can be resolved from `variables` (or when `expr` has no free symbols).
    If unresolved free symbols remain, returns the substituted `sympy.Basic` expression.
    """
    if not isinstance(expr, sympy.Basic):
        return expr
    modules = _resolve_lambdify_modules(lambdify_backend)
    free_syms = sorted(expr.free_symbols, key=lambda s: str(s))
    if not free_syms:
        fn = sympy.lambdify((), expr, modules=modules)
        return fn()
    arg_vals = []
    for sym in free_syms:
        if variables is not None and sym in variables:
            arg_vals.append(variables[sym])
        elif variables is not None and str(sym) in variables:
            arg_vals.append(variables[str(sym)])
        else:
            return expr
    fn = sympy.lambdify(tuple(free_syms), expr, modules=modules)
    return fn(*arg_vals)


class BackendCircuitSymbolic(BackendCircuit):
    # compiler instructions
    compiler_arguments = {
        "trotterized": True,
        "swap": True,
        "multitarget": True,
        "controlled_rotation": False,
        "generalized_rotation": True,
        "exponential_pauli": True,
        "controlled_exponential_pauli": True,
        "phase": True,
        "power": True,
        "hadamard_power": True,
        "controlled_power": True,
        "controlled_phase": True,
        "toffoli": True,
        "phase_to_z": True,
        "cc_max": True,
    }

    convert_to_numpy = True
    use_lambdify = True
    lambdify_backend: str = DEFAULT_LAMBDIFY_BACKEND

    def __init__(self, abstract_circuit, variables=None, qubit_map=None, noise=None, device=None, *args, **kwargs):
        self.lambdify_backend = str(
            kwargs.pop("lambdify_backend", None) or self.__class__.lambdify_backend or DEFAULT_LAMBDIFY_BACKEND
        )
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
        return cls.lambdify_backend

    def create_circuit(self, abstract_circuit: QCircuit, variables=None):
        return abstract_circuit

    def update_variables(self, variables):
        pass

    @classmethod
    def apply_gate(
        cls,
        state: QubitWaveFunction,
        gate: QGate,
        qubits: dict,
        variables,
        use_lambdify: bool | None = None,
        lambdify_backend: str | None = None,
    ) -> QubitWaveFunction:
        n_qubits = len(qubits.keys())
        active_lambdify = cls.use_lambdify if use_lambdify is None else bool(use_lambdify)
        active_backend = str(lambdify_backend or cls.lambdify_backend or DEFAULT_LAMBDIFY_BACKEND)
        result = QubitWaveFunction(n_qubits) if active_lambdify else sympy.Integer(1) * QubitWaveFunction(n_qubits)
        for s, v in state.items():
            result += v * cls.apply_on_standard_basis(
                gate=gate,
                basis_state=s,
                qubits=qubits,
                variables=variables,
                use_lambdify=active_lambdify,
                lambdify_backend=active_backend,
            )
        return result

    @classmethod
    def apply_on_standard_basis(
        cls,
        gate: QGate,
        basis_state: BitString,
        qubits: dict,
        variables,
        use_lambdify: bool | None = None,
        lambdify_backend: str | None = None,
    ) -> QubitWaveFunction:
        n_qubits = len(qubits.keys())
        basis_array = basis_state.array
        if gate.is_controlled() and not all(basis_array[qubits[c]] == 1 for c in gate.control):
            return QubitWaveFunction.from_basis_state(n_qubits, basis_state)

        if len(gate.target) > 1:
            raise Exception("Multi-targets not supported for symbolic simulators")

        active_lambdify = cls.use_lambdify if use_lambdify is None else bool(use_lambdify)
        active_backend = str(lambdify_backend or cls.lambdify_backend or DEFAULT_LAMBDIFY_BACKEND)
        result = QubitWaveFunction(n_qubits) if active_lambdify else sympy.Integer(1) * QubitWaveFunction(n_qubits)
        for tt in gate.target:
            t = qubits[tt]
            qt = basis_array[t]
            a_array = copy.deepcopy(basis_array)
            a_array[t] = (a_array[t] + 1) % 2
            current_state = QubitWaveFunction.from_basis_state(n_qubits, basis_state)
            altered_state = QubitWaveFunction.from_basis_state(n_qubits, BitString.from_array(a_array))

            g_upper = "H" if gate.name == "H" else gate.name.upper()
            raw_angle = 0.0
            if g_upper in ("RX", "RY", "RZ"):
                raw_angle = gate.parameter(variables)
                if active_lambdify and isinstance(raw_angle, sympy.Basic):
                    raw_angle = evaluate_sympy_with_lambdify(
                        raw_angle, variables=variables, lambdify_backend=active_backend
                    )

            if active_lambdify and not isinstance(raw_angle, sympy.Basic):
                fn = get_lambdified_gate_factors(g_upper, lambdify_backend=active_backend)
                f1_val, f2_val = fn(float(raw_angle), int(qt))
                fac1 = complex(f1_val) if f1_val != 0 else None
                fac2 = complex(f2_val) if f2_val != 0 else None
            else:
                fac1 = None
                fac2 = None
                if gate.name == "H":
                    fac1 = sympy.Integer(-1) ** qt * sympy.sqrt(sympy.Rational(1 / 2))
                    fac2 = sympy.sqrt(sympy.Rational(1 / 2))
                elif g_upper == "CNOT" or g_upper == "X":
                    fac2 = sympy.Integer(1)
                elif g_upper == "Y":
                    fac2 = sympy.I * sympy.Integer(-1) ** (qt)
                elif g_upper == "Z":
                    fac1 = sympy.Integer(-1) ** (qt)
                elif g_upper == "RX":
                    angle = sympy.Rational(1 / 2) * raw_angle
                    fac1 = sympy.cos(angle)
                    fac2 = -sympy.sin(angle) * sympy.I
                elif g_upper == "RY":
                    angle = -sympy.Rational(1 / 2) * raw_angle
                    fac1 = sympy.cos(angle)
                    fac2 = +sympy.sin(angle) * sympy.Integer(-1) ** (qt + 1)
                elif g_upper == "RZ":
                    angle = sympy.Rational(1 / 2) * raw_angle
                    fac1 = sympy.exp(-angle * sympy.I * sympy.Integer(-1) ** (qt))
                else:
                    raise Exception("Gate is not known to simulators, " + str(gate))

            if fac1 is None or fac1 == 0:
                result += fac2 * altered_state
            elif fac2 is None or fac2 == 0:
                result += fac1 * current_state
            elif fac1 is None and fac2 is None:
                raise Exception("???")
            else:
                result += fac1 * current_state + fac2 * altered_state

        return result

    def do_simulate(self, variables, initial_state: int = None, *args, **kwargs) -> QubitWaveFunction:
        qubits = dict()
        count = 0
        for q in self.abstract_circuit.qubits:
            qubits[q] = count
            count += 1

        n_qubits = len(self.abstract_circuit.qubits)

        if initial_state is None:
            initial_state = 0
        initial_state = QubitWaveFunction.from_basis_state(n_qubits, initial_state)

        active_lambdify = bool(self.use_lambdify and self.convert_to_numpy)
        active_backend = str(
            kwargs.get("lambdify_backend", None) or self.lambdify_backend or DEFAULT_LAMBDIFY_BACKEND
        )

        result = initial_state
        for g in self.abstract_circuit.gates:
            result = self.apply_gate(
                state=result,
                gate=g,
                qubits=qubits,
                variables=variables,
                use_lambdify=active_lambdify,
                lambdify_backend=active_backend,
            )

        wfn = QubitWaveFunction(n_qubits)
        if self.convert_to_numpy:
            for k, v in result.items():
                if isinstance(v, sympy.Basic):
                    v = evaluate_sympy_with_lambdify(
                        v, variables=variables, lambdify_backend=active_backend
                    )
                wfn[k] = complex(v)
        else:
            wfn = result

        return wfn


class BackendExpectationValueSymbolic(BackendExpectationValue):
    BackendCircuitType = BackendCircuitSymbolic
