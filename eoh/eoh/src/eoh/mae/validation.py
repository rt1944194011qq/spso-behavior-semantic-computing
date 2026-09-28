from __future__ import annotations

import ast
import copy
import hashlib
import types
from collections.abc import Callable

import numpy as np

from ..problem import _get_entry_name
from .models import ValidationResult


class _NumericConstantNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=0), node)
        return node


def structural_fingerprint(code: str, ignore_numeric_constants: bool = True) -> str:
    """Hash AST structure, optionally treating every numeric constant as equal."""
    tree = ast.parse(code)
    if ignore_numeric_constants:
        tree = _NumericConstantNormalizer().visit(tree)
        ast.fix_missing_locations(tree)
    dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def parameter_only_mutation(code: str, seed: str = "") -> str | None:
    """Create an H5 fallback by changing an existing floating-point literal only.

    H5 is intentionally parameter-only.  LLMs sometimes return a valid program
    but change its AST structure anyway.  This deterministic fallback keeps the
    parent's structure and changes one existing float, so the candidate remains
    a genuine parameter variation instead of being discarded.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    numeric_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and not isinstance(node.value, bool)
    ]
    if not numeric_nodes:
        return None

    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    node = numeric_nodes[int.from_bytes(digest[:4], "big") % len(numeric_nodes)]
    old_value = float(node.value)

    # A bounded deterministic perturbation avoids changing the algorithm's
    # scale too aggressively while also ensuring the result differs from the
    # parent.  Zero needs a non-zero replacement to avoid duplicate code.
    fraction = int.from_bytes(digest[4:6], "big") / 65535.0
    factor = 0.85 + 0.30 * fraction
    if abs(old_value) < 1e-12:
        node.value = 0.05 if digest[6] % 2 == 0 else -0.05
    else:
        node.value = old_value * factor

    try:
        repaired = ast.unparse(tree).strip()
        ast.parse(repaired)
    except (AttributeError, SyntaxError):
        return None
    return repaired + "\n"


class ProgramValidator:
    """Validate syntax, signature and an optional task-specific smoke test."""

    def __init__(
        self,
        template_program: str,
        smoke_test: Callable[[Callable], None] | None = None,
    ) -> None:
        self.template_program = template_program
        self.entry_name = _get_entry_name(template_program)
        self.smoke_test = smoke_test
        self._expected_args = self._function_args(template_program, self.entry_name)

    @staticmethod
    def _function_args(source: str, name: str) -> list[str]:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return [arg.arg for arg in node.args.args]
        return []

    def validate(self, code: str) -> ValidationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return ValidationResult(False, "syntax_error", str(exc))

        function = next(
            (
                node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == self.entry_name
            ),
            None,
        )
        if function is None:
            return ValidationResult(False, "signature_error", f"missing {self.entry_name}")
        actual_args = [arg.arg for arg in function.args.args]
        if actual_args != self._expected_args:
            return ValidationResult(
                False,
                "signature_error",
                f"expected args {self._expected_args}, got {actual_args}",
            )

        if self.smoke_test is not None:
            try:
                module = types.ModuleType("mae_candidate")
                module.__dict__["np"] = np
                exec(compile(tree, "<mae-candidate>", "exec"), module.__dict__)
                function_obj = getattr(module, self.entry_name)
                self.smoke_test(function_obj)
            except AssertionError as exc:
                return ValidationResult(False, "contract_shape", str(exc))
            except Exception as exc:
                return ValidationResult(False, "smoke_test", f"{type(exc).__name__}: {exc}")

        return ValidationResult(True)


def make_tsp_smoke_test(
    require_symmetric: bool = True,
    require_nonnegative: bool = True,
) -> Callable[[Callable], None]:
    def smoke_test(function: Callable) -> None:
        distance = np.array(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.5, 2.5],
             [2.0, 1.5, 0.0, 1.0], [3.0, 2.5, 1.0, 0.0]],
            dtype=float,
        )
        tour = np.array([0, 1, 2, 3], dtype=int)
        used = np.zeros_like(distance)
        before_distance = copy.deepcopy(distance)
        before_tour = copy.deepcopy(tour)
        before_used = copy.deepcopy(used)

        result = np.asarray(function(distance, tour, used), dtype=float)
        assert np.array_equal(distance, before_distance), "edge_distance was modified in place"
        assert np.array_equal(tour, before_tour), "local_opt_tour was modified in place"
        assert np.array_equal(used, before_used), "edge_n_used was modified in place"
        assert result.shape == distance.shape, f"expected shape {distance.shape}, got {result.shape}"
        assert np.all(np.isfinite(result)), "output contains NaN or Inf"
        if require_nonnegative:
            assert np.all(result >= 0), "output contains negative distances"
        if require_symmetric:
            assert np.allclose(result, result.T), "output is not symmetric"

    return smoke_test
