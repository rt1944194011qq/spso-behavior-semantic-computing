"""Deterministic compiler for the typed TSP GLS recipe language.

The same role renderers are used by ``compile`` and ``module_code``. The
second method therefore exposes executable compiler fragments rather than an
independent collection of illustrative snippets.
"""

from __future__ import annotations

import textwrap
from typing import Any, Mapping

from spso.models import ModuleChoice, ParticlePosition
from spso.registry import ModuleRegistry, ModuleSpec


class TSPGLSCompiler:
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry

    @staticmethod
    def _number(value: Any) -> str:
        return repr(float(value))

    @staticmethod
    def _kind(spec: ModuleSpec, fallback: str) -> str:
        return str(spec.recipe.get("kind", fallback))

    def _render_edge(self, spec: ModuleSpec, params: Mapping[str, Any]) -> str:
        kind = self._kind(spec, spec.module_id)
        if kind == "tour":
            return """    pairs = []
    for i in range(n):
        u = int(tour[i])
        v = int(tour[(i + 1) % n])
        if u != v:
            pairs.append((u, v))
"""
        if kind == "tour_neighbors":
            return """    pairs = []
    for i in range(n):
        u = int(tour[i])
        v = int(tour[(i + 1) % n])
        if u != v:
            pairs.append((u, v))
        nearest = int(np.argsort(base[u])[1]) if n > 1 else u
        if u != nearest:
            pairs.append((u, nearest))
"""
        if kind == "tour_k_neighbors":
            k = int(params["k"])
            return f"""    pairs = []
    for i in range(n):
        u = int(tour[i])
        v = int(tour[(i + 1) % n])
        if u != v:
            pairs.append((u, v))
        nearest_order = np.argsort(base[u])
        for rank in range(1, min({k + 1}, n)):
            neighbour = int(nearest_order[rank])
            if u != neighbour:
                pairs.append((u, neighbour))
"""
        if kind == "two_opt_cross":
            return """    pairs = []
    for i in range(n):
        a = int(tour[i])
        b = int(tour[(i + 1) % n])
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c = int(tour[j])
            d = int(tour[(j + 1) % n])
            pairs.append((a, c))
            pairs.append((b, d))
"""
        if kind in {"usage_topk", "tour_usage_mix"}:
            k = int(params["k"])
            prefix = "    pairs = []\n"
            if kind == "tour_usage_mix":
                prefix += """    for i in range(n):
        u = int(tour[i])
        v = int(tour[(i + 1) % n])
        if u != v:
            pairs.append((u, v))
"""
            return prefix + f"""    upper_rows, upper_cols = np.triu_indices(n, 1)
    usage_order = np.argsort(used[upper_rows, upper_cols])[::-1]
    for index in usage_order[:{k}]:
        if used[upper_rows[index], upper_cols[index]] > 0.0:
            pairs.append((int(upper_rows[index]), int(upper_cols[index])))
    if not pairs:
        for i in range(n):
            pairs.append((int(tour[i]), int(tour[(i + 1) % n])))
"""
        raise ValueError(f"unsupported E recipe: {kind}")

    def _render_feature(self, spec: ModuleSpec, params: Mapping[str, Any]) -> str:
        kind = self._kind(spec, spec.module_id)
        fixed = {
            "distance": "base[u, v]",
            "relative_distance": "base[u, v] / scale",
            "tour_relative": "base[u, v] / tour_scale",
            "inverse_distance": "scale / max(base[u, v], 1e-12)",
            "excess": "max(base[u, v] / scale - 1.0, 0.0)",
            "power_distance": "(base[u, v] / scale) ** 2",
        }
        if kind in fixed:
            return fixed[kind]
        if kind == "power_relative":
            return f"(base[u, v] / scale) ** {self._number(params['exponent'])}"
        if kind == "inverse_power":
            return f"(scale / max(base[u, v], 1e-12)) ** {self._number(params['exponent'])}"
        if kind == "offset_relative":
            return f"max(base[u, v] / scale + {self._number(params['offset'])}, 0.0)"
        if kind == "blend_relative":
            weight = self._number(params["weight"])
            return f"{weight} * (base[u, v] / scale) + (1.0 - {weight}) * (base[u, v] / tour_scale)"
        if kind == "bounded_relative":
            lower = self._number(params["lower"])
            upper = self._number(params["upper"])
            return f"min(max(base[u, v] / scale, {lower}), max({upper}, {lower}))"
        raise ValueError(f"unsupported F recipe: {kind}")

    def _render_history(self, spec: ModuleSpec, params: Mapping[str, Any]) -> str:
        kind = self._kind(spec, spec.module_id)
        if kind == "constant":
            return "1.0"
        if kind == "inverse_count":
            return "1.0 / (1.0 + used[u, v])"
        if kind == "inverse_sqrt":
            return "1.0 / np.sqrt(1.0 + used[u, v])"
        if kind == "inverse_log":
            return "1.0 / np.log2(2.0 + used[u, v])"
        if kind == "power_decay":
            return f"(1.0 + used[u, v]) ** (-{self._number(params['gamma'])})"
        if kind == "exponential_decay":
            return f"np.exp(-used[u, v] / {self._number(params['tau'])})"
        if kind == "usage_boost":
            return f"1.0 + {self._number(params['strength'])} * used[u, v] / (mean_used + 1e-12)"
        raise ValueError(f"unsupported H recipe: {kind}")

    def _render_transform(self, spec: ModuleSpec, params: Mapping[str, Any]) -> str:
        kind = self._kind(spec, spec.module_id)
        if kind == "identity":
            return "raw_score"
        if kind == "sqrt":
            return "np.sqrt(max(raw_score, 0.0))"
        if kind == "log1p":
            return "np.log1p(max(raw_score, 0.0))"
        if kind == "square":
            return "raw_score * raw_score"
        if kind == "tanh":
            return f"np.tanh(max(raw_score, 0.0) / {self._number(params['tau'])})"
        if kind == "clip":
            return f"min(max(raw_score, 0.0), {self._number(params['cap'])})"
        raise ValueError(f"unsupported T recipe: {kind}")

    def _render_write(self, spec: ModuleSpec, params: Mapping[str, Any]) -> str:
        kind = self._kind(spec, spec.module_id)
        lam = self._number(params.get("lambda", 1.0))
        if kind == "add_all":
            return f"""    for score, u, v in scored:
        delta = {lam} * score
        aug[u, v] += delta
        aug[v, u] += delta
"""
        if kind == "add_topk":
            return f"""    scored.sort(key=lambda item: item[0], reverse=True)
    for score, u, v in scored[:{int(params['k'])}]:
        delta = {lam} * score
        aug[u, v] += delta
        aug[v, u] += delta
"""
        if kind == "add_normalized":
            return f"""    max_score = max((item[0] for item in scored), default=0.0)
    if max_score > 0.0:
        for score, u, v in scored:
            delta = {lam} * score / max_score
            aug[u, v] += delta
            aug[v, u] += delta
"""
        if kind == "add_quantile":
            fraction = self._number(params["fraction"])
            return f"""    if scored:
        threshold = float(np.quantile([item[0] for item in scored], 1.0 - {fraction}))
        for score, u, v in scored:
            if score >= threshold:
                delta = {lam} * score
                aug[u, v] += delta
                aug[v, u] += delta
"""
        if kind == "rank_weighted":
            exponent = self._number(params["exponent"])
            return f"""    scored.sort(key=lambda item: item[0], reverse=True)
    for rank, (score, u, v) in enumerate(scored):
        delta = {lam} * score / ((rank + 1.0) ** {exponent})
        aug[u, v] += delta
        aug[v, u] += delta
"""
        if kind == "usage_balanced":
            return f"""    for score, u, v in scored:
        delta = {lam} * score / (1.0 + 0.5 * used[u, v])
        aug[u, v] += delta
        aug[v, u] += delta
"""
        raise ValueError(f"unsupported W recipe: {kind}")

    def _normalized_choice(self, slot_id: str, module_id: str, params=None) -> ModuleChoice:
        return self.registry.validate_choice(ModuleChoice.create(slot_id, module_id, params))

    def module_code(self, slot_id: str, module_id: str, params=None) -> str:
        """Return the exact typed fragment inserted by :meth:`compile`."""
        choice = self._normalized_choice(slot_id, module_id, params)
        spec = self.registry.module(slot_id, module_id)
        values = choice.params_dict()
        if slot_id == "E":
            fragment = self._render_edge(spec, values)
        elif slot_id == "F":
            fragment = f"raw_feature = {self._render_feature(spec, values)}"
        elif slot_id == "H":
            fragment = f"history_factor = {self._render_history(spec, values)}"
        elif slot_id == "T":
            fragment = f"score = {self._render_transform(spec, values)}"
        elif slot_id == "W":
            fragment = self._render_write(spec, values)
        else:
            raise ValueError(f"unknown slot: {slot_id}")
        return f"# slot {slot_id} / module {module_id}\n{textwrap.dedent(fragment).rstrip()}\n"

    def compile(self, position: ParticlePosition) -> tuple[str, str]:
        position = self.registry.validate_position(position)
        choices = position.as_mapping()
        e, f, h, t, w = (choices[slot] for slot in ("E", "F", "H", "T", "W"))
        e_spec = self.registry.module("E", e.module)
        f_spec = self.registry.module("F", f.module)
        h_spec = self.registry.module("H", h.module)
        t_spec = self.registry.module("T", t.module)
        w_spec = self.registry.module("W", w.module)
        edge_block = self._render_edge(e_spec, e.params_dict())
        feature_expr = self._render_feature(f_spec, f.params_dict())
        history_expr = self._render_history(h_spec, h.params_dict())
        transform_expr = self._render_transform(t_spec, t.params_dict())
        write_block = self._render_write(w_spec, w.params_dict())
        code = f'''import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Compiled TSP GLS heuristic with the public EoH signature."""
    base = np.asarray(edge_distance, dtype=float)
    tour = np.asarray(local_opt_tour, dtype=int).ravel()
    used = np.asarray(edge_n_used, dtype=float)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("edge_distance must be a square matrix")
    n = base.shape[0]
    if tour.size != n or used.shape != base.shape:
        raise ValueError("TSP GLS input shapes are inconsistent")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(used)):
        raise ValueError("TSP GLS inputs must be finite")
    if n == 0:
        return base.copy()
    if np.any(tour < 0) or np.any(tour >= n):
        raise ValueError("tour contains an invalid node")
    aug = base.copy()
    positive = base[np.isfinite(base) & (base > 0.0)]
    scale = float(np.mean(positive)) if positive.size else 1.0
    if scale <= 0.0:
        scale = 1.0
    mean_used = float(np.mean(used))
    tour_edges = [base[int(tour[i]), int(tour[(i + 1) % n])] for i in range(n)]
    tour_positive = [value for value in tour_edges if value > 0.0]
    tour_scale = float(np.mean(tour_positive)) if tour_positive else scale
{edge_block}    unique_pairs = []
    seen_pairs = set()
    for u, v in pairs:
        key = (min(u, v), max(u, v))
        if u != v and key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append(key)
    scored = []
    for u, v in unique_pairs:
        raw_score = max(0.0, {feature_expr}) * ({history_expr})
        score = max(0.0, {transform_expr})
        scored.append((float(score), u, v))
{write_block}    aug = 0.5 * (aug + aug.T)
    np.fill_diagonal(aug, 0.0)
    if not np.all(np.isfinite(aug)) or np.any(aug < 0.0):
        raise ValueError("compiled heuristic produced an invalid matrix")
    return aug
'''
        description = (
            f"candidate={e.module}; feature={f.module}; history={h.module}; "
            f"transform={t.module}; write={w.module}"
        )
        return code, description


def classic_gls_position(registry: ModuleRegistry) -> ParticlePosition:
    """The exact symbolic representation of the classic GLS penalty."""
    return registry.validate_position(ParticlePosition((
        ModuleChoice.create("E", "tour_edges"),
        ModuleChoice.create("F", "edge_length"),
        ModuleChoice.create("H", "inverse_count"),
        ModuleChoice.create("T", "identity"),
        ModuleChoice.create("W", "symmetric_add_all", {"lambda": 1.0}),
    )))
