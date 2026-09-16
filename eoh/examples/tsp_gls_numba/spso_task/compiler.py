"""Deterministic compiler from a TSP GLS module sequence to Python source."""

from __future__ import annotations

from spso.models import ModuleChoice, ParticlePosition
from spso.registry import ModuleRegistry


class TSPGLSCompiler:
    """Compile symbolic modules into the exact EoH public function contract."""

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry

    @staticmethod
    def _number(value) -> str:
        return repr(float(value)) if isinstance(value, float) else repr(value)

    def module_code(self, slot_id: str, module_id: str, params=None) -> str:
        """Return the exact internal snippet represented by one module.

        These snippets are explanatory compiler fragments sent to the LLM. They
        use the adapter's internal context variables (``base``, ``tour``,
        ``used``, ``pairs``, ``scored`` and ``aug``), while the compiler still
        emits the final complete function deterministically.
        """
        choice = self.registry.validate_choice(
            ModuleChoice.create(slot_id, module_id, params or {})
        )
        values = choice.params_dict()
        e_k = int(values.get("k", 3))
        h_gamma = self._number(values.get("gamma", 0.5))
        h_tau = self._number(values.get("tau", 3.0))
        h_strength = self._number(values.get("strength", 0.5))
        t_tau = self._number(values.get("tau", 1.0))
        t_cap = self._number(values.get("cap", 2.0))
        w_k = int(values.get("k", 5))
        w_fraction = self._number(values.get("fraction", 0.5))
        w_exponent = self._number(values.get("exponent", 0.5))
        w_lambda = self._number(values.get("lambda", 1.0))
        if slot_id == "E":
            snippets = {
                "tour_edges": """pairs = []
for i in range(n):
    u = int(tour[i]); v = int(tour[(i + 1) % n])
    if u != v: pairs.append((u, v))""",
                "tour_edges_plus_neighbors": """pairs = []
for i in range(n):
    u = int(tour[i]); v = int(tour[(i + 1) % n])
    if u != v: pairs.append((u, v))
    nearest = int(np.argsort(base[u])[1]) if n > 1 else u
    if u != nearest: pairs.append((u, nearest))""",
                "tour_edges_k_neighbors": f"""pairs = []
for i in range(n):
    u = int(tour[i]); v = int(tour[(i + 1) % n])
    if u != v: pairs.append((u, v))
    nearest_order = np.argsort(base[u])
    for rank in range(1, min({e_k + 1}, n)):
        pairs.append((u, int(nearest_order[rank])))""",
                "two_opt_cross_edges": """pairs = []
for i in range(n):
    a = int(tour[i]); b = int(tour[(i + 1) % n])
    for j in range(i + 2, n):
        if i == 0 and j == n - 1: continue
        c = int(tour[j]); d = int(tour[(j + 1) % n])
        pairs.extend([(a, c), (b, d)])""",
                "high_usage_edges": f"""pairs = []
rows, cols = np.triu_indices(n, 1)
 for index in np.argsort(used[rows, cols])[::-1][:{e_k}]:
    if used[rows[index], cols[index]] > 0: pairs.append((int(rows[index]), int(cols[index])))""",
                "tour_and_high_usage": f"""pairs = []
for i in range(n): pairs.append((int(tour[i]), int(tour[(i + 1) % n])))
rows, cols = np.triu_indices(n, 1)
 for index in np.argsort(used[rows, cols])[::-1][:{e_k}]:
    if used[rows[index], cols[index]] > 0: pairs.append((int(rows[index]), int(cols[index])))""",
            }
            return snippets[module_id]
        if slot_id == "F":
            return {
                "edge_length": "raw_feature = base[u, v]",
                "relative_length": "raw_feature = base[u, v] / scale",
                "tour_deviation": "raw_feature = base[u, v] / tour_scale",
                "inverse_length": "raw_feature = scale / max(base[u, v], 1e-12)",
                "excess_over_mean": "raw_feature = max(base[u, v] / scale - 1.0, 0.0)",
                "squared_relative_length": "raw_feature = (base[u, v] / scale) ** 2",
            }[module_id]
        if slot_id == "H":
            return {
                "none": "history_factor = 1.0",
                "inverse_count": "history_factor = 1.0 / (1.0 + used[u, v])",
                "inverse_sqrt_count": "history_factor = 1.0 / np.sqrt(1.0 + used[u, v])",
                "inverse_log_count": "history_factor = 1.0 / np.log2(2.0 + used[u, v])",
                "power_decay": f"history_factor = (1.0 + used[u, v]) ** (-{h_gamma})",
                "exponential_decay": f"history_factor = np.exp(-used[u, v] / {h_tau})",
                "usage_boost": f"history_factor = 1.0 + {h_strength} * used[u, v] / (mean_used + 1e-12)",
            }[module_id]
        if slot_id == "T":
            return {
                "identity": "score = raw_score",
                "sqrt": "score = np.sqrt(max(raw_score, 0.0))",
                "log1p": "score = np.log1p(max(raw_score, 0.0))",
                "square": "score = raw_score * raw_score",
                "tanh_scale": f"score = np.tanh(max(raw_score, 0.0) / {t_tau})",
                "clip": f"score = min(max(raw_score, 0.0), {t_cap})",
            }[module_id]
        if slot_id == "W":
            lam = w_lambda
            return {
                "symmetric_add_all": f"""for score, u, v in scored:
    delta = {lam} * score
    aug[u, v] += delta; aug[v, u] += delta""",
                "symmetric_add_topk": f"""scored.sort(key=lambda item: item[0], reverse=True)
for score, u, v in scored[:{w_k}]:
    delta = {lam} * score
    aug[u, v] += delta; aug[v, u] += delta""",
                "symmetric_add_normalized": f"""max_score = max((item[0] for item in scored), default=0.0)
for score, u, v in scored:
    delta = {lam} * score / max(max_score, 1e-12)
    aug[u, v] += delta; aug[v, u] += delta""",
                "symmetric_add_quantile": f"""threshold = np.quantile([item[0] for item in scored], 1.0 - {w_fraction})
for score, u, v in scored:
    if score >= threshold:
        delta = {lam} * score
        aug[u, v] += delta; aug[v, u] += delta""",
                "symmetric_add_rank_weighted": f"""scored.sort(key=lambda item: item[0], reverse=True)
for rank, (score, u, v) in enumerate(scored):
    delta = {lam} * score / ((rank + 1.0) ** {w_exponent})
    aug[u, v] += delta; aug[v, u] += delta""",
                "symmetric_add_usage_balanced": f"""for score, u, v in scored:
    delta = {lam} * score / (1.0 + 0.5 * used[u, v])
    aug[u, v] += delta; aug[v, u] += delta""",
            }[module_id]
        raise ValueError(f"unknown slot: {slot_id}")

    def compile(self, position: ParticlePosition) -> tuple[str, str]:
        position = self.registry.validate_position(position)
        choices = position.as_mapping()
        e = choices["E"]
        f = choices["F"].module
        h = choices["H"]
        t = choices["T"]
        w = choices["W"]
        ep = e.params_dict()
        hp = h.params_dict()
        tp = t.params_dict()
        wp = w.params_dict()

        if e.module == "tour_edges":
            edge_block = """    pairs = []
    for i in range(n):
        u = int(tour[i])
        v = int(tour[(i + 1) % n])
        if u != v:
            pairs.append((u, v))
"""
        elif e.module == "tour_edges_plus_neighbors":
            edge_block = """    pairs = []
    for i in range(n):
        u = int(tour[i])
        v = int(tour[(i + 1) % n])
        if u != v:
            pairs.append((u, v))
        nearest = int(np.argsort(base[u])[1]) if n > 1 else u
        if u != nearest:
            pairs.append((u, nearest))
"""
        elif e.module == "tour_edges_k_neighbors":
            k = int(ep["k"])
            edge_block = f"""    pairs = []
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
        elif e.module == "two_opt_cross_edges":
            edge_block = """    pairs = []
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
        elif e.module in {"high_usage_edges", "tour_and_high_usage"}:
            k = int(ep["k"])
            prefix = """    pairs = []
"""
            if e.module == "tour_and_high_usage":
                prefix += """    for i in range(n):
        u = int(tour[i])
        v = int(tour[(i + 1) % n])
        if u != v:
            pairs.append((u, v))
"""
            edge_block = prefix + f"""    upper_rows, upper_cols = np.triu_indices(n, 1)
    usage_order = np.argsort(used[upper_rows, upper_cols])[::-1]
    for index in usage_order[:{k}]:
        if used[upper_rows[index], upper_cols[index]] > 0.0:
            pairs.append((int(upper_rows[index]), int(upper_cols[index])))
    if not pairs:
        for i in range(n):
            pairs.append((int(tour[i]), int(tour[(i + 1) % n])))
"""
        else:  # registry validation makes this unreachable
            raise ValueError(f"unsupported E module: {e.module}")

        if f == "edge_length":
            feature_expr = "base[u, v]"
        elif f == "relative_length":
            feature_expr = "base[u, v] / scale"
        elif f == "tour_deviation":
            feature_expr = "base[u, v] / tour_scale"
        elif f == "inverse_length":
            feature_expr = "scale / max(base[u, v], 1e-12)"
        elif f == "excess_over_mean":
            feature_expr = "max(base[u, v] / scale - 1.0, 0.0)"
        elif f == "squared_relative_length":
            feature_expr = "(base[u, v] / scale) ** 2"
        else:
            raise ValueError(f"unsupported F module: {f}")

        if h.module == "none":
            history_expr = "1.0"
        elif h.module == "inverse_count":
            history_expr = "1.0 / (1.0 + used[u, v])"
        elif h.module == "inverse_sqrt_count":
            history_expr = "1.0 / np.sqrt(1.0 + used[u, v])"
        elif h.module == "inverse_log_count":
            history_expr = "1.0 / np.log2(2.0 + used[u, v])"
        elif h.module == "power_decay":
            history_expr = f"(1.0 + used[u, v]) ** (-{self._number(hp['gamma'])})"
        elif h.module == "exponential_decay":
            history_expr = f"np.exp(-used[u, v] / {self._number(hp['tau'])})"
        elif h.module == "usage_boost":
            history_expr = f"1.0 + {self._number(hp['strength'])} * used[u, v] / (mean_used + 1e-12)"
        else:
            raise ValueError(f"unsupported H module: {h.module}")

        if t.module == "identity":
            transform_expr = "raw_score"
        elif t.module == "sqrt":
            transform_expr = "np.sqrt(max(raw_score, 0.0))"
        elif t.module == "log1p":
            transform_expr = "np.log1p(max(raw_score, 0.0))"
        elif t.module == "square":
            transform_expr = "raw_score * raw_score"
        elif t.module == "tanh_scale":
            transform_expr = f"np.tanh(max(raw_score, 0.0) / {self._number(tp['tau'])})"
        elif t.module == "clip":
            transform_expr = f"min(max(raw_score, 0.0), {self._number(tp['cap'])})"
        else:
            raise ValueError(f"unsupported T module: {t.module}")

        lambda_value = self._number(wp.get("lambda", 1.0))
        if w.module == "symmetric_add_all":
            write_block = f"""    for score, u, v in scored:
        delta = {lambda_value} * score
        aug[u, v] += delta
        aug[v, u] += delta
"""
        elif w.module == "symmetric_add_topk":
            k_value = int(wp["k"])
            write_block = f"""    scored.sort(key=lambda item: item[0], reverse=True)
    for score, u, v in scored[:{k_value}]:
        delta = {lambda_value} * score
        aug[u, v] += delta
        aug[v, u] += delta
"""
        elif w.module == "symmetric_add_normalized":
            write_block = f"""    max_score = max((item[0] for item in scored), default=0.0)
    if max_score > 0.0:
        for score, u, v in scored:
            delta = {lambda_value} * score / max_score
            aug[u, v] += delta
            aug[v, u] += delta
"""
        elif w.module == "symmetric_add_quantile":
            fraction = self._number(wp["fraction"])
            write_block = f"""    if scored:
        threshold = float(np.quantile([item[0] for item in scored], 1.0 - {fraction}))
        for score, u, v in scored:
            if score >= threshold:
                delta = {lambda_value} * score
                aug[u, v] += delta
                aug[v, u] += delta
"""
        elif w.module == "symmetric_add_rank_weighted":
            exponent = self._number(wp["exponent"])
            write_block = f"""    scored.sort(key=lambda item: item[0], reverse=True)
    for rank, (score, u, v) in enumerate(scored):
        delta = {lambda_value} * score / ((rank + 1.0) ** {exponent})
        aug[u, v] += delta
        aug[v, u] += delta
"""
        elif w.module == "symmetric_add_usage_balanced":
            write_block = f"""    for score, u, v in scored:
        delta = {lambda_value} * score / (1.0 + 0.5 * used[u, v])
        aug[u, v] += delta
        aug[v, u] += delta
"""
        else:
            raise ValueError(f"unsupported W module: {w.module}")

        code = f'''import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Compiled S-PSO TSP GLS heuristic.

    Contract: edge_distance and edge_n_used are n-by-n matrices; local_opt_tour
    is a length-n tour; the returned value is a new finite symmetric n-by-n
    distance matrix.
    """
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
            f"candidate={e.module}; feature={f}; history={h.module}; "
            f"transform={t.module}; write={w.module}({wp})"
        )
        return code, description


def classic_gls_position(registry: ModuleRegistry) -> ParticlePosition:
    """The exact symbolic representation of the classic GLS penalty."""
    return registry.validate_position(
        ParticlePosition(
            (
                ModuleChoice.create("E", "tour_edges"),
                ModuleChoice.create("F", "edge_length"),
                ModuleChoice.create("H", "inverse_count"),
                ModuleChoice.create("T", "identity"),
                ModuleChoice.create("W", "symmetric_add_all", {"lambda": 1.0}),
            )
        )
    )
