import numpy as np

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
    pairs = []
    for i in range(n):
        u = int(tour[i])
        v = int(tour[(i + 1) % n])
        if u != v:
            pairs.append((u, v))
        nearest = int(np.argsort(base[u])[1]) if n > 1 else u
        if u != nearest:
            pairs.append((u, nearest))
    unique_pairs = []
    seen_pairs = set()
    for u, v in pairs:
        key = (min(u, v), max(u, v))
        if u != v and key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append(key)
    scored = []
    for u, v in unique_pairs:
        raw_score = max(0.0, base[u, v] / scale) * (1.0 / np.sqrt(1.0 + used[u, v]))
        score = max(0.0, np.log1p(max(raw_score, 0.0)))
        scored.append((float(score), u, v))
    max_score = max((item[0] for item in scored), default=0.0)
    if max_score > 0.0:
        for score, u, v in scored:
            delta = 1.0 * score / max_score
            aug[u, v] += delta
            aug[v, u] += delta
    aug = 0.5 * (aug + aug.T)
    np.fill_diagonal(aug, 0.0)
    if not np.all(np.isfinite(aug)) or np.any(aug < 0.0):
        raise ValueError("compiled heuristic produced an invalid matrix")
    return aug
