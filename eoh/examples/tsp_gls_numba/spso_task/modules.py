"""Seed examples and typed recipes for the TSP GLS module library.

These modules are examples for the online LLM library builder. They are kept
in a separate prototype registry and are never silently counted as the ten
LLM-generated modules in an online run.
"""

from spso.registry import ModuleRegistry, ModuleSpec, ParameterSpec, SlotSpec


def _m(module_id, description, kind, *, parameters=(), tags=(), reverse_of=None,
       similar_to=(), family="", polarity="neutral"):
    return ModuleSpec(
        module_id, description, parameters=tuple(parameters), tags=tuple(tags),
        reverse_of=reverse_of, similar_to=tuple(similar_to),
        recipe={"kind": kind}, semantic_family=family, polarity=polarity,
    )


def build_tsp_gls_registry() -> ModuleRegistry:
    """Return the fixed prototype library used for prompts and legacy tests."""
    return ModuleRegistry((
        SlotSpec("E", "candidate edge generator; outputs undirected (u, v) pairs", (
            _m("tour_edges", "edges of the current local-optimal tour", "tour", tags=("baseline",), family="tour", polarity="neutral"),
            _m("tour_edges_plus_neighbors", "tour edges plus each tour endpoint's nearest neighbour", "tour_neighbors", tags=("exploration",), reverse_of="tour_edges", similar_to=("tour_edges", "tour_edges_k_neighbors"), family="tour_neighbors", polarity="broad"),
            _m("tour_edges_k_neighbors", "tour edges plus the k nearest neighbours of every tour node", "tour_k_neighbors", parameters=(ParameterSpec("k", "int", 3, minimum=1, maximum=10),), tags=("candidate_list",), similar_to=("tour_edges_plus_neighbors",), family="tour_neighbors", polarity="broad"),
            _m("two_opt_cross_edges", "edges introduced by non-adjacent 2-opt exchanges on the current tour", "two_opt_cross", tags=("move_aware",), similar_to=("tour_edges_plus_neighbors",), family="move_aware", polarity="broad"),
            _m("high_usage_edges", "highest-history edges, with a tour-edge fallback when history is empty", "usage_topk", parameters=(ParameterSpec("k", "int", 5, minimum=1, maximum=50),), tags=("history_aware",), reverse_of="tour_edges", similar_to=("tour_and_high_usage",), family="usage", polarity="history_high"),
            _m("tour_and_high_usage", "current tour edges together with the highest-history edges", "tour_usage_mix", parameters=(ParameterSpec("k", "int", 5, minimum=1, maximum=50),), tags=("history_aware",), similar_to=("tour_edges", "high_usage_edges"), family="usage", polarity="mixed"),
        ), max_cut=4, set_capacity=3),
        SlotSpec("F", "positive edge feature used as the base penalty", (
            _m("edge_length", "original distance d(u,v)", "distance", tags=("baseline",), similar_to=("relative_length", "tour_deviation"), family="distance", polarity="long"),
            _m("relative_length", "distance divided by the positive matrix mean", "relative_distance", tags=("scale_invariant",), reverse_of="edge_length", similar_to=("edge_length", "tour_deviation"), family="distance", polarity="long"),
            _m("tour_deviation", "distance divided by the current tour mean edge length", "tour_relative", tags=("tour_aware",), similar_to=("relative_length", "edge_length"), family="distance", polarity="long"),
            _m("inverse_length", "matrix-scale divided by distance, favouring short edges", "inverse_distance", tags=("reverse_preference",), reverse_of="edge_length", similar_to=("relative_length",), family="distance", polarity="short"),
            _m("inverse_power_length", "raise the inverse normalized distance to a bounded exponent", "inverse_power", parameters=(ParameterSpec("exponent", "float", 1.5, minimum=0.25, maximum=3.0),), tags=("reverse_preference", "parameterized"), similar_to=("inverse_length",), family="distance_power", polarity="short"),
            _m("excess_over_mean", "positive part of distance divided by matrix mean minus one", "excess", tags=("outlier_aware",), similar_to=("relative_length", "squared_relative_length"), family="outlier", polarity="long"),
            _m("squared_relative_length", "square of distance divided by matrix mean", "power_distance", tags=("outlier_aware",), similar_to=("relative_length", "excess_over_mean"), family="outlier", polarity="amplify"),
            _m("power_relative_length", "raise normalized distance to a bounded exponent", "power_relative", parameters=(ParameterSpec("exponent", "float", 1.5, minimum=0.25, maximum=3.0),), tags=("parameterized",), similar_to=("squared_relative_length", "relative_length"), family="distance_power", polarity="amplify"),
            _m("offset_relative_length", "add a bounded nonnegative offset to normalized distance", "offset_relative", parameters=(ParameterSpec("offset", "float", 0.25, minimum=0.0, maximum=2.0),), tags=("parameterized",), similar_to=("relative_length", "excess_over_mean"), family="distance_offset", polarity="long"),
            _m("blended_length", "blend matrix-relative and tour-relative edge length", "blend_relative", parameters=(ParameterSpec("weight", "float", 0.5, minimum=0.0, maximum=1.0),), tags=("parameterized",), similar_to=("relative_length", "tour_deviation"), family="distance_blend", polarity="long"),
            _m("bounded_relative_length", "clamp normalized distance to bounded lower and upper scales", "bounded_relative", parameters=(ParameterSpec("lower", "float", 0.0, minimum=0.0, maximum=2.0), ParameterSpec("upper", "float", 2.0, minimum=0.1, maximum=5.0)), tags=("parameterized",), similar_to=("relative_length", "excess_over_mean"), family="distance_bound", polarity="bound"),
        ), max_cut=4, set_capacity=3),
        SlotSpec("H", "history factor based on edge_n_used", (
            _m("none", "ignore previous penalty count", "constant", tags=("aggressive",), reverse_of="inverse_count", similar_to=("inverse_count", "usage_boost"), family="history", polarity="neutral"),
            _m("inverse_count", "multiply by 1/(1+count), the classic GLS factor", "inverse_count", tags=("baseline",), reverse_of="none", similar_to=("inverse_sqrt_count", "inverse_log_count"), family="history", polarity="decay"),
            _m("inverse_sqrt_count", "multiply by 1/sqrt(1+count)", "inverse_sqrt", similar_to=("inverse_count", "inverse_log_count", "power_decay"), family="history", polarity="decay"),
            _m("inverse_log_count", "multiply by 1/log2(2+count)", "inverse_log", similar_to=("inverse_count", "inverse_sqrt_count", "exponential_decay"), family="history", polarity="decay"),
            _m("power_decay", "multiply by (1+count)^(-gamma)", "power_decay", parameters=(ParameterSpec("gamma", "float", 0.5, minimum=0.1, maximum=2.0),), similar_to=("inverse_sqrt_count", "inverse_count"), family="history", polarity="decay"),
            _m("exponential_decay", "multiply by exp(-count/tau)", "exponential_decay", parameters=(ParameterSpec("tau", "float", 3.0, minimum=0.5, maximum=20.0),), similar_to=("inverse_log_count", "inverse_count"), family="history", polarity="decay"),
            _m("usage_boost", "increase the score of repeatedly used edges", "usage_boost", parameters=(ParameterSpec("strength", "float", 0.5, minimum=0.0, maximum=2.0),), tags=("diversification",), reverse_of="inverse_count", similar_to=("none",), family="history", polarity="boost"),
        ), max_cut=4, set_capacity=3),
        SlotSpec("T", "nonnegative score transformation", (
            _m("identity", "keep the penalty score unchanged", "identity", tags=("baseline",), reverse_of="square", similar_to=("sqrt", "log1p"), family="transform", polarity="linear"),
            _m("sqrt", "take the square root of the positive score", "sqrt", similar_to=("identity", "log1p"), family="transform", polarity="compress"),
            _m("log1p", "use log(1+score) to compress large penalties", "log1p", similar_to=("identity", "sqrt", "tanh_scale"), family="transform", polarity="compress"),
            _m("square", "square the score to emphasise large penalties", "square", reverse_of="identity", similar_to=("identity", "tanh_scale"), family="transform", polarity="amplify"),
            _m("tanh_scale", "use tanh(score/tau) to bound very large penalties", "tanh", parameters=(ParameterSpec("tau", "float", 1.0, minimum=0.1, maximum=10.0),), similar_to=("log1p", "clip"), family="transform", polarity="compress"),
            _m("clip", "cap the score at a fixed scale", "clip", parameters=(ParameterSpec("cap", "float", 2.0, minimum=0.1, maximum=20.0),), similar_to=("tanh_scale", "identity"), family="transform", polarity="bound"),
        ), max_cut=4, set_capacity=3),
        SlotSpec("W", "write the nonnegative score back to the symmetric matrix", (
            _m("symmetric_add_all", "add every candidate score to both matrix directions", "add_all", parameters=(ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0),), tags=("baseline",), similar_to=("symmetric_add_topk", "symmetric_add_quantile"), family="write", polarity="dense"),
            _m("symmetric_add_topk", "add only the top k candidate scores symmetrically", "add_topk", parameters=(ParameterSpec("k", "int", 5, minimum=1, maximum=50), ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0)), similar_to=("symmetric_add_all", "symmetric_add_quantile"), family="write", polarity="sparse"),
            _m("symmetric_add_normalized", "normalize candidate scores by their maximum before writing", "add_normalized", parameters=(ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0),), tags=("scale_invariant",), similar_to=("symmetric_add_all",), family="write", polarity="normalized"),
            _m("symmetric_add_quantile", "write candidates above a score quantile", "add_quantile", parameters=(ParameterSpec("fraction", "float", 0.5, minimum=0.1, maximum=1.0), ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0)), tags=("adaptive_sparsity",), similar_to=("symmetric_add_topk", "symmetric_add_all"), family="write", polarity="sparse"),
            _m("symmetric_add_rank_weighted", "write all candidates with a rank-decaying weight", "rank_weighted", parameters=(ParameterSpec("exponent", "float", 0.5, minimum=0.0, maximum=2.0), ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0)), tags=("smooth_sparsity",), similar_to=("symmetric_add_all", "symmetric_add_topk"), family="write", polarity="ranked"),
            _m("symmetric_add_usage_balanced", "write scores after an additional usage balance factor", "usage_balanced", parameters=(ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0),), tags=("history_aware",), similar_to=("symmetric_add_all",), family="write", polarity="balanced"),
        ), max_cut=4, set_capacity=3),
    ))
