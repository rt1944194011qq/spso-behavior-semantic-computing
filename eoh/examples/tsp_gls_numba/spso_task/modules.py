"""Typed module library for the TSP GLS distance-update contract.

The library is broader than the first smoke-test library. Every entry has a
deterministic implementation in ``compiler.py`` and a bounded parameter
domain, so adding a module never creates an unexecutable candidate.
"""

from spso.registry import ModuleRegistry, ModuleSpec, ParameterSpec, SlotSpec


def build_tsp_gls_registry() -> ModuleRegistry:
    return ModuleRegistry(
        (
            SlotSpec(
                "E",
                "candidate edge generator; outputs undirected (u, v) pairs",
                (
                    ModuleSpec("tour_edges", "edges of the current local-optimal tour", tags=("baseline",), similar_to=("tour_edges_plus_neighbors", "tour_edges_k_neighbors")),
                    ModuleSpec("tour_edges_plus_neighbors", "tour edges plus each tour endpoint's nearest neighbour", tags=("exploration",), reverse_of="tour_edges", similar_to=("tour_edges", "tour_edges_k_neighbors")),
                    ModuleSpec("tour_edges_k_neighbors", "tour edges plus the k nearest neighbours of every tour node", parameters=(ParameterSpec("k", "int", 3, minimum=1, maximum=10),), tags=("candidate_list",), similar_to=("tour_edges_plus_neighbors",)),
                    ModuleSpec("two_opt_cross_edges", "edges introduced by non-adjacent 2-opt exchanges on the current tour", tags=("move_aware",), similar_to=("tour_edges_plus_neighbors",)),
                    ModuleSpec("high_usage_edges", "highest-history edges, with a tour-edge fallback when history is empty", parameters=(ParameterSpec("k", "int", 5, minimum=1, maximum=50),), tags=("history_aware",), reverse_of="tour_edges", similar_to=("tour_and_high_usage",)),
                    ModuleSpec("tour_and_high_usage", "current tour edges together with the highest-history edges", parameters=(ParameterSpec("k", "int", 5, minimum=1, maximum=50),), tags=("history_aware",), similar_to=("tour_edges", "high_usage_edges")),
                ),
                max_cut=4,
            ),
            SlotSpec(
                "F",
                "positive edge feature used as the base penalty",
                (
                    ModuleSpec("edge_length", "original distance d(u,v)", tags=("baseline",), similar_to=("relative_length", "tour_deviation")),
                    ModuleSpec("relative_length", "distance divided by the positive matrix mean", tags=("scale_invariant",), reverse_of="edge_length", similar_to=("edge_length", "tour_deviation")),
                    ModuleSpec("tour_deviation", "distance divided by the current tour mean edge length", tags=("tour_aware",), similar_to=("relative_length", "edge_length")),
                    ModuleSpec("inverse_length", "matrix-scale divided by distance, favouring short edges", tags=("reverse_preference",), reverse_of="edge_length", similar_to=("relative_length",)),
                    ModuleSpec("excess_over_mean", "positive part of distance divided by matrix mean minus one", tags=("outlier_aware",), similar_to=("relative_length", "squared_relative_length")),
                    ModuleSpec("squared_relative_length", "square of distance divided by matrix mean", tags=("outlier_aware",), similar_to=("relative_length", "excess_over_mean")),
                ),
                max_cut=4,
            ),
            SlotSpec(
                "H",
                "history factor based on edge_n_used",
                (
                    ModuleSpec("none", "ignore previous penalty count", tags=("aggressive",), reverse_of="inverse_count", similar_to=("inverse_count", "usage_boost")),
                    ModuleSpec("inverse_count", "multiply by 1/(1+count), the classic GLS factor", tags=("baseline",), reverse_of="none", similar_to=("inverse_sqrt_count", "inverse_log_count")),
                    ModuleSpec("inverse_sqrt_count", "multiply by 1/sqrt(1+count)", similar_to=("inverse_count", "inverse_log_count", "power_decay")),
                    ModuleSpec("inverse_log_count", "multiply by 1/log2(2+count)", similar_to=("inverse_count", "inverse_sqrt_count", "exponential_decay")),
                    ModuleSpec("power_decay", "multiply by (1+count)^(-gamma)", parameters=(ParameterSpec("gamma", "float", 0.5, minimum=0.1, maximum=2.0),), similar_to=("inverse_sqrt_count", "inverse_count")),
                    ModuleSpec("exponential_decay", "multiply by exp(-count/tau)", parameters=(ParameterSpec("tau", "float", 3.0, minimum=0.5, maximum=20.0),), similar_to=("inverse_log_count", "inverse_count")),
                    ModuleSpec("usage_boost", "increase the score of repeatedly used edges", parameters=(ParameterSpec("strength", "float", 0.5, minimum=0.0, maximum=2.0),), tags=("diversification",), reverse_of="inverse_count", similar_to=("none",)),
                ),
                max_cut=4,
            ),
            SlotSpec(
                "T",
                "nonnegative score transformation",
                (
                    ModuleSpec("identity", "keep the penalty score unchanged", tags=("baseline",), reverse_of="square", similar_to=("sqrt", "log1p")),
                    ModuleSpec("sqrt", "take the square root of the positive score", similar_to=("identity", "log1p")),
                    ModuleSpec("log1p", "use log(1+score) to compress large penalties", similar_to=("identity", "sqrt", "tanh_scale")),
                    ModuleSpec("square", "square the score to emphasise large penalties", reverse_of="identity", similar_to=("identity", "tanh_scale")),
                    ModuleSpec("tanh_scale", "use tanh(score/tau) to bound very large penalties", parameters=(ParameterSpec("tau", "float", 1.0, minimum=0.1, maximum=10.0),), similar_to=("log1p", "clip")),
                    ModuleSpec("clip", "cap the score at a fixed scale", parameters=(ParameterSpec("cap", "float", 2.0, minimum=0.1, maximum=20.0),), similar_to=("tanh_scale", "identity")),
                ),
                max_cut=4,
            ),
            SlotSpec(
                "W",
                "write the nonnegative score back to the symmetric matrix",
                (
                    ModuleSpec("symmetric_add_all", "add every candidate score to both matrix directions", parameters=(ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0),), tags=("baseline",), similar_to=("symmetric_add_topk", "symmetric_add_quantile")),
                    ModuleSpec("symmetric_add_topk", "add only the top k candidate scores symmetrically", parameters=(ParameterSpec("k", "int", 5, minimum=1, maximum=50), ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0)), similar_to=("symmetric_add_all", "symmetric_add_quantile")),
                    ModuleSpec("symmetric_add_normalized", "normalize candidate scores by their maximum before writing", parameters=(ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0),), tags=("scale_invariant",), similar_to=("symmetric_add_all",)),
                    ModuleSpec("symmetric_add_quantile", "write candidates above a score quantile", parameters=(ParameterSpec("fraction", "float", 0.5, minimum=0.1, maximum=1.0), ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0)), tags=("adaptive_sparsity",), similar_to=("symmetric_add_topk", "symmetric_add_all")),
                    ModuleSpec("symmetric_add_rank_weighted", "write all candidates with a rank-decaying weight", parameters=(ParameterSpec("exponent", "float", 0.5, minimum=0.0, maximum=2.0), ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0)), tags=("smooth_sparsity",), similar_to=("symmetric_add_all", "symmetric_add_topk")),
                    ModuleSpec("symmetric_add_usage_balanced", "write scores after an additional usage balance factor", parameters=(ParameterSpec("lambda", "float", 1.0, minimum=0.1, maximum=2.0),), tags=("history_aware",), similar_to=("symmetric_add_all",)),
                ),
                max_cut=4,
            ),
        )
    )
