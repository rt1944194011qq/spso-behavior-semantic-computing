# TSP GLS S-PSO adapter

This adapter searches a typed five-slot representation:

`E | F | H | T | W`

- `E`: candidate edge generation.
- `F`: positive edge feature.
- `H`: historical penalty factor.
- `T`: score transformation.
- `W`: symmetric write-back to the distance matrix.

The expanded library currently contains 6 E modules, 6 F modules, 7 H
modules, 6 T modules and 6 W modules. This gives 9072 combinations before
counting bounded parameter values. The velocity cut keeps only a small subset
per slot during a run, so the engine does not enumerate this Cartesian product.

The compiler always emits the public EoH contract:

`update_edge_distance(edge_distance, local_opt_tour, edge_n_used) -> updated_edge_distance`

The returned matrix is a new finite, nonnegative, symmetric square matrix. The
classic GLS penalty is represented by:

`tour_edges | edge_length | inverse_count | identity | symmetric_add_all(lambda=1.0)`

The search state follows the set-based PSO representation from the project
paper. Each TSP slot has a crisp module set with capacity `K_s=3`, while its
velocity stores a possibility in `[0, 1]` for every registered module. The
update uses the slot-wise differences `PBest-X` and `GBest-X`, possibility-set
addition (maximum for duplicate modules), and an absolute alpha-cut
`{module | possibility >= alpha}`. `ALPHA=None` samples one alpha per particle
and generation; a numeric value such as `0.5` is a fixed-alpha ablation.

The alpha-cut is expanded into the next set by the order `cut -> current set
-> random remaining modules`, up to `K_s`. Only then does the semantic selector
choose one module and its parameters per slot. This decoded
`ParticlePosition` is compiled into the original three-argument GLS function,
so the evaluator and downstream `heuristic.py` workflow remain unchanged.

The runner uses an EoH-style two-level pipeline. `NUM_SAMPLERS` controls the
threads that perform cut construction, LLM selection and compilation;
`NUM_EVALUATORS` controls the shared pool of isolated GLS evaluations. A
generation freezes its particle/pbest/gbest snapshot before submitting sampler
tasks, and applies all particle updates after that generation's candidates
finish.

Set `OFFLINE = True` and reduce the constants in `runSPSO.py` for a local
deterministic smoke test. Then run from this directory with:

```text
python runSPSO.py
```

For an online run, fill `DEEPSEEK_API_KEY` in `runSPSO.py`, leave `OFFLINE =
False`, and edit the experiment constants in the same file. The original EoH
`runEoH.py` is kept as a separate baseline entry point.

The default output directory is `results_spso/run_001`.

To use the original EoH test workflow, edit `BEST_FILE` in
`install_best_to_heuristic.py`, run it, then enter the `evaluation` directory
and execute `python runEval.py`. The installer backs up the original
`heuristic.py` beside `best.json` before replacing it.
