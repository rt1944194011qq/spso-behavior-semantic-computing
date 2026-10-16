# MAE v4 local runtime revision (2026-10-16)

## Final experiment configuration

Edit `eoh/examples/tsp_gls_numba/runMAEoH.py` and click Run with the existing
`envs/spso_semantic/python.exe` interpreter. The launcher uses the following defaults:

| Setting | Value |
|---|---|
| Dataset | `TrainingData/custom_TSP200_seed2026_solved.pkl` |
| Evaluation subset | Fixed first 64 instances, each with 200 cities |
| Dataset SHA-256 | `7808cd48b46fc134bcc15278ec12378ecb83603d0c8e55ef0c4174a8a90a86d0` |
| Agents / generations | 10 / 20, initialization separate |
| Initial candidates per agent | 10 |
| GLS per instance | At most 1000 iterations or 60 seconds |
| Whole-candidate hard deadline | 1800 seconds, including startup and compilation |
| Samplers / evaluators | 16 / 16 |
| Async candidate pipeline | Enabled |
| Numba | Preferred, `nopython`, no fast-math or automatic parallel loops |
| Python compatibility fallback | Allowed; execution mode and compilation reason recorded |
| Per-callback time cutoff | Disabled (`MAX_HEURISTIC_CALL_SECONDS=0`) |

The `.sh` launcher has the same formal defaults. Existing exported environment
variables can override shell defaults. No server files or online experiment were
started by this local revision. Repackage and upload the changed `eoh` directory
before starting a new server run.

## Execution order

Each bounded sampler generates a candidate, validates it, evaluates it in an
isolated subprocess and records its result immediately. The agent's original
current code remains in its base pool with its existing score. Once that agent's
base candidates have completed, it advances through H3, H4, H5 and H6 in order;
each result is compared against its latest worst pool member before continuing.
Another agent need not finish the same operator first.

Current/pbest/gbest/neighbor references used to refresh semantic experience are
still frozen at the start of a generation. Final agent selection and checkpoints
are written at the generation boundary. This preserves the MAE generation model
while applying EoH's generate/evaluate/refill scheduling pattern to candidates.
H1/H2 scheduling, cut probabilities, experience decay and operator arities retain
the v3 definitions. Random parents are selected from a canonically ranked pool.

Concurrent identical programs share an in-flight evaluation. Archive writes,
IDs, counters and deduplication are synchronized. A large result is read from
the multiprocessing queue before joining the worker, avoiding join/queue-buffer
deadlock. MAE records `evaluation_timeout` separately from numerical or compiler
failures, and kills the evaluation process group on its hard deadline.

## What was measured

The retrieved conversation is **论文实验对标**. Its archived EoH run in
`D:/eoh_200_train/run_log.txt` records 200-instance training, 16 samplers,
16 evaluators and a 5400-second candidate cap. It has no reliable individual
64-instance evaluation timing; intervals between asynchronous sample completions
cannot supply that timing.

Fresh local measurements on the requested fixed **64 x TSP200** subset, with
1000 iterations and 60 seconds per instance, including process startup:

| Program and evaluator | Wall time | Mean gap (%) |
|---|---:|---:|
| User-trained EoH best, EoH Python evaluator | 122.562 s | 0.28266991176270007 |
| Author run3, EoH Python evaluator | 181.409 s | 0.754281064451421 |
| Same author run3, MAE Numba evaluator | 175.791 s | 0.754281064451421 |

These are single local measurements, not server speed guarantees. The Numba
reference diagnostic included optional 5 ms callback timing instrumentation;
the final training defaults disable that instrumentation. Its 64 instances all
reached the iteration cap. Native compilation did not change its aggregate gap.
The end-to-end gain for this already-vectorized reference is about 3%, not the
much larger speedup possible for an isolated Python numeric loop.

Final optional-Numba policy was separately exercised on archived slow programs
`ind-00000141` and `ind-00000148`: both correctly used `python-fallback` and
completed a one-instance, 50-iteration diagnostic in 2.339 s and 2.663 s.
This does not imply a full 64-instance pass of those slow programs is fast;
their full evaluation is bounded by the configurable 1800-second deadline.

See [raw benchmark and table](../reports/mae_async_numba_20261016_141012/README.md).
The two strict-Numba diagnostic failures in that table were measured during
implementation; strict rejection is **not** the final default. The earlier
`140011` report used TSP100 and its full-200 pressure case was manually cancelled
after the requested 64 x TSP200 protocol was clarified.

## Verification

Regression coverage includes generation/evaluation overlap, a fast agent reaching
H6 while a different agent remains blocked in H3, per-agent replacement ordering,
concurrent evaluation sharing, JIT helper calls and history writeback, optional
Python fallback, real-size contract checks, whole-candidate termination and a
1 MiB result crossing the subprocess queue. A final offline integration run
completed initialization plus one generation: 13/13 valid evolution proposals,
about 13.4 seconds, with the callback cutoff disabled. No online LLM success rate
or complete 20-generation experiment was measured in this revision.

Async per-agent stage durations overlap. Use generation wall time or the
`asynchronous_evolution` stage for elapsed-time plots; do not sum all agents'
H3/H4/H5/H6 durations. `summarize_mae_run.py` labels these scopes explicitly.
