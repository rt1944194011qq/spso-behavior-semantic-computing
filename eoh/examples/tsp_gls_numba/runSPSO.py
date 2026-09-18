"""Run the typed semantic PSO on the TSP GLS Numba task.

Edit the experiment constants below, then run ``python runSPSO.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "eoh" / "src"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SRC))

# Match the original EoH example: configure the remote LLM in this entrypoint.
# Fill in the laboratory key locally before an online run. Do not commit a real
# credential to a shared repository.
DEEPSEEK_ENDPOINT = "api.deepseek.com"
DEEPSEEK_API_KEY = ""
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_TIMEOUT = 150

# ── experiment configuration ─────────────────────────────────────────────────
# Change these values here, matching the style of the original EoH example.
OFFLINE = False                 # True = deterministic run without an LLM
SMOKE_TEST = False              # True = quick 3-instance/1-generation pipeline check
# Formal Fig. 3(a)-style run.  Use a fresh directory for each seed so that
# checkpoints, samples and curves from separate repetitions never mix.
OUTPUT_DIR = HERE / "results_spso" / "fig3a_full_seed2024"
SEED = 2024
POPULATION_SIZE = 10
GENERATIONS = 20
INITIAL_SAMPLES = 20            # None = 2 * POPULATION_SIZE
MAX_EVALUATIONS = None          # Full run: generations * population_size
RESUME_FROM = None              # e.g. OUTPUT_DIR / "checkpoints/generation_0001.json"

TRAIN_INSTANCES = 64            # Fig. 3(a): all 64 TSP100 instances
GLS_TIME_LIMIT = 60.0
GLS_ITE_MAX = 1000              # Fig. 3(a): local search max iterations
PERTURBATION_MOVES = 1
TASK_TIMEOUT = 4200             # 64 * 60 seconds plus outer-process margin
# The allocated server node has 30 CPU cores.  Evaluations are the CPU-bound
# stage; sampler threads mostly wait for LLM/network and evaluation futures.
NUM_SAMPLERS = 30
NUM_EVALUATORS = 30
CACHE_EVALUATIONS = False     # False = fair comparison: evaluate every generated sample
PRINT_BEST_ROUTES = True      # Print all 64 closed tours from the winning evaluation
ALPHA = None                  # None = sample one paper alpha per particle/generation; 0.5 = fixed-alpha ablation
MODULES_PER_SLOT = 5
HEURISTIC_OPERATORS = ["p1", "p2", "p3", "p4"]
USE_MODULE_DESCRIPTIONS = True
REUSE_EXISTING_MODULES = True
REUSE_EXISTING_PROBABILITY = 0.5

# Initial module-library policy:
#   False -> call the LLM and generate a new library;
#   True  -> load an existing module_library.json and skip library generation.
REUSE_MODULE_LIBRARY = True
# None means OUTPUT_DIR/module_library/module_library.json.  Set an explicit
# path when reusing a library from another experiment.
MODULE_LIBRARY_PATH = (
    HERE / "results_spso" / "fig3a_spso"
    / "module_library" / "module_library.json"
)
# Backward-compatible alias for older local scripts. Prefer the two settings
# above in new experiments.
FROZEN_MODULE_LIBRARY = None

# Keep the paper configuration above unchanged, but provide a switch for
# validating the complete LLM -> compile -> evaluate -> PSO pipeline before a
# full 64-instance run.  The smoke output uses a separate directory so it
# cannot contaminate the paper result files.
if SMOKE_TEST:
    OUTPUT_DIR = HERE / "results_spso" / "smoke_test"
    GENERATIONS = 1
    TRAIN_INSTANCES = 3
    GLS_TIME_LIMIT = 10.0
    TASK_TIMEOUT = 120
    NUM_SAMPLERS = 4
    NUM_EVALUATORS = 4

from eoh.eoh.evolution import _eval_with_timeout  # noqa: E402
from spso import SPSOConfig, SPSOEngine  # noqa: E402
from spso.selector import SemanticSelector  # noqa: E402

from spso_task import (  # noqa: E402
    DeterministicFakeLLM,
    TSPGLSCompiler,
    TSPModuleEvolution,
    TSPModuleLibraryBuilder,
    build_tsp_gls_registry,
)
from spso_task.reporting import TSPGLSWithDetails, print_route_report  # noqa: E402


def create_selector(registry, compiler, offline: bool):
    if offline:
        return SemanticSelector(
            registry,
            llm=None,
            module_code_provider=compiler.module_code,
            position_code_provider=lambda position: compiler.compile(position)[0],
        )
    from eoh.llm.interface_LLM import InterfaceLLM

    endpoint = DEEPSEEK_ENDPOINT
    api_key = DEEPSEEK_API_KEY
    model = DEEPSEEK_MODEL
    if not api_key or api_key.startswith("在这里填写"):
        raise RuntimeError("fill DEEPSEEK_API_KEY in runSPSO.py or use --offline")
    llm = InterfaceLLM(
        endpoint,
        api_key,
        model,
        use_local=False,
        local_url=None,
        timeout=DEEPSEEK_TIMEOUT,
    )
    return SemanticSelector(
        registry,
        llm=llm,
        module_code_provider=compiler.module_code,
        position_code_provider=lambda position: compiler.compile(position)[0],
    )


def create_llm(prototype, offline: bool):
    if offline:
        return DeterministicFakeLLM(prototype)
    from eoh.llm.interface_LLM import InterfaceLLM
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("在这里填写"):
        raise RuntimeError("fill DEEPSEEK_API_KEY in runSPSO.py or use --offline")
    return InterfaceLLM(
        DEEPSEEK_ENDPOINT,
        DEEPSEEK_API_KEY,
        DEEPSEEK_MODEL,
        use_local=False,
        local_url=None,
        timeout=DEEPSEEK_TIMEOUT,
    )


def load_module_registry(prototype, llm):
    """Load a checkpoint/library or build a new module library."""
    resume_library = None
    if RESUME_FROM:
        import json
        resume_payload = json.loads(Path(RESUME_FROM).read_text(encoding="utf-8"))
        resume_library = resume_payload.get("module_library")
        if not resume_library:
            raise RuntimeError("resume checkpoint has no module_library; cannot safely restore dynamic modules")
    if resume_library:
        registry = build_tsp_gls_registry()
        registry.restore_library(resume_library)
    elif REUSE_MODULE_LIBRARY or FROZEN_MODULE_LIBRARY:
        library_path = Path(MODULE_LIBRARY_PATH or FROZEN_MODULE_LIBRARY) if (
            MODULE_LIBRARY_PATH or FROZEN_MODULE_LIBRARY
        ) else (
            Path(OUTPUT_DIR) / "module_library" / "module_library.json"
        )
        if not library_path.is_file():
            raise RuntimeError(
                "REUSE_MODULE_LIBRARY=True, but module library was not found: "
                f"{library_path}. Set MODULE_LIBRARY_PATH or use "
                "REUSE_MODULE_LIBRARY=False to generate it."
            )
        import json
        registry = build_tsp_gls_registry()
        registry.restore_library(json.loads(library_path.read_text(encoding="utf-8")))
    else:
        registry = TSPModuleLibraryBuilder(
            prototype,
            llm,
            modules_per_slot=MODULES_PER_SLOT,
            output_dir=Path(OUTPUT_DIR) / "module_library",
            use_descriptions=USE_MODULE_DESCRIPTIONS,
        ).build()
    # Offline tests use a stateful fake whose candidate choices must follow
    # the frozen/restored library, not the six-module prototype.  The real
    # remote interface is unaffected by this assignment.
    if hasattr(llm, "prototype"):
        llm.prototype = registry
    return registry


def main():
    task = TSPGLSWithDetails(
        n_inst_eva=TRAIN_INSTANCES,
        time_limit=GLS_TIME_LIMIT,
        ite_max=GLS_ITE_MAX,
        perturbation_moves=PERTURBATION_MOVES,
        timeout=TASK_TIMEOUT,
    )
    prototype = build_tsp_gls_registry()
    llm = create_llm(prototype, OFFLINE)
    registry = load_module_registry(prototype, llm)
    compiler = TSPGLSCompiler(registry)
    evolution = TSPModuleEvolution(
        registry,
        compiler,
        llm,
        use_descriptions=USE_MODULE_DESCRIPTIONS,
        output_dir=Path(OUTPUT_DIR),
        reuse_existing_modules=REUSE_EXISTING_MODULES,
        reuse_existing_probability=REUSE_EXISTING_PROBABILITY,
    )
    selector = SemanticSelector(
        registry,
        llm=llm,
        module_code_provider=compiler.module_code,
        position_code_provider=lambda position: compiler.compile(position)[0],
        use_module_descriptions=USE_MODULE_DESCRIPTIONS,
        evolution=evolution,
    )

    def evaluate(code: str):
        return _eval_with_timeout(task, code, task.timeout)

    config = SPSOConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        initial_samples=INITIAL_SAMPLES,
        max_evaluations=MAX_EVALUATIONS,
        alpha=ALPHA,
        seed=SEED,
        output_dir=str(OUTPUT_DIR),
        use_llm=not OFFLINE,
        resume_from=str(RESUME_FROM) if RESUME_FROM else None,
        num_samplers=NUM_SAMPLERS,
        num_evaluators=NUM_EVALUATORS,
        cache_evaluations=CACHE_EVALUATIONS,
        modules_per_slot=MODULES_PER_SLOT,
        heuristic_operators=tuple(HEURISTIC_OPERATORS),
        use_module_descriptions=USE_MODULE_DESCRIPTIONS,
        llm_model=DEEPSEEK_MODEL,
    )
    engine = SPSOEngine(registry, compiler, evaluate, config, selector)
    engine.metadata["child_reuse_policy"] = {
        "enabled": REUSE_EXISTING_MODULES,
        "probability_for_p1_p2_p3": REUSE_EXISTING_PROBABILITY,
        "p4_reuses_module": True,
    }
    summary = engine.run()
    import json
    reuse_stats_path = Path(OUTPUT_DIR) / "child_reuse_stats.json"
    reuse_stats_path.write_text(
        json.dumps(evolution.reuse_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("S-PSO finished")
    print(f"best objective: {summary['best_objective']}")
    print(f"best position: {summary['best_position_text']}")
    print(f"samples: {summary['samples']}  independent evaluations: {summary['independent_evaluations']}")
    print(f"child reuse stats: {evolution.reuse_stats}")
    if PRINT_BEST_ROUTES:
        import json
        best_path = Path(OUTPUT_DIR) / "samples" / "best.json"
        best = json.loads(best_path.read_text(encoding="utf-8"))
        if best.get("details"):
            print_route_report(best)
            with (Path(OUTPUT_DIR) / "run_log.txt").open("a", encoding="utf-8") as handle:
                print_route_report(best, emit=lambda line: handle.write(line + "\n"))
        else:
            print("Legacy best has no routes; use inspectBestRoutes.py to reevaluate it.")


if __name__ == "__main__":
    main()
