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
DEEPSEEK_API_KEY = "在这里填写 DeepSeek API Key"
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_TIMEOUT = 150

# ── experiment configuration ─────────────────────────────────────────────────
# Change these values here, matching the style of the original EoH example.
OFFLINE = False                 # True = deterministic run without an LLM
OUTPUT_DIR = HERE / "results_spso" / "run_001"
SEED = 2024
POPULATION_SIZE = 10
GENERATIONS = 3
INITIAL_SAMPLES = 20            # None = 2 * POPULATION_SIZE
MAX_EVALUATIONS = 50            # None = no independent-evaluation cap
RESUME_FROM = None              # e.g. OUTPUT_DIR / "checkpoints/generation_0001.json"

TRAIN_INSTANCES = 3
GLS_TIME_LIMIT = 60.0
GLS_ITE_MAX = 1000
PERTURBATION_MOVES = 1
TASK_TIMEOUT = 3600
NUM_SAMPLERS = 16
NUM_EVALUATORS = 16
CACHE_EVALUATIONS = False     # False = fair comparison with EoH: evaluate all 50 samples
PRINT_BEST_ROUTES = True      # Print the three closed tours from the winning evaluation

from eoh.eoh.evolution import _eval_with_timeout  # noqa: E402
from spso import SPSOConfig, SPSOEngine  # noqa: E402
from spso.selector import SemanticSelector  # noqa: E402

from spso_task import TSPGLSCompiler, build_tsp_gls_registry  # noqa: E402
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


def main():
    task = TSPGLSWithDetails(
        n_inst_eva=TRAIN_INSTANCES,
        time_limit=GLS_TIME_LIMIT,
        ite_max=GLS_ITE_MAX,
        perturbation_moves=PERTURBATION_MOVES,
        timeout=TASK_TIMEOUT,
    )
    registry = build_tsp_gls_registry()
    compiler = TSPGLSCompiler(registry)
    selector = create_selector(registry, compiler, OFFLINE)

    def evaluate(code: str):
        return _eval_with_timeout(task, code, task.timeout)

    config = SPSOConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        initial_samples=INITIAL_SAMPLES,
        max_evaluations=MAX_EVALUATIONS,
        seed=SEED,
        output_dir=str(OUTPUT_DIR),
        use_llm=not OFFLINE,
        resume_from=str(RESUME_FROM) if RESUME_FROM else None,
        num_samplers=NUM_SAMPLERS,
        num_evaluators=NUM_EVALUATORS,
        cache_evaluations=CACHE_EVALUATIONS,
    )
    engine = SPSOEngine(registry, compiler, evaluate, config, selector)
    summary = engine.run()
    print("S-PSO finished")
    print(f"best objective: {summary['best_objective']}")
    print(f"best position: {summary['best_position_text']}")
    print(f"samples: {summary['samples']}  independent evaluations: {summary['independent_evaluations']}")
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
