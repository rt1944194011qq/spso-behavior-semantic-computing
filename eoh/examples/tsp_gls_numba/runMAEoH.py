"""Run memory-guided multi-agent heuristic evolution on TSP-GLS-numba.

Edit the constants in this file, then run:
    python eoh/examples/tsp_gls_numba/runMAEoH.py

Keep SMOKE_TEST=True and OFFLINE=True for the first local verification. For a
real LLM experiment set both to False and fill the API settings below.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "eoh" / "src"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SRC))

from eoh.llm.interface_LLM import InterfaceLLM
from eoh.mae import LLMProgramGenerator, MAEConfig, MAEEngine
from mae_tsp_adapter import OfflineTSPGenerator, TracedTSPGLS, build_tsp_validator


# ------------------------------- run mode -----------------------------------
SMOKE_TEST = False
OFFLINE = False

# ------------------------------- LLM ----------------------------------------
# The key is intentionally blank in version control. Fill it only in your local
# or server copy, or provide DEEPSEEK_API_KEY as a process environment variable.
DEEPSEEK_ENDPOINT = "api.deepseek.com"
# Keep credentials out of source control. Set DEEPSEEK_API_KEY in the shell
# environment on the local machine or the laboratory server.
DEEPSEEK_API_KEY = ""
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_TIMEOUT = 180

# ------------------------------- experiment ---------------------------------
# Use a fresh name for every independent run. This local smoke run is
# deliberately separated from all previous MAE-EoH outputs.
RUN_NAME = "mae_smoke_seed2024_run005" if SMOKE_TEST else "mae_full_seed2024_run001"
OUTPUT_DIR = HERE / "results_mae" / RUN_NAME
SEED = 2024
RESUME_FROM = None

NUM_AGENTS = 2 if SMOKE_TEST else 10
POSITION_SIZE = 5
GENERATIONS = 1 if SMOKE_TEST else 20
INITIAL_MULTIPLIER = 2

TRAIN_INSTANCES = 1 if SMOKE_TEST else 64
GLS_TIME_LIMIT = 0.25 if SMOKE_TEST else 60.0
GLS_ITE_MAX = 20 if SMOKE_TEST else 1000
PERTURBATION_MOVES = 1
TASK_TIMEOUT = 120 if SMOKE_TEST else 4200

# Threads overlap LLM network I/O. Evaluation workers launch isolated EoH
# subprocesses, so keep NUM_EVALUATORS within the allocated CPU count.
NUM_LLM_WORKERS = 4 if SMOKE_TEST else 16
NUM_EVALUATORS = 4 if SMOKE_TEST else 16

# ------------------------------- MA-EoH --------------------------------------
ALPHA = 0.50
MAX_CUT_SIZE = 10
OMEGA = 0.50
NEIGHBOR_LEFT = 0.40
NEIGHBOR_RIGHT = 0.40
H1_PERSONAL = 0.90
H1_GLOBAL = 0.25
H2_PERSONAL = 0.25
H2_GLOBAL = 0.90
VARIATION_OPERATORS = ("H3", "H4", "H5")
DETAILED_LOGGING = True
PROGRESS_LOG_EVERY = 5
HEARTBEAT_INTERVAL_SECONDS = 30.0
RECORD_FULL_PROMPTS = True
RECORD_FULL_RESPONSES = True


def build_generator(task):
    if OFFLINE:
        return OfflineTSPGenerator(seed=SEED)

    api_key = DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Online mode requires DEEPSEEK_API_KEY in this file or in the process environment."
        )
    llm = InterfaceLLM(
        api_endpoint=DEEPSEEK_ENDPOINT,
        api_key=api_key,
        model=DEEPSEEK_MODEL,
        use_local=False,
        local_url=None,
        timeout=DEEPSEEK_TIMEOUT,
    )
    return LLMProgramGenerator(
        llm,
        task.task_description,
        task.template_program,
        retries=3,
        require_description=True,
        trace_path=OUTPUT_DIR / "prompts" / "llm_calls.jsonl",
        record_full_prompts=RECORD_FULL_PROMPTS,
        record_full_responses=RECORD_FULL_RESPONSES,
    )


def main():
    task = TracedTSPGLS(
        n_inst_eva=TRAIN_INSTANCES,
        time_limit=GLS_TIME_LIMIT,
        ite_max=GLS_ITE_MAX,
        perturbation_moves=PERTURBATION_MOVES,
        timeout=TASK_TIMEOUT,
    )
    generator = build_generator(task)
    validator = build_tsp_validator(task)
    config = MAEConfig(
        num_agents=NUM_AGENTS,
        position_size=POSITION_SIZE,
        generations=GENERATIONS,
        initial_multiplier=INITIAL_MULTIPLIER,
        seed=SEED,
        alpha=ALPHA,
        max_cut_size=MAX_CUT_SIZE,
        omega=OMEGA,
        neighbor_left=NEIGHBOR_LEFT,
        neighbor_right=NEIGHBOR_RIGHT,
        h1_personal=H1_PERSONAL,
        h1_global=H1_GLOBAL,
        h2_personal=H2_PERSONAL,
        h2_global=H2_GLOBAL,
        variation_operators=VARIATION_OPERATORS,
        num_llm_workers=NUM_LLM_WORKERS,
        num_evaluators=NUM_EVALUATORS,
        isolate_evaluation=True,
        output_dir=str(OUTPUT_DIR),
        resume_from=str(RESUME_FROM) if RESUME_FROM else None,
        evaluation_fingerprint=(
            f"tspgls:n={TRAIN_INSTANCES}:time={GLS_TIME_LIMIT}:"
            f"iters={GLS_ITE_MAX}:moves={PERTURBATION_MOVES}:seed={SEED}"
        ),
        detailed_logging=DETAILED_LOGGING,
        progress_log_every=PROGRESS_LOG_EVERY,
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
        record_full_prompts=RECORD_FULL_PROMPTS,
        record_full_responses=RECORD_FULL_RESPONSES,
    )
    summary = MAEEngine(config, task, generator, validator).run()
    print("MA-EoH finished")
    print(f"best fitness: {summary['best_fitness']}")
    print(f"best heuristic: {summary['best_code_path']}")
    print(f"results: {summary['output_dir']}")


if __name__ == "__main__":
    main()
