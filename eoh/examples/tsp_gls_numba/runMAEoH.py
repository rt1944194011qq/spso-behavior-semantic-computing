"""修改下面的参数后，直接在编辑器中运行本文件。

当前配置为完整在线实验：10 个智能体、20 代、200 个训练实例。
服务器后台脚本可显式启用环境变量配置。
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "eoh" / "src"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SRC))

from eoh.llm.interface_LLM import InterfaceLLM
from eoh.mae import LLMProgramGenerator, MAEConfig, MAEEngine
from eoh.mae.validation import TSP_INPUT_CONTRACT, TSP_INPUT_MUTATION_POLICY_ID
from mae_tsp_adapter import OfflineTSPGenerator, TracedTSPGLS, build_tsp_validator


# =========================== 在这里修改运行参数 ===========================
# True：离线预设代码生成器，用于检查流程；False：调用真实 LLM 生成代码。
OFFLINE = False

# 默认直接使用本文件参数。服务器启动脚本会显式启用环境变量覆盖。
USE_ENVIRONMENT_OVERRIDES = False

# ------------------------------- LLM ----------------------------------------
# The key is intentionally blank in version control. Fill it only in your local
# or server copy, or provide DEEPSEEK_API_KEY as a process environment variable.
DEEPSEEK_ENDPOINT = "api.deepseek.com"
# Keep credentials out of source control. Set DEEPSEEK_API_KEY in the shell
# environment on the local machine or the laboratory server.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_TIMEOUT = 180

# ------------------------------- experiment ---------------------------------
SEED = 2026
RUN_NAME = ""                  # 留空：每次点击运行自动生成时间戳名称
RESUME_FROM = None              # 续跑时填写检查点的完整路径
TRAIN_DATASET = HERE / "TrainingData" / "custom_TSP200_seed2026_solved.pkl"

NUM_AGENTS = 10                # 智能体数
POSITION_SIZE = 1
GENERATIONS = 20               # 演化代数，不含初始化
INITIAL_MULTIPLIER = 10        # 每个智能体初始化候选数

TRAIN_INSTANCES = 200          # 使用全部 200 个训练实例
GLS_TIME_LIMIT = 60.0          # 单实例时间上限（秒）
GLS_ITE_MAX = 1000              # 单实例迭代上限
PERTURBATION_MOVES = 1
TASK_TIMEOUT = 900             # 单候选全部200个实例的总评测上限（15分钟），超时判失败

# Threads overlap LLM network I/O. Evaluation workers launch isolated EoH
# subprocesses, so keep NUM_EVALUATORS within the allocated CPU count.
NUM_LLM_WORKERS = 16
NUM_EVALUATORS = 16
ISOLATE_EVALUATION = True

# ------------------------------- MA-EoH --------------------------------------
EXPERIENCE_BIRTH_MIN = 0.20
EXPERIENCE_BIRTH_MAX = 0.60
EXPERIENCE_TARGET_WEIGHT = 0.60
EXPERIENCE_HALF_LIFE = 4.0
EXPERIENCE_DROP_BELOW = 0.05
EXPERIENCE_CAPACITY = 24
CUT_TARGET_SIZE = 4
CUT_MIN_SIZE = 2
NEIGHBOR_RESERVED_SLOTS = 1
H1_SELECTION_BOOST = 1.5
H2_SELECTION_BOOST = 1.5
ENABLED_OPERATORS = ("H1", "H2", "H3", "H4", "H5")
DETAILED_LOGGING = True
PROGRESS_LOG_EVERY = 5
HEARTBEAT_INTERVAL_SECONDS = 30.0
RECORD_FULL_PROMPTS = True
RECORD_FULL_RESPONSES = True

# ======================== 以下是运行逻辑，通常不用改 ========================
# 保留服务器后台脚本兼容性；普通点击运行不会读取旧 MAE_* 参数。
if USE_ENVIRONMENT_OVERRIDES or os.environ.get("MAE_USE_ENVIRONMENT") == "1":
    _smoke = os.environ.get("MAE_SMOKE_TEST", "0") == "1"
    OFFLINE = os.environ.get("MAE_OFFLINE", "0") == "1"
    _defaults = {
        "NUM_AGENTS": 2 if _smoke else 10,
        "GENERATIONS": 1 if _smoke else 20,
        "INITIAL_MULTIPLIER": 2 if _smoke else 10,
        "TRAIN_INSTANCES": 1 if _smoke else 200,
        "GLS_TIME_LIMIT": 0.25 if _smoke else 60.0,
        "GLS_ITE_MAX": 20 if _smoke else 1000,
        "TASK_TIMEOUT": 120 if _smoke else 900,
        "NUM_LLM_WORKERS": 4 if _smoke else 16,
        "NUM_EVALUATORS": 4 if _smoke else 16,
    }
    for _name in (
        "SEED", "NUM_AGENTS", "GENERATIONS", "INITIAL_MULTIPLIER",
        "TRAIN_INSTANCES", "GLS_TIME_LIMIT", "GLS_ITE_MAX", "PERTURBATION_MOVES",
        "TASK_TIMEOUT", "NUM_LLM_WORKERS", "NUM_EVALUATORS",
        "EXPERIENCE_BIRTH_MIN", "EXPERIENCE_BIRTH_MAX", "EXPERIENCE_TARGET_WEIGHT",
        "EXPERIENCE_HALF_LIFE", "EXPERIENCE_DROP_BELOW", "EXPERIENCE_CAPACITY",
        "CUT_TARGET_SIZE", "CUT_MIN_SIZE", "NEIGHBOR_RESERVED_SLOTS",
        "H1_SELECTION_BOOST", "H2_SELECTION_BOOST",
    ):
        _default = _defaults.get(_name, globals()[_name])
        globals()[_name] = type(_default)(os.environ.get(f"MAE_{_name}", _default))
    RUN_NAME = os.environ.get("MAE_RUN_NAME", "")
    RESUME_FROM = os.environ.get("MAE_RESUME_FROM") or None
    TRAIN_DATASET = Path(os.environ.get("MAE_TRAIN_DATASET", str(TRAIN_DATASET)))
    ISOLATE_EVALUATION = os.environ.get("MAE_ISOLATE_EVAL", "1") != "0"
    ENABLED_OPERATORS = tuple(
        item.strip() for item in os.environ.get(
            "MAE_OPERATORS", ",".join(ENABLED_OPERATORS)
        ).split(",") if item.strip()
    )

TRAIN_DATASET = Path(TRAIN_DATASET).expanduser().resolve()
OUTPUT_DIR = HERE / "results_mae" / RUN_NAME


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        input_contract=TSP_INPUT_CONTRACT,
    )


def main():
    global RUN_NAME, OUTPUT_DIR
    if not RUN_NAME:
        if RESUME_FROM:
            RUN_NAME = Path(RESUME_FROM).resolve().parent.parent.name
        else:
            mode = "offline" if OFFLINE else "online"
            RUN_NAME = f"mae_tsp200_{mode}_seed{SEED}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    OUTPUT_DIR = HERE / "results_mae" / RUN_NAME
    if not TRAIN_DATASET.is_file():
        raise FileNotFoundError(f"training dataset does not exist: {TRAIN_DATASET}")
    dataset_sha256 = file_sha256(TRAIN_DATASET)
    task = TracedTSPGLS(
        n_inst_eva=TRAIN_INSTANCES,
        time_limit=GLS_TIME_LIMIT,
        ite_max=GLS_ITE_MAX,
        perturbation_moves=PERTURBATION_MOVES,
        timeout=TASK_TIMEOUT,
        datafile=TRAIN_DATASET,
    )
    task.dataset_sha256 = dataset_sha256
    generator = build_generator(task)
    validator = build_tsp_validator(task)
    config = MAEConfig(
        num_agents=NUM_AGENTS,
        position_size=POSITION_SIZE,
        generations=GENERATIONS,
        initial_multiplier=INITIAL_MULTIPLIER,
        seed=SEED,
        experience_birth_min=EXPERIENCE_BIRTH_MIN,
        experience_birth_max=EXPERIENCE_BIRTH_MAX,
        experience_target_weight=EXPERIENCE_TARGET_WEIGHT,
        experience_half_life=EXPERIENCE_HALF_LIFE,
        experience_drop_below=EXPERIENCE_DROP_BELOW,
        experience_capacity=EXPERIENCE_CAPACITY,
        cut_target_size=CUT_TARGET_SIZE,
        cut_min_size=CUT_MIN_SIZE,
        neighbor_reserved_slots=NEIGHBOR_RESERVED_SLOTS,
        h1_selection_boost=H1_SELECTION_BOOST,
        h2_selection_boost=H2_SELECTION_BOOST,
        enabled_operators=ENABLED_OPERATORS,
        num_llm_workers=NUM_LLM_WORKERS,
        num_evaluators=NUM_EVALUATORS,
        isolate_evaluation=ISOLATE_EVALUATION,
        output_dir=str(OUTPUT_DIR),
        resume_from=str(RESUME_FROM) if RESUME_FROM else None,
        evaluation_fingerprint=(
            f"tspgls:dataset={TRAIN_DATASET.name}:sha256={dataset_sha256}:"
            f"n={TRAIN_INSTANCES}:time={GLS_TIME_LIMIT}:"
            f"iters={GLS_ITE_MAX}:moves={PERTURBATION_MOVES}:seed={SEED}:"
            f"validation={TSP_INPUT_MUTATION_POLICY_ID}"
        ),
        detailed_logging=DETAILED_LOGGING,
        progress_log_every=PROGRESS_LOG_EVERY,
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
        record_full_prompts=RECORD_FULL_PROMPTS,
        record_full_responses=RECORD_FULL_RESPONSES,
    )
    print(f"Python: {sys.executable}")
    print(f"generation mode: {'offline preset generator' if OFFLINE else 'online LLM'}")
    print(f"training dataset: {TRAIN_DATASET}")
    print(f"dataset sha256: {dataset_sha256}")
    print(f"training instances: {TRAIN_INSTANCES}")
    summary = MAEEngine(config, task, generator, validator).run()
    print("MA-EoH finished")
    print(f"best fitness: {summary['best_fitness']}")
    print(f"best heuristic: {summary['best_code_path']}")
    print(f"results: {summary['output_dir']}")


if __name__ == "__main__":
    main()
