import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import TSPGLS


# Reproduction configuration shared with runSPSO.py.
TRAIN_INSTANCES = 64
GLS_TIME_LIMIT = 60.0
GLS_ITE_MAX = 1000
PERTURBATION_MOVES = 1
TASK_TIMEOUT = 4200  # 64 * 60 seconds plus outer-process margin
NUM_SAMPLERS = 16
NUM_EVALUATORS = 16
POPULATION_SIZE = 10
GENERATIONS = 20
N_PARENTS = 5  # E1/E2 only; mutation operators still use one parent.


if __name__ == "__main__":
    llm = LLMConfig(
        api_endpoint='api.deepseek.com',
        api_key='在这里填写 DeepSeek API Key',
        model='deepseek-flash',
        timeout=150,
    )

    task = TSPGLS(
        n_inst_eva=TRAIN_INSTANCES,
        time_limit=GLS_TIME_LIMIT,
        ite_max=GLS_ITE_MAX,
        perturbation_moves=PERTURBATION_MOVES,
        timeout=TASK_TIMEOUT,
    )

    eoh = EoH(
        llm=llm,
        problem=task,
        num_samplers=NUM_SAMPLERS,
        num_evaluators=NUM_EVALUATORS,
        pop_size=POPULATION_SIZE,
        n_pop=GENERATIONS,
        n_parents=N_PARENTS,
        operators=['e1', 'e2', 'm1', 'm2'],
        output_dir=os.path.dirname(__file__),
    )

    eoh.run()
