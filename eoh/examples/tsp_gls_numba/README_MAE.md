# 语义经验库 MA-EoH v2：TSP-GLS-numba 运行说明

`runMAEoH.py` 默认调用新版 `eoh.mae.MAEEngine`。旧五程序位置实现保存在
`eoh.mae.LegacyMAEEngine`，旧结果目录不会被迁移或覆盖。每次运行必须使用新
运行名；v1 检查点不能直接恢复到 v2。直接点击运行时，运行名留空会自动生成时间戳。

## 本地直接点击运行

打开 `runMAEoH.py`，在编辑器中选择已有解释器
`D:\spso-behavior-semantic-computing\envs\spso_semantic\python.exe`，然后点击运行本文件。
运行参数全部集中在文件开头，直接修改即可，无需在终端输入环境变量。

当前文件配置为 `OFFLINE=False` 的完整在线实验：10个智能体、初始化各10个候选、
20代、全部200个200节点实例、GLS上限60秒/1000次迭代、16个评测进程。
在线运行需在文件中填写 `DEEPSEEK_API_KEY`，或使用已设置的密钥环境变量。
若要重新做离线小测试，改为 `OFFLINE=True`，并将智能体/初始化候选/代数/
训练实例分别设为2/2/1/1，GLS上限设为0.25秒/20次迭代，并发设为2。
`RUN_NAME=""` 会自动创建独立结果目录，路径在结束时打印。

普通点击运行默认使用文件参数；只有 `USE_ENVIRONMENT_OVERRIDES=True` 或显式设置
`MAE_USE_ENVIRONMENT=1` 才读取 `MAE_*` 环境变量。服务器后台脚本已自动启用这一开关。

当前正式训练数据默认改为
`TrainingData/custom_TSP200_seed2026_solved.pkl`：共 200 个非均匀分布实例，
每个实例 200 个节点，均包含已求得的 `optimal_tour` 和 `cost`。数据文件名和
SHA-256 会写入评测指纹及运行环境记录，避免跨数据集错误复用适应度缓存。

## 与 EoH 对齐的输入读写规则

当前 TSP 校验默认允许启发式原地更新 `edge_n_used`，与 EoH GLS 的历史状态
传递方式一致。该计数仅在单个实例内持续存在，每次求解新实例都会重新置零；
GLS 宿主在启发式返回后仍会执行自身的计数更新。更新历史计数是可选能力，
不要求所有候选使用，也不改变最终按原始浮点距离计算路线长度的规则。

`edge_distance` 与 `local_opt_tour` 仍禁止原地修改；输出矩阵仍校验形状、
有限值，以及原有的对称/非负要求。H0、DIFF、H3/H4/H5 的代码生成提示词
统一说明此读写策略。需要旧规则消融时，可显式调用
`build_tsp_validator(task, allow_edge_n_used_update=False)`，并配套使用禁止修改
历史计数的提示词和独立运行指纹。

新训练的评测指纹增加 `validation=eoh-edge-history-v1`。旧严格只读规则的
检查点不能直接用于新规则续跑，必须使用新的运行目录；旧训练记录、冻结程序
和已完成测试结果不作修改。本次对齐不改变种群选择、算子预算或评测超时。

## 算法步骤

1. H0 为每个智能体生成若干完整程序，只保留一个当前程序和单独的历史最优。
2. 每代开始冻结所有当前、本地最优、全局最优和环形左右邻居最优。当前代码与
   四个参照程序分别比较，形成带方向、来源和适应度的语义差异提示词。
3. 旧经验按半衰期衰减，新经验以适应度决定的出生概率合并进唯一速度库。
4. H1/H2 仅临时放大本地/全局经验的 cut 抽取概率。放大后及乘以数量调节系数
   `lambda` 后都截断到 `[0,1]`；邻居保留一个轮换名额。超额裁剪，不足补选。
5. cut 中每条差异结合当前程序生成一个完整基底。H3/H4/H5 均从同一批基底
   分别尝试生成 `N` 个程序；H5 随机选父代。正常预算是每智能体 `4N` 个提案。
6. 只从本代有效新程序中选一个当前程序，旧当前仅在全失效时回退；`pbest`
   与 `gbest` 始终单独保存。所有程序继续使用原 EoH TSP 评测函数与接口。

经验不足时额外运行 H0 探索；无效、重复或超时程序不会假装成有效子代。

## 服务器启动脚本的正式默认参数

| 参数 | 值 | 含义 |
|---|---:|---|
| `SEED` | 2026 | 算法随机种子 |
| `TRAIN_DATASET` | `custom_TSP200_seed2026_solved.pkl` | 已求解的训练数据文件 |
| `NUM_AGENTS` | 10 | 智能体数 |
| `POSITION_SIZE` | 1 | 每个智能体的当前完整程序数 |
| `GENERATIONS` | 20 | 演化代数，不含 H0 |
| `INITIAL_MULTIPLIER` | 10 | H0 首轮每智能体提案数 |
| `EXPERIENCE_BIRTH_MIN/MAX` | 0.20 / 0.60 | 新经验出生概率范围 |
| `EXPERIENCE_TARGET_WEIGHT` | 0.60 | 出生概率中对标代码质量的权重；当前代码权重为 0.40 |
| `EXPERIENCE_HALF_LIFE` | 4.0 代 | 旧经验约四代概率减半，保留系数约 0.8409 |
| `EXPERIENCE_DROP_BELOW` | 0.05 | 清理过低概率的旧经验 |
| `EXPERIENCE_CAPACITY` | 24 | 每智能体最多保留的经验数 |
| `CUT_TARGET_SIZE` | 4 | 有足够经验时每智能体 cut 目标数 |
| `CUT_MIN_SIZE` | 2 | 不足时由 H0 另行补探索，不虚构经验 |
| `NEIGHBOR_RESERVED_SLOTS` | 1 | 邻居经验名额，左右轮换 |
| `H1/H2_SELECTION_BOOST` | 1.5 / 1.5 | 仅本次抽取的放大系数 |
| `ENABLED_OPERATORS` | H1,H2,H3,H4,H5 | 本地修改文件中的列表，服务器也可用 `MAE_OPERATORS` 做消融 |
| `TRAIN_INSTANCES` | 200 | 每个候选评测使用的训练实例数 |
| `GLS_TIME_LIMIT` / `GLS_ITE_MAX` | 60 秒 / 1000 | 单实例 GLS 限制 |
| `TASK_TIMEOUT` | 900 秒 | 单候选全部200实例的总评测上限，超时判失败 |
| `NUM_LLM_WORKERS` / `NUM_EVALUATORS` | 16 / 16 | LLM 请求 / 独立评测并发 |

单候选的15分钟总上限不是固定运行时长。它不覆盖200个实例都用满60秒的最坏情况；
即使单实例未达到上限，总评测超过900秒也会终止并记为失败。正式实验需记录超时率。

新经验的出生概率同时考虑当前代码和对标代码。先把适应度按优化方向转换为
“越小越好”的排名值 `r_s`（当前代码）和 `r_t`（对标代码），令本代最好值为
`r_*`，尺度 `s` 为本代相对 `r_*` 的正差中位数：
`q_s=exp(-max(0,r_s-r_*)/s)`，`q_t=exp(-max(0,r_t-r_*)/s)`，
`p=clip(0.20+0.40×(0.40q_s+0.60q_t),0,1)`。该式使用相对差和本代尺度，
不受适应度整体很小的量纲影响；对标端权重略高，因为差异提示词的方向是从
当前代码指向对标代码。旧概率每代乘 `2^(-1/4)≈0.8409`，
再次遇到同一条经验时取“新出生概率”和“衰减后概率”的较大值。
H1/H2 对其对应来源只在抽取时计算 `w=min(1,1.5p)`；其他来源 `w=p`。
每代、每智能体独立求 `lambda`，使剩余候选的
`Σmin(1,lambda×w)=目标 cut 数−邻居预留数`。抽样后超额裁剪、不足补选，
所以最终 cut 不会超过 4 条；经验库本身的 `p` 不受 H1/H2 或 `lambda` 修改。

切实生成数由 cut 大小与合法性决定。若 `N=4` 且 H3/H4/H5 全开，正常每智能体
16 个、10 个智能体每代 160 个提案，比旧版 60 个/代成本更高；比较方法时必须
以相同的独立评测次数为横轴，而不是仅比较代数。

正式默认预算上限为 H0 的 100 个候选，加上 20 代约 3200 个候选，共约 3300 次
候选评测；每个候选使用 200 个实例，对应约 660000 次 GLS 实例运行。正式启动前
必须先做小规模试跑测量单候选耗时，并确保 `NUM_EVALUATORS` 不超过作业分配的
CPU 核数。

## 推荐运行方案

### 1. 校验训练数据

```bash
cd /public2/home/rentao/eoh/examples/tsp_gls_numba
PY=/public2/home/rentao/miniconda3/envs/mae_eoh/bin/python
"$PY" validate_tsp_training_data.py \
  --dataset TrainingData/custom_TSP200_seed2026_solved.pkl
```

输出必须显示 `valid: true`、`instance_count: 200` 和
`node_count_distribution: {200: 200}`。

### 2. 离线冒烟测试

不请求 LLM，只验证加载数据、生成、评测、记录和检查点流程：

```bash
MAE_USE_ENVIRONMENT=1 MAE_SMOKE_TEST=1 MAE_OFFLINE=1 \
MAE_RUN_NAME=mae_tsp200_seed2026_smoke \
  "$PY" -u runMAEoH.py
```

### 3. 在线小规模试跑

先使用 20 个训练实例、4 个智能体和 3 代估算耗时与成功率：

```bash
export DEEPSEEK_API_KEY='你的密钥'
MAE_RUN_NAME=mae_tsp200_seed2026_pilot001 \
MAE_NUM_AGENTS=4 MAE_GENERATIONS=3 MAE_INITIAL_MULTIPLIER=4 \
MAE_TRAIN_INSTANCES=20 MAE_GLS_TIME_LIMIT=5 MAE_GLS_ITE_MAX=200 \
MAE_TASK_TIMEOUT=900 MAE_NUM_LLM_WORKERS=4 MAE_NUM_EVALUATORS=4 \
  bash run_mae_tsp200.sh
```

### 4. 正式后台运行

正式脚本会先校验数据集，再使用 `nohup` 启动训练和每60秒一次的独立监控器：

```bash
export DEEPSEEK_API_KEY='你的密钥'
export MAE_RUN_NAME=mae_semantic_tsp200_seed2026_run001
bash run_mae_tsp200.sh
```

脚本采用本节参数表中的正式默认值。若计算节点只分配8核，请在启动前设置：

```bash
export MAE_NUM_EVALUATORS=8
```

## 在服务器上先跑离线冒烟

先将本地新版代码同步到服务器；只在本地修改不会自动更新服务器副本。
以下命令使用已有的 `mae_eoh` 环境，不联系 LLM。先进入任务分配的计算节点：

```bash
cd /public2/home/rentao/eoh/examples/tsp_gls_numba
PY=/public2/home/rentao/miniconda3/envs/mae_eoh/bin/python
MAE_USE_ENVIRONMENT=1 MAE_SMOKE_TEST=1 MAE_OFFLINE=1 MAE_RUN_NAME=mae_semantic_smoke_001 \
  "$PY" -u runMAEoH.py
```

应看到 `best fitness`，并生成
`results_mae/mae_semantic_smoke_001/best/best_heuristic.py`。冒烟配置是
2 智能体、1 代、每智能体 2 个 H0 提案、1 个训练实例、GLS 最多 20 次迭代；
正式运行请勿设置 `MAE_SMOKE_TEST` 和 `MAE_OFFLINE`。

## 正式运行，退出终端后继续

推荐直接使用上一节的 `run_mae_tsp200.sh`。手工启动时，确保计算节点分配了
足够 CPU、网络可访问 LLM，且 `DEEPSEEK_API_KEY` 已在 shell 环境中设置。
使用一个**未出现过**的运行名：

```bash
cd /public2/home/rentao/eoh/examples/tsp_gls_numba
PY=/public2/home/rentao/miniconda3/envs/mae_eoh/bin/python
export MAE_RUN_NAME=mae_semantic_tsp200_seed2026_run001
export MAE_USE_ENVIRONMENT=1
mkdir -p results_mae
nohup "$PY" -u runMAEoH.py > "results_mae/${MAE_RUN_NAME}.console.log" 2>&1 < /dev/null &
echo $! > "results_mae/${MAE_RUN_NAME}.pid"
```

查看状态：

```bash
tail -n 30 "results_mae/${MAE_RUN_NAME}.console.log"
ps -p "$(cat "results_mae/${MAE_RUN_NAME}.pid")" -o pid,stat,etime,cmd
```

若旧进程已经结束且需要从 v2 检查点续跑，保持同一个 `MAE_RUN_NAME`，另设
`MAE_RESUME_FROM=/绝对路径/checkpoints/generation_XXXX.json` 后再启动。
不要用 v1 的检查点，也不要同时启动两个写同一个运行目录的进程。

## 中间数据记录和汇总脚本

训练引擎本身持续写入 `metrics/`、`telemetry/`、`velocities/`、`cuts/`、
`lineage/`、`failures/` 和 `checkpoints/`。新增三个独立工具：

- `validate_tsp_training_data.py`：验证字段、路线、cost、距离矩阵和数据哈希。
- `monitor_mae_run.py`：每60秒保存进程状态、代数、最优值、有效率、评测数、
  内存、检查点、失败数和磁盘占用。
- `summarize_mae_run.py`：运行中或结束后生成代、智能体、算子、经验来源和阶段
  耗时的 CSV，以及一份 `intermediate_summary.json`。

正式启动脚本会自动运行监控器，最新状态位于：

```text
results_mae/<RUN_NAME>/monitor/latest_snapshot.json
```

随时生成阶段汇总：

```bash
"$PY" summarize_mae_run.py \
  --run-dir "results_mae/${MAE_RUN_NAME}"
```

输出位于 `results_mae/<RUN_NAME>/reports/`，重复执行会刷新汇总，不影响训练。

## 输出及 EoH 测试

每个运行位于 `results_mae/<MAE_RUN_NAME>/`：

- `best/best_heuristic.py`：可直接作为 EoH `heuristic.py` 的完整函数。
- `agents/generation_XXXX.json`：每代单个当前程序、`pbest` 和速度库。
- `velocities/experience_events.jsonl`、`velocities/generation_XXXX/`：经验创建、刷新及概率快照。
- `cuts/generation_XXXX/`：cut 内容、`lambda`、截断后概率和邻居保留名额。
- `metrics/generation_metrics.csv`：全局历史最佳、当前程序群均值与评测次数。
- `metrics/experience_source_metrics.jsonl`：本地、全局、左右邻居经验进入 cut、产生合法代码和胜出程序的次数。
- `lineage/experience_edges.jsonl`：经验与有效子代的关联。
- `checkpoints/`、`failures/`、`telemetry/`：恢复、失败原因和过程事件。

## 自定义非均匀 TSP 实例

可以使用 [generate_custom_tsp.py](./generate_custom_tsp.py) 生成新的、可复现的
坐标和距离矩阵数据。默认配置生成 600 个训练实例和 100 个验证实例，城市数从
50、75、100、125、150、200 中取值，分布比例为：均匀 25%、聚类 30%、环形 15%、
网格扰动 15%、带状走廊 15%。输出 PKL 与原始 `TSPAEL64.pkl` 完全相同，只保留
`coordinate`、`optimal_tour`、`distance_matrix`、`cost` 四个字段。

此步骤**不求最优路线**。生成文件仍只有上述四个字段，其中 `optimal_tour` 和
`cost` 都是 `None`，不能直接作为 EoH/MAE 训练集；后续求解器完成后只填充这两个
字段，不增加额外元数据字段。

```bash
cd /public2/home/rentao/eoh/examples/tsp_gls_numba
PY=/public2/home/rentao/miniconda3/envs/mae_eoh/bin/python

# 先用小数据检查生成流程
"$PY" generate_custom_tsp.py \
  --split both --train-count 12 --validation-count 6 \
  --output-dir TrainingData

# 正式生成推荐规模
"$PY" generate_custom_tsp.py --split both --seed 2026

# 生成一批 200 个实例，每个实例包含 200 个城市
"$PY" generate_custom_tsp.py \
  --split train --train-count 200 --sizes 200 --seed 2026 \
  --output-dir TrainingData --output-name custom_TSP200_seed2026.pkl
```

默认输出到 `TrainingData/`：

```text
custom_train_seed2026.pkl
custom_validation_seed2027.pkl
```

原来的 `TrainingData/TSPAEL64.pkl` 和 TSPLIB 测试集不会被覆盖。

## 用 Concorde 补充最优路线和 cost

你下载的 `D:\tsp求解器\concorde` 当前是 Concorde 源码目录，不是可直接运行的
`concorde.exe`。Concorde 的 README 说明，精确求解需要先配置 QSopt 或 CPLEX，
再编译 `TSP/concorde`。编译得到可执行文件后，使用
[solve_custom_tsp_concorde.py](./solve_custom_tsp_concorde.py) 处理上面的四字段 PKL。

```powershell
$PY = "C:\Users\rt\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$CONCORDE = "D:\tsp求解器\concorde\TSP\concorde.exe"
$INPUT = "TrainingData\custom_TSP200_seed2026.pkl"
$OUTPUT = "TrainingData\custom_TSP200_seed2026_solved.pkl"

& $PY solve_custom_tsp_concorde.py `
  --concorde $CONCORDE `
  --input $INPUT `
  --output $OUTPUT `
  --scale 1000000
```

脚本会为每个实例临时生成 TSPLIB `.tsp` 文件，调用 Concorde，读取 `.sol` 中的
路线，并用原始浮点距离矩阵重新计算 `cost`；每完成一个实例就保存一次输出文件，
中断后重复同一命令会跳过已经完成的实例。由于输入坐标在 `[0,1]`，默认先乘
`1,000,000` 转为 Concorde 的整数 EUC_2D 坐标；因此得到的是高精度缩放问题的
精确解，路线长度仍按项目当前浮点距离矩阵记录。

测试集评测建议复制到**独立目录**，不覆盖仓库原有 `evaluation/heuristic.py`：

```bash
cd /public2/home/rentao/eoh/examples/tsp_gls_numba
PY=/public2/home/rentao/miniconda3/envs/mae_eoh/bin/python
EVAL_DIR=$(mktemp -d "$PWD/evaluation_semantic_XXXXXX")
cp evaluation/runEval.py evaluation/evaluation.py "$EVAL_DIR/"
cp "results_mae/${MAE_RUN_NAME}/best/best_heuristic.py" "$EVAL_DIR/heuristic.py"
cd "$EVAL_DIR"
nohup "$PY" -u runEval.py > run.log 2>&1 < /dev/null &
echo $! > run.pid
```

完成后查看 `"$EVAL_DIR/results.txt"`。该测试入口仍使用原 EoH 的 29 个
TSPLIB 实例和原有打分方式。旧版五位置算法及其结果不受新版运行影响。
