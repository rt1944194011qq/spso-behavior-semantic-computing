# S-PSO 语义模块搜索实现说明

本目录中的实现把 EoH 的“生成完整程序”流程扩展为有类型的模块序列搜索。EoH 的 `BaseProblem.evaluate()` 和任务评测逻辑继续作为统一的函数编译与目标函数入口，S-PSO 只负责生成合法模块序列。

## 目录职责

```text
eoh/eoh/src/spso/
  models.py       模块选择、粒子位置等不可变数据对象
  registry.py     槽位、模块、参数约束
  velocity.py     分类速度更新与 cut 集
  selector.py     LLM JSON 选择与严格校验
  compiler.py     编译器协议
  engine.py       初始化、pbest/gbest、缓存、检查点、恢复与并行流水线

eoh/examples/tsp_gls_numba/
  spso_task/modules.py   TSP GLS 五槽模块库
  spso_task/compiler.py  五槽序列到 update_edge_distance 的确定性编译器
  runSPSO.py             训练入口
  evaluateSPSO.py        TSPLIB 测试入口
```

## TSP GLS 接口契约

所有候选函数必须具有如下公开入口：

```text
update_edge_distance(edge_distance, local_opt_tour, edge_n_used)
```

输入分别是 `n×n` 原始距离矩阵、长度为 `n` 的城市回路和 `n×n` 边历史惩罚次数矩阵。返回值必须是新的 `n×n` 有限、非负、对称距离矩阵。编译器生成的函数会检查形状、有限性、城市编号和最终矩阵性质；不符合条件的候选由 EoH 的超时/异常评测路径判为无效。

五个槽为：

```text
E 候选边 | F 边特征 | H 历史调节 | T 分数变换 | W 对称写回
```

经典 GLS 函数的精确表示是：

```text
tour_edges | edge_length | inverse_count | identity | symmetric_add_all(lambda=1.0)
```

这组序列应作为编译器的等价性回归基准。

## 运行

将 `runSPSO.py` 顶部的 `OFFLINE` 设为 `True`，并把实验参数调小后运行离线烟测：

```text
cd eoh/examples/tsp_gls_numba
python runSPSO.py
```

在线运行时，在 `runSPSO.py` 顶部填写配置：

```text
DEEPSEEK_ENDPOINT = "api.deepseek.com"
DEEPSEEK_API_KEY = "..."
DEEPSEEK_MODEL = "deepseek-chat"
python runSPSO.py
```

LLM 只能从每个槽的 cut 集中选择模块并返回有限参数。响应不是可执行代码；代码由任务编译器确定性生成。`runSPSO.py` 中的 `NUM_SAMPLERS` 控制并行采样/LLM 请求数，`NUM_EVALUATORS` 控制并行 GLS 评估数；两者默认均为 16。恢复实验时，将 `RESUME_FROM` 设置为对应的 `checkpoints/generation_XXXX.json`。

运行时采用两层流水线：采样线程负责 cut 集整理、LLM 选择和候选编译；评估线程池负责带硬超时的 GLS 评分。每一代开始时先冻结当前粒子、pbest 和 gbest 快照，因此并行不会改变一代内部的 PSO 语义；代内候选完成后统一更新粒子状态。

原来的 `runEoH.py` 仍然是完整程序生成基线，S-PSO 通过独立的 `runSPSO.py` 运行，便于按相同任务评测器做比较。
