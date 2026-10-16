# MAE v3 进化流程与提示词

旧版本快照：GitHub 分支 `codex/mae-before-operator-rework`，提交 `71e451e`。
这里描述的新流程只作用于 `eoh.mae.MAEEngine`；原 EoH 和
`LegacyMAEEngine` 不使用这个逐次替换池。

## 每代单智能体流程

1. 冻结当前程序、个人最优、全局最优和左右邻居最优；比较程序差异，衰减并更新速度库。
2. 对总共 `G` 代、第 `g` 代（`1 <= g <= G`），设
   `t=(g-1)/(G-1)`（仅一代时 `t=0`），
   `P(H1)=0.20+0.60[1+cos(pi*t)]/2`。先把 `智能体数 × P(H1)` 随机舍入成
   H1 配额，再无放回抽取智能体。10 个智能体时，首代为 8 个 H1，末代为 2 个 H1。
   每个智能体本代只使用一种引导；单独禁用 H1 或 H2 时所有智能体使用另一种。
3. 在 cut 抽取中，H1 只临时放大个人最优来源的经验，H2 只临时放大全局最优来源；
   邻居预留、数量校准和概率截断仍有效，速度库保存的原概率不变。
4. cut 中每条差异与本智能体旧当前程序生成一个完整程序，并按原 TSP 评测器评分。
   旧当前程序直接加入基础池并复用已知分数。`cut=4` 且四条全有效时，基础池大小为 5。
   基础池不足两个有效程序时，最多再尝试 `fill_retries+1` 轮 H0；若仍不足，
   H3/H4 会记录跳过原因，H5/H6 继续执行。
5. 依次运行 H3、H4、H5、H6，每个启用的算子只生成并评测一个候选。
   当前算子读到的是之前算子更新后的池。候选只有在**严格优于当前池最差适应度**时
   才替换那个最差者；无效、重复、同分和较差候选都不入池。最后从池中选最优作为
   下一代当前程序，个人最优和全局最优单独更新。

正常每智能体每代提出 `N+4` 个新程序，其中 `N` 是 cut 数；旧当前程序不重复评测。
默认 `N=4`、10 智能体、20 代时约有 1600 个演化提案，加上初始化 100 个。
成功率、H0 补试、重复代码和跨候选评测缓存会改变独立评测次数。

## 实际代码生成提示词

所有算子都会附加共同约束：任务说明、`update_edge_distance` 的完整函数模板、
TSP 输入读写规则，以及“先用花括号给出一句算法描述，然后给完整 Python 代码”。
H3/H4 的两个父代由代码按适应度选出，H5/H6 的单父代由固定种子随机选出；
模型不会决定父代或是否替换。

| 算子 | 提交给模型的核心英文指令 | 评测前的代码约束 |
|---|---|---|
| H1 | `Give extra attention to useful personal-best experience while preserving the current program's working behavior.` 接着给出选中的差异、参照描述和当前完整代码。 | 完整程序必须符合模板。 |
| H2 | `Give extra attention to useful global-best experience while adapting it to this agent's current program.` 接着给出选中的差异、参照描述和当前完整代码。 | 完整程序必须符合模板。 |
| H3 | `These are the two best programs in the current evaluated pool. Identify their common effective backbone, then create one new program with similar algorithmic behavior. It must not be a verbatim copy of either parent.` | 去重、模板和输入契约校验。 |
| H4 | `These are the two worst programs in the current evaluated pool. Use them as contrastive examples: identify a likely weakness and propose one new algorithm with a substantially different form from both parents. Do not merely combine their weak behavior. Preserve the task contract.` | 与两父代相比，忽略数值常量后的 AST 结构必须不同。 |
| H5 | `Perform a STRICT parameter-only mutation: copy the complete parent program and change only one or more existing numeric literals.` 不允许添加、删除或改变语句、循环、分支、变量、数组操作和返回逻辑。 | 与父代相比，忽略数值常量后的 AST 结构必须相同；失败时仍可用已有的自动纯参数修复。 |
| H6 | `Create one modified version by changing its internal algorithmic structure: change at least one branch, loop, computation, or edge-update rule. A numeric-literal-only change is insufficient.` | 与父代相比，忽略数值常量后的 AST 结构必须不同。 |

H3 的“共同骨架”参考了原 EoH `e2` 的写法；H4 的不同形式参考 `e1`；
H5 的参数设置参考 `m2`；H6 的结构修改参考 `m1`，并与项目当前的 TSP 函数契约对齐。
完整动态提示词实现在 `eoh/eoh/src/eoh/mae/generator.py`。

## 核对中间数据

- `cuts/generation_XXXX/agent_XXX.json`：本代 H1/H2 选择、cut、概率和 `lambda`。
- `metrics/base_pools.jsonl`：每个智能体评测 cut 子代后，包含旧当前程序的基础池。
- `metrics/pool_decisions.jsonl`：H3–H6 各自的父代、候选分数、当时最差分数、
  替换结果以及替换前后的池。
- `metrics/agent_metrics.jsonl`：引导算子、当前/个人最优、池大小和算子入池次数。
- `metrics/operator_metrics.jsonl`：每算子的生成、有效、入池和最终胜出数量。
- `checkpoints/generation_XXXX.json`：v3 恢复状态；v1/v2 检查点不兼容。

这种严格精英筛选保证本智能体当前分数不因换代而变差，但 H4 的差异候选也必须
打败最差者才能留下。是否改善全局搜索，需要用相同评测预算和多个种子对照实验检验。
