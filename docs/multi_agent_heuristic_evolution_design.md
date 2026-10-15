# 多智能体协同自演化启发式设计方案

> 历史 v1 设计记录：本文描述五个完整程序/智能体的旧方案。当前默认实现已改为
> 单程序 + 持久语义经验库；参数、运行步骤和结果格式以
> `eoh/examples/tsp_gls_numba/README_MAE.md` 的 v2 说明为准。

## 1. 文档定位

本文是项目后续实现、组会汇报和论文方法章节的统一设计说明。方法暂命名为 **MA-EoH**（Multi-Agent Evolution of Heuristics）。

当前基线是 EoH 的 TSP-GLS-numba 任务。新方案保留 EoH 的 LLM 接口、问题接口、隔离评测、GLS 评测逻辑和结果验证方式，但将单一种群演化扩展为具有本地记忆、邻居协作和全局协作的多智能体演化。

本文只规定方案，不包含具体实现代码。

## 2. 研究动机

前期固定槽位方案存在以下局限：

- 槽位语义由人工预先划分，限制启发式结构表达能力。
- 不同槽位之间有强耦合，单独分析一个槽位未必能解释整体性能。
- 初始候选可能已经取得很好的分数，后续局部变化难以体现持续学习。
- LLM 生成的新模块容易重复、非法或发生语义漂移。
- 模块历史、来源和跨代经验没有以完整程序为中心保存。

因此，新方案不再把位置解释为五个固定模块槽位，而是把每个智能体的位置解释为五个完整 GLS 启发式函数组成的候选集合。

研究叙事为：

> 历史记录形成概率记忆，概率记忆决定经验激活；智能体通过本地最优、邻居最优和全局最优进行协同；LLM 根据激活的经验生成新启发式程序；程序通过 EoH 评测后重新写回记忆。

应准确描述 EoH：EoH 已有候选种群和短期历史，但没有显式智能体身份、个体本地最优、邻居拓扑和概率化程序记忆。MA-EoH 将这些信息显式化。

## 3. EoH 评测接口必须保持不变

对于当前 TSP-GLS-numba 任务，LLM 最终生成的是完整函数：

~~~python
def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    ...
    return augmented_edge_distance
~~~

输入输出合同：

| 项目 | 要求 |
|---|---|
| edge_distance | n×n 距离矩阵，不能被原地修改 |
| local_opt_tour | 长度为 n 的局部最优回路节点序列 |
| edge_n_used | n×n 的边使用次数或惩罚统计矩阵 |
| 返回值 | 新的 n×n 数值矩阵 |
| 数值 | 不能有 NaN 或 Inf |
| 结构 | 通常要求非负、对称、合理的对角线 |

候选必须经过语法检查、接口检查、小矩阵 smoke test、输入不变性检查和原 EoH/TSPGLS 隔离评测。MA-EoH 不改变原始问题的打分定义。

## 4. 多智能体状态

设有 M 个智能体，第 i 个智能体在第 t 代的状态为：

~~~text
Agent_i(t) = {
    position X_i(t),
    velocity V_i(t),
    pbest_i(t),
    neighborhood_i(t)
}
~~~

### 4.1 位置

每个智能体的位置包含五个完整启发式程序：

~~~text
X_i(t) = {
    program_i,1,
    program_i,2,
    program_i,3,
    program_i,4,
    program_i,5
}
~~~

这五个元素不是五个固定语义槽，而是五个可以独立运行、独立评测的完整函数。

### 4.2 记忆和邻居

- pbest_i：智能体 i 历史上最好的完整程序。
- gbest：所有智能体的 pbest 中最好的完整程序。
- 邻居：默认采用环形拓扑，智能体 i 的左右邻居为 left(i) 和 right(i)。
- 邻居本地最优：pbest_left(i) 和 pbest_right(i)。

目标是最小化平均相对误差；无效程序不参与最优更新。

## 5. 程序个体和谱系

位置元素必须保存完整元数据：

~~~text
ProgramIndividual = {
    individual_id,
    description,
    code,
    code_hash,
    fitness,
    valid,
    generation,
    agent_id,
    source_operator,
    guidance_operator,
    variation_operator,
    operator_path,
    parent_ids,
    source_cut_ids,
    prompt_hash,
    failure_stage,
    failure_reason
}
~~~

字段含义：

- description：LLM 生成代码前给出的一句话算法描述。
- code_hash：用于去重、缓存和追踪程序是否真正改变。
- guidance_operator：H1 或 H2，表示受到本地或全局记忆引导。
- variation_operator：H3、H4 或 H5，表示具体变异方法。
- operator_path：例如 H1->H3、H2->H5。
- parent_ids：交叉时记录两个父代，参数修改时记录一个父代。
- source_cut_ids：记录本次使用的 cut 集候选。
- failure_stage：非法候选具体在哪一步失败。

初始候选使用 H0 标记。用户描述中曾出现 H6，但当前明确规则实际是 H0–H5；本方案暂不定义 H6，避免论文和代码编号不一致。

## 6. H0–H5 策略

| 策略 | 作用 | 是否产生新代码 |
|---|---|---|
| H0 | 不使用历史引导，生成或直接选择候选 | 初始化阶段生成代码 |
| H1 | 增强本地最优 pbest_i 的概率 | 主要改变候选激活概率 |
| H2 | 增强全局最优 gbest 的概率 | 主要改变候选激活概率 |
| H3 | 随机选择两个父代，生成语义相似的组合程序 | 是 |
| H4 | 选择两个差异较大的父代，生成探索性程序 | 是 |
| H5 | 根据描述和代码修改参数，不改变主体接口 | 是 |

H1、H2 是记忆引导层，回答“向哪类历史经验靠近”；H3、H4、H5 是程序变异层，回答“如何生成新程序”。

因此每个后代应记录一条完整路径：

~~~text
H1 -> H3   本地最优引导 + 相似交叉
H1 -> H4   本地最优引导 + 差异交叉
H1 -> H5   本地最优引导 + 参数修改
H2 -> H3   全局最优引导 + 相似交叉
H2 -> H4   全局最优引导 + 差异交叉
H2 -> H5   全局最优引导 + 参数修改
~~~

H1/H2 不是直接复制 pbest 或 gbest，而是先改变速度和 cut 集，再由 H3/H4/H5 生成代码。

## 7. 概率速度和集合运算

速度不是连续向量，而是带概率的程序集合：

~~~text
V_i,h(t) = {(program_id, probability)}
~~~

建议保存两个独立通道：

- V_i,H1：本地最优优先。
- V_i,H2：全局最优优先。

这样 H1 和 H2 都在同一个原始位置快照上计算，互不污染。

集合运算：

- 标量乘法：概率乘以系数，再裁剪到 [0, 1]。
- 集合加法：相同程序合并并取最大概率，不同程序全部保留。
- 集合减法：保留左集合中不在右集合的程序。
- 程序相同性：以 code_hash 或稳定 individual_id 判断，不以格式化后的字符串判断。

令 X_i 为当前位置，P_i 为 pbest，G 为 gbest，P_l 和 P_r 为左右邻居 pbest。对 h 属于 {H1, H2}：

~~~text
V_i,h(t+1) =
    omega ⊗ V_i,h(t)
  ⊕ c_p,h r_p ⊗ ({P_i} \ X_i)
  ⊕ c_g,h r_g ⊗ ({G}   \ X_i)
  ⊕ c_l r_l ⊗ ({P_l}   \ X_i)
  ⊕ c_r r_r ⊗ ({P_r}   \ X_i)
~~~

其中 omega 是历史速度保留系数，r 是可复现随机数，⊗ 是概率集合标量乘法，⊕ 是概率集合加法。

建议初始配置：

~~~text
omega = 0.50
c_l = 0.40
c_r = 0.40

H1: c_p,H1 = 0.90, c_g,H1 = 0.25
H2: c_p,H2 = 0.25, c_g,H2 = 0.90
~~~

这些只是实验起点，不能直接当作最优参数。

## 8. Cut 集

速度更新后，将当前五个程序与概率达到阈值的历史程序组成 cut 集：

~~~text
C_i,h(t) = X_i(t) ∪ {
    program_id in V_i,h(t) | probability >= alpha
}
~~~

建议 alpha 默认取 0.5 并做成配置项。cut 集必须始终保留当前位置中的五个程序，并设置 max_cut_size 防止 prompt 过长。

每个 cut 项至少记录：

~~~text
program_id, probability, source, description, code_hash, fitness
~~~

LLM 先根据描述和概率从 cut 集中选择有限数量的候选，再读取被选候选的完整代码。这样可以减少上下文长度和调用时间。

## 9. 一代演化流程

### 9.1 初始化：H0

每个智能体先生成 2K 个候选。当前 K=5，因此先生成 10 个：

1. LLM 先给出一句话算法描述。
2. LLM 根据任务合同生成完整函数。
3. 解析描述和代码。
4. 进行语法、接口、数值和隔离运行检查。
5. 调用原 EoH/TSPGLS 评测器。
6. 去除无效和重复候选。
7. 按适应度保留前 5 个，形成第 0 代位置。
8. 更新每个智能体的 pbest，再选出 gbest。

初始化候选统一标记为 H0。

### 9.2 后续代

每代使用同步快照：

1. 固定所有智能体的当前位置、pbest、gbest、邻居 pbest 和旧速度。
2. 根据快照计算 H1、H2 两个速度。
3. 根据速度生成 H1、H2 cut 集。
4. LLM 从 cut 集中选择候选。
5. 依次执行 H3、H4、H5 产生子代。
6. 并行解析、验证、编译和评测。
7. 合并旧位置与有效子代。
8. 每个智能体保留适应度最好的 5 个，多样性作为平分时的第二排序指标。
9. 更新各智能体 pbest。
10. 所有智能体完成后统一更新 gbest。
11. 保存 checkpoint、速度、cut 集、谱系和指标。

不能让某个线程完成后立即修改 gbest，否则不同智能体看到的全局最优时间点不一致，H2 就不再是可复现实验。

## 10. LLM 生成协议

每次产生新代码必须先有一句描述，再有完整函数：

~~~text
一句话算法描述
~~~python
def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    ...
    return augmented_edge_distance
~~~
~~~

H3 要求 LLM 识别两个父代的共同策略，再生成保留主要行为但有组合变化的新函数。

H4 要求 LLM 明确两个父代的差异，再生成探索性新函数，不能只复制其中一个父代。还要使用 code_hash、描述相似度或 AST 特征做差异检查。

H5 只允许修改数值参数或明确允许的参数表达式。最可靠的方式是让 LLM 返回参数补丁，由本地程序应用补丁并检查 AST 结构。如果暂时直接让 LLM 返回完整代码，也必须检查父子 AST 结构变化。

## 11. 合法性、失败处理和可靠性

建议检查顺序：

1. JSON 或代码块解析。
2. 描述存在性检查。
3. Python 语法检查。
4. 函数名和参数签名检查。
5. 允许导入和安全检查。
6. 小型矩阵 smoke test。
7. 检查输入是否被原地修改。
8. 检查输出形状、有限性、非负性和对称性。
9. code_hash 去重。
10. H4 差异性或 H5 结构检查。
11. EoH 隔离子进程硬超时评测。

失败阶段统一使用：

~~~text
json_parse
missing_description
syntax_error
signature_error
contract_shape
input_mutation
non_finite_output
duplicate_code
similarity_violation
compile_failure
evaluation_timeout
evaluation_failure
~~~

非法子代不能让线程退出。应记录失败原因，并用旧位置中的有效程序填充，保证每个智能体每代始终保留五个元素。

## 12. 多线程方案

采用“代际同步、阶段内并行”：

~~~text
同步快照
  ├─ 并行计算 H1/H2 速度和 cut 集
  ├─ 并行执行 LLM 请求
  ├─ 并行解析和验证
  └─ 并行执行 EoH 隔离评测
统一提交子代、更新 pbest/gbest、保存 checkpoint
~~~

配置分开设置：

- num_llm_workers：LLM 请求并发数。
- num_evaluators：评测进程并发数。
- max_inflight_candidates：等待中的候选上限。

可以复用 EoH 的 InterfaceLLM、_eval_with_timeout 和 TSPGLS 评测逻辑，但不直接复用 EoH 的稳态异步主循环，因为 MA-EoH 需要严格的邻居和 gbest 快照。

## 13. 项目结构规划

核心算法不应依赖 TSP，TSP 内容通过任务适配器提供：

~~~text
eoh/eoh/src/eoh/mae/
├── __init__.py
├── config.py                  配置、种子、并发参数
├── models.py                  智能体、程序个体、速度项
├── archive.py                 程序档案、pbest、gbest、缓存
├── set_algebra.py             加权集合运算
├── topology.py                环形邻居和可扩展拓扑
├── velocity.py                H1/H2 速度更新
├── cut.py                     cut 集生成和裁剪
├── selection.py               LLM 选择和候选保留
├── validation.py              合法性和 AST 检查
├── evaluator.py               EoH 评测器适配
├── engine.py                  代际同步主循环
├── checkpoint.py              保存和恢复
├── reporting.py               指标、谱系和日志
├── llm_program_generator.py   prompt 和代码提取
└── operators/
    ├── h0_initialize.py
    ├── h1_local_guidance.py
    ├── h2_global_guidance.py
    ├── h3_similar_crossover.py
    ├── h4_different_crossover.py
    └── h5_parameter_search.py

eoh/examples/tsp_gls_numba/
├── runMAEoH.py                TSP 实验入口
├── mae_tsp_adapter.py         TSP 输入输出合同
├── evaluateMAE.py             测试集评估
├── install_best_mae.py        导出 best_heuristic.py
└── README_MAE.md
~~~

原有 EoH 文件尽量保持不变，只通过适配层复用其公共能力。

## 14. EoH 复用边界

可以复用：

- InterfaceLLM：远程或本地 LLM 接口。
- BaseProblem：问题模板、任务描述和评测入口。
- Evolution 中的函数规范、prompt 拼接和代码提取思想。
- _eval_with_timeout：隔离子进程硬超时。
- TSPGLS、guided_local_search 和 solve_instance。
- 原 EoH 的日志和结果目录工具。

不直接复用：

- EoH 的异步稳态主循环：不符合同步邻居和全局快照。
- 只按 objective 去重的 population_management：会删除适应度相同但代码不同的程序。
- EoH 固定的父代抽样：无法表达 H1/H2 与 H3/H4/H5 的完整谱系。

## 15. 结果目录

每次实验使用独立 run_id：

~~~text
results_mae/<run_id>/
├── metadata.json
├── config.json
├── run_log.txt
├── archive/programs.jsonl
├── agents/generation_0000.json
├── velocities/generation_x/agent_x_H1.json
├── velocities/generation_x/agent_x_H2.json
├── cuts/generation_x/agent_x_H1.json
├── cuts/generation_x/agent_x_H2.json
├── prompts/generation_x.jsonl
├── lineage/individuals.jsonl
├── lineage/edges.jsonl
├── failures/failures.jsonl
├── metrics/generation_metrics.jsonl
├── checkpoints/generation_x.json
└── best/
    ├── best.json
    ├── best_heuristic.py
    └── best_routes.json
~~~

每个候选都应能追溯到产生代数、智能体、H 引导、变异操作、父代、cut 集、prompt 哈希、代码哈希、适应度和失败原因。

## 16. 评测预算和缓存

设智能体数量为 M，位置容量为 K=5，代数为 T：

- 初始化候选上限约为 2KM。
- 后续每代的候选数取决于 H1/H2 是否都执行、H3/H4/H5 是否全部启用以及重试次数。
- 评测预算必须包含 LLM 重试、非法候选重试和缓存命中情况。

缓存键至少包含：

~~~text
code_hash + problem_fingerprint + evaluation_config
~~~

只有问题实例、GLS 参数、随机种子策略和评测协议完全一致时才允许复用适应度。缓存命中也要记录，不能把缓存结果误认为新的独立评测。

开发阶段先使用 3 个实例和短超时；正式实验使用完整训练集；最终测试与 EoH baseline 统一实例、GLS 参数、时间限制、随机种子和总评测预算。

## 17. 测试计划

### 单元测试

- 加权集合相同元素取最大概率，不同元素全部保留。
- 集合差集和概率裁剪正确。
- H1 本地权重高于 H2 本地权重。
- H2 全局权重高于 H1 全局权重。
- 环形拓扑左右邻居正确。
- cut 集始终包含当前位置五个程序。
- H5 不改变函数签名和不允许的 AST 结构。
- 谱系中的父代、H 路径和代码哈希一致。

### 集成测试

用 FakeLLM 和 FakeProblem 跑 3 个智能体、2 代，验证：

- 每个智能体每代有五个有效位置元素。
- pbest 适应度不会变差。
- gbest 只在整代提交后更新。
- 断点恢复后随机状态和结果可复现。
- 评测线程失败不会使主循环静默卡死。

### TSP 冒烟测试

使用当前 TSPGLS 和 3 个实例验证：

- H0 候选可以生成、编译和评测。
- H1/H2 可以生成 cut 集。
- H3/H4/H5 的有效率和失败阶段可以统计。
- 最优函数可以导出为 best_heuristic.py，并再次交给原 EoH 测试入口。

## 18. 论文实验和消融

主实验建议包含：

- EoH baseline。
- 单智能体版本。
- 完整 MA-EoH。
- 去掉 pbest。
- 去掉 gbest。
- 去掉邻居信息。
- 环形拓扑与全连接拓扑。
- 仅 H1、仅 H2。
- 去掉 H3、H4 或 H5。
- 输入描述与不输入描述。
- 不同 alpha 阈值。

建议报告：

- best-so-far 随评测次数变化的曲线。
- 平均值、中位数、标准差和最差实例 gap。
- 有效子代率和失败阶段比例。
- 各 H 路径的生成率、存活率和最终贡献。
- AST 多样性和去重后的程序数量。
- pbest、邻居 pbest、gbest 的贡献。
- LLM 请求次数、缓存命中率、总时间和评测时间。

## 19. 风险和控制

1. 过早收敛：加入 H4、代码多样性保留、环形拓扑和 novelty tie-break。
2. H3/H4 非法率高：保留父代、重试、失败统计和旧位置填充。
3. prompt 过长：限制 cut 大小，先用 ID/描述筛选，再传少量完整代码。
4. H5 结构漂移：采用 AST 结构检查或结构化参数补丁。
5. 评测不公平：EoH 和 MA-EoH 统一所有评测协议和总预算。
6. 线程不可复现：按 seed、generation、agent_id、operator 派生随机数，并使用代际同步。
7. 描述与真实性能不一致：描述只用于语义选择和消融，最终保留由 EoH 适应度决定。

## 20. 汇报用一句话总结

> MA-EoH 将每个智能体表示为五个完整启发式程序组成的位置集合，将带概率的程序历史表示为速度，并通过本地最优、邻居最优和全局最优共同更新概率记忆；LLM 在 H0–H5 策略下完成候选选择、交叉和参数修改，所有程序最终通过原 EoH 的统一 GLS 评测体系验证。
