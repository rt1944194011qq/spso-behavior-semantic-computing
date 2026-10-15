# MA-EoH v1：TSP-GLS-numba 历史运行说明

> 仅作为旧实验记录保留。当前 `runMAEoH.py` 已切换到语义经验库 v2；下面的
> 参数和命令不能原样用于新版，请以 `README_MAE.md` 为准。

## 首次验证

打开 runMAEoH.py，保持：

~~~python
SMOKE_TEST = True
OFFLINE = True
~~~

在项目根目录运行：

~~~powershell
& .\envs\spso_semantic\python.exe .\eoh\examples\tsp_gls_numba\runMAEoH.py
~~~

离线模式不调用大模型，但会运行完整的多智能体、H0–H5、并行评测、GLS 和结果保存流程。

## 在线实验

在 runMAEoH.py 中修改运行模式；密钥不要写入源码：

~~~python
SMOKE_TEST = False
OFFLINE = False
RUN_NAME = "mae_full_seed2024_run001"
~~~

Linux 服务器上在当前 shell 中设置密钥：

~~~bash
export DEEPSEEK_API_KEY='你的实验室密钥'
~~~

正式默认配置为 10 个智能体、每个位置 5 个完整程序、20 代、64 个训练实例。H1/H2 每代分别执行 H3/H4/H5，因此每个智能体每代最多生成 6 个子代。

NUM_LLM_WORKERS 控制 LLM 网络请求并发，NUM_EVALUATORS 控制 EoH 隔离评测并发。在 30 核节点上建议先设为 16，确认负载稳定后再提高评测并发。

## 防止旧实验污染

每个 RUN_NAME 只能启动一次。若输出目录已经存在，程序会拒绝覆盖。继续运行时设置 RESUME_FROM；重新实验时更换 RUN_NAME，例如 run002。

## 结果位置

结果保存在：

~~~text
results_mae/<RUN_NAME>/
├── agents/          每代各智能体位置和 pbest
├── archive/         所有合法完整程序
├── best/            best.json 和 best_heuristic.py
├── checkpoints/     可恢复检查点
├── cuts/            每代 H1/H2 cut 集
├── failures/        非法代码和失败阶段
├── lineage/         个体和父子谱系
├── metrics/         每代分数、有效率和 H 路径统计
├── prompts/         在线模式的完整提示词和响应
├── telemetry/       结构化中间事件、阶段耗时和线程进度
├── velocities/      每个智能体的 H1/H2 概率速度
├── config.json
├── metadata.json
└── run_log.txt
~~~

## 将最优函数交给原 EoH 测试框架

先安装最优函数：

~~~powershell
& .\envs\spso_semantic\python.exe .\eoh\examples\tsp_gls_numba\install_best_mae.py .\eoh\examples\tsp_gls_numba\results_mae\<RUN_NAME>
~~~

再运行原测试集：

~~~powershell
Set-Location .\eoh\examples\tsp_gls_numba\evaluation
& ..\..\..\..\envs\spso_semantic\python.exe .\runEval.py
~~~

runEval.py 会使用 evaluation/heuristic.py，并把测试结果写入 evaluation/results.txt。

## 中间状态与论文数据

运行过程中可以在另一个终端查看状态：

~~~powershell
& .\envs\spso_semantic\python.exe .\eoh\examples\tsp_gls_numba\inspect_mae_run.py .\eoh\examples\tsp_gls_numba\results_mae\<RUN_NAME>
~~~

主要结构化文件：

- metrics/generation_metrics.csv：每代最优值、均值、标准差、有效率、缓存和耗时，可直接用于绘图。
- metrics/agent_metrics.jsonl：每个智能体每代的位置分数、pbest、速度大小和子代存活数。
- metrics/operator_metrics.jsonl：六条 H 路径的生成数、合法率、存活率和最优适应度。
- telemetry/velocity_updates.jsonl：速度更新所用随机数、系数、pbest/gbest/邻居来源和最终概率。
- telemetry/cut_decisions.jsonl：每个程序进入或未进入 cut 集的具体原因。
- telemetry/selections.jsonl：LLM 从 cut 集选择了哪些程序、是否触发回退。
- telemetry/candidate_lifecycle.jsonl：每个候选的生成、解析、验证、接受或拒绝过程。
- telemetry/evaluations.jsonl：评测耗时、缓存命中、平均 fitness 和逐实例 gap/路径长度。
- telemetry/agent_transitions.jsonl：每个智能体更新前后的位置、淘汰程序和 pbest 变化。
- telemetry/stage_timings.jsonl：初始化、速度、LLM、验证和评测各阶段耗时。
- telemetry/progress.jsonl：并发任务进度以及长任务心跳，可用于判断程序是否卡死。
- failures/failures.jsonl：失败阶段、操作路径、父代、cut 集和失败原因。

DETAILED_LOGGING 控制详细事件；RECORD_FULL_PROMPTS 和 RECORD_FULL_RESPONSES 控制是否保存完整 LLM 内容。正式实验建议保留详细日志，但包含敏感实验内容时可以关闭完整 prompt/response，只保留哈希和字符数。
