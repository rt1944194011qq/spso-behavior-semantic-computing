# 六分布 TSP 数据集生成与 Concorde 求解

脚本沿用 `generate_custom_tsp.py` 与 `custom_TSP200_seed2026.pkl` 的数据格式。每个 PKL 只有四个顶层字段，且每个字段都是列表：`coordinate`、`optimal_tour`、`distance_matrix`、`cost`。坐标和欧氏距离矩阵均为 `float64`；城市坐标在 `[0,1]²`。原始文件中的路径和成本均为 `None`；求解结果写入独立的 `_solved.pkl`。

生成五种规模 × 六种分布，共 30 个文件：

| 规模 | 每种分布的实例数 | 六种分布合计 |
|---|---:|---:|
| TSP20 | 1000 | 6000 |
| TSP50 | 1000 | 6000 |
| TSP100 | 1000 | 6000 |
| TSP200 | 500 | 3000 |
| TSP500 | 100（可选 200） | 600（可选 1200） |

分布名分别是 `Uniform`、`Clustered`、`GaussianMixture`、`Grid`、`Circle`、`Urban`。其中 `Clustered`、`Grid`、`Circle` 延续第一版脚本的簇、抖动网格、多圆环生成规则；`Urban` 由密集中心城区和较稀疏的街区网格组成。文件示例：`TSP200_GaussianMixture_N0500_seed2026.pkl`。

## 放置文件

将 `tsp_dataset_pipeline.py` 和 `run_tsp_via_wsl.bat` 放入：

```text
D:\spso-behavior-semantic-computing\eoh\examples\tsp_gls_numba\TrainingData
```

Concorde 使用当前文件：

```text
D:\tsp求解器\concorde
```

在 WSL Ubuntu 中执行下面命令。需要 WSL 中的 Python 3 和 NumPy，使用此前已能读取 PKL 的 `python` 环境。

```bash
cd /mnt/d/spso-behavior-semantic-computing/eoh/examples/tsp_gls_numba/TrainingData
python -c 'import numpy; print(numpy.__version__)'
```

## 先运行小样本

测试文件放入独立目录，避免混入正式数据：

```bash
python tsp_dataset_pipeline.py generate --root ./GeneratedTSP_test --sizes 20 --distributions Uniform --count 3
python tsp_dataset_pipeline.py verify --root ./GeneratedTSP_test --sizes 20 --distributions Uniform
python tsp_dataset_pipeline.py solve --root ./GeneratedTSP_test --sizes 20 --distributions Uniform --max-total 1 --stop-on-error
python tsp_dataset_pipeline.py verify --root ./GeneratedTSP_test --sizes 20 --distributions Uniform
```

`solve` 应显示 `OK`，并生成 `_progress.sqlite3` 和 `_solved.pkl`。`verify` 会检查路径排列，以及 `cost` 是否等于原始浮点距离矩阵上的闭环长度。Concorde 严格证明的是放大取整后的整数距离问题；原始浮点 `cost` 是同一路径的重新计算值。

## 生成并求解全部正式数据

```bash
python tsp_dataset_pipeline.py generate
python tsp_dataset_pipeline.py status
python tsp_dataset_pipeline.py solve --max-total 1 --stop-on-error
python tsp_dataset_pipeline.py solve
python tsp_dataset_pipeline.py verify
```

默认将文件写到 `TrainingData/GeneratedTSP/`。TSP500 若需要每种分布 200 例，在**第一次**生成时改用：

```bash
python tsp_dataset_pipeline.py generate --tsp500-count 200
```

请预留数 GB 空间，完整求解包含 21,600 个实例，TSP500 的精确求解可能耗时较长。脚本每成功一例就提交 SQLite 进度，每 25 例及每个文件结束时更新 `_solved.pkl`。中断后重新执行 `solve` 会跳过已完成的实例。`status` 用于查看进度。

每例默认最长运行 3600 秒，可用 `--timeout 7200` 调整。`--max-total` 限制本次新尝试的实例数，适合逐批运行。默认首次失败即停止，失败记录保存在 `_progress.sqlite3`；确认要跳过失败实例继续时可传 `--continue-on-error`。即使 Concorde 返回码为 255，脚本仍会要求日志存在 `Optimal Solution`、`.sol` 路径有效且整数成本相符，满足后才写入标签。

Windows 命令提示符也可通过批处理调用 WSL：

```bat
run_tsp_via_wsl.bat status
run_tsp_via_wsl.bat solve --max-total 1 --stop-on-error
```

脚本不会修改 `custom_TSP200_seed2026.pkl` 或其已求解文件。原始数据也不被覆盖。`generate` 遇到同名文件默认跳过；`--force` 会替换原始 PKL，但若已经存在对应的求解进度，须先另外备份并处理进度数据库。
