
import argparse
import pickle
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

# ========== 路径配置 ==========
DATA_DIR = Path(
    "/mnt/d/spso-behavior-semantic-computing/eoh/"
    "examples/tsp_gls_numba/TrainingData"
)

INPUT = DATA_DIR / "custom_TSP200_seed2026.pkl"
OUTPUT = DATA_DIR / "custom_TSP200_seed2026_solved.pkl"

CONCORDE = "/mnt/d/tsp求解器/concorde"
SCALE = 1_000_000
TIMEOUT = 3600


def save_data(data):
    """安全保存，不覆盖原始数据集。"""
    temp = OUTPUT.with_suffix(".pkl.tmp")
    with open(temp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(OUTPUT)


def make_weights(dist):
    """将浮点距离矩阵转换为整数矩阵。"""
    if dist.shape != (200, 200):
        raise ValueError(f"矩阵形状错误: {dist.shape}")

    if not np.isfinite(dist).all():
        raise ValueError("距离矩阵存在非有限数值")

    if np.any(dist < 0):
        raise ValueError("距离矩阵存在负数")

    if not np.allclose(dist, dist.T, rtol=0, atol=1e-10):
        raise ValueError("距离矩阵不对称")

    weights = np.floor(dist * SCALE + 0.5).astype(np.int64)
    np.fill_diagonal(weights, 0)
    return weights


def write_tsp(weights, path):
    n = weights.shape[0]

    with open(path, "w") as f:
        f.write("NAME: TSP200\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")

        for row in weights:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("EOF\n")


def solve_instance(dist, index):
    """求解单个实例并验证最优路径。"""
    weights = make_weights(dist)

    with tempfile.TemporaryDirectory(prefix=f"tsp_{index:03d}_") as tmp:
        folder = Path(tmp)

        tsp_file = folder / "instance.tsp"
        sol_file = folder / "optimal.sol"

        write_tsp(weights, tsp_file)

        result = subprocess.run(
            [
                CONCORDE,
                "-o", str(sol_file),
                str(tsp_file),
            ],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

        log = result.stdout + "\n" + result.stderr

        # 必须有明确的最优性证明
        match = re.search(
            r"Optimal Solution:\s*([0-9]+(?:\.[0-9]+)?)",
            log,
        )

        if not match:
            raise RuntimeError(
                f"未证明最优，退出码 {result.returncode}\n"
                + log[-1200:]
            )

        if not sol_file.exists():
            raise RuntimeError(
                f"已报告最优，但没有生成路径文件，"
                f"退出码 {result.returncode}\n"
                + log[-1200:]
            )

        # 解析 Concorde 路径
        numbers = [
            int(x) for x in sol_file.read_text().split()
        ]

        n = len(dist)
        if len(numbers) != n + 1 or numbers[0] != n:
            raise ValueError("路径文件长度不正确")

        tour = np.asarray(numbers[1:], dtype=np.int64)

        # 验证是否包含全部城市且没有重复
        if sorted(tour.tolist()) != list(range(n)):
            raise ValueError("Concorde 返回无效路径")

        next_tour = np.roll(tour, -1)

        # 验证整数路径长度
        int_cost = int(weights[tour, next_tour].sum())
        reported = float(match.group(1))

        if abs(int_cost - reported) > 0.01:
            raise ValueError(
                f"路径长度不匹配: {int_cost} != {reported}"
            )

        # 使用原始浮点距离计算真实路径长度
        real_cost = float(dist[tour, next_tour].sum())

        return tour.tolist(), real_cost, int_cost, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多尝试多少个新实例，0=全部",
    )
    args = parser.parse_args()

    if OUTPUT.exists():
        print("继续之前的求解进度...")
        with open(OUTPUT, "rb") as f:
            data = pickle.load(f)
    else:
        print("读取原始数据集...")
        with open(INPUT, "rb") as f:
            data = pickle.load(f)

    total = len(data["distance_matrix"])
    attempted = 0
    succeeded = 0
    failed = []

    print(f"共有 {total} 个 TSP200 实例", flush=True)

    for i in range(total):
        # 跳过已经求解的实例
        if (
            data["optimal_tour"][i] is not None
            and data["cost"][i] is not None
        ):
            continue

        # 注意：限制的是尝试次数，而不是成功次数
        if args.limit > 0 and attempted >= args.limit:
            break

        attempted += 1
        print(f"\n[{i + 1}/{total}] 正在求解...", flush=True)

        start = time.perf_counter()

        try:
            dist = np.asarray(
                data["distance_matrix"][i],
                dtype=np.float64,
            )

            tour, cost, int_cost, exit_code = solve_instance(dist, i)

            data["optimal_tour"][i] = tour
            data["cost"][i] = cost

            save_data(data)
            succeeded += 1

            print("求解成功")
            print(f"整数最优值：{int_cost}")
            print(f"原始浮点路径长度：{cost:.10f}")
            print(f"Concorde 退出码：{exit_code}")
            print(f"耗时：{time.perf_counter() - start:.2f} 秒")
            print("结果已保存", flush=True)

        except Exception as e:
            failed.append(i)
            print(f"求解失败：{e}", flush=True)

    completed = sum(
        t is not None and c is not None
        for t, c in zip(
            data["optimal_tour"],
            data["cost"],
        )
    )

    print("\n========== 求解结束 ==========")
    print(f"本次尝试：{attempted}")
    print(f"本次成功：{succeeded}")
    print(f"累计完成：{completed}/{total}")
    print(f"失败实例编号：{failed}")
    print(f"输出文件：{OUTPUT}")


if __name__ == "__main__":
    main()
