from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install a MA-EoH best heuristic into the original EoH evaluation folder."
    )
    parser.add_argument("run_dir", help="Path to results_mae/<run_name>")
    args = parser.parse_args()

    source = Path(args.run_dir).resolve() / "best" / "best_heuristic.py"
    target = Path(__file__).resolve().parent / "evaluation" / "heuristic.py"
    if not source.is_file():
        raise FileNotFoundError(f"best heuristic not found: {source}")
    shutil.copy2(source, target)
    print(f"Installed {source} -> {target}")


if __name__ == "__main__":
    main()
