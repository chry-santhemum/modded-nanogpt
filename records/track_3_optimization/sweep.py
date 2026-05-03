"""
Launch a serial MuonFOOF hparam sweep.

Each dict in grid_blocks is one Cartesian-product grid. Missing keys, or values
set to None, are omitted from the train_gpt_simple.py command.
"""

import itertools
import subprocess
from datetime import datetime
from pathlib import Path


sweep_name = datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
nproc_per_node = 8
log_dir = "logs"

grid_blocks = [
    {},
    # {"foof_lr": [0.01, 0.02, 0.04]},
    # {"foof_alpha_mult": [0.5, 1.0, 2.0]},
    # {"foof_lr": [0.01, 0.02], "foof_alpha_mult": [1.0, 2.0]},
]


def main():
    repo_root = Path(__file__).resolve().parents[2]
    combos = []
    for grid in grid_blocks:
        keys = list(grid)
        combos.extend(
            dict(zip(keys, values))
            for values in itertools.product(*(grid[key] for key in keys))
        )
    print(f"sweep_name={sweep_name}")
    print(f"num_runs={len(combos)}")
    for index, combo in enumerate(combos, start=1):
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
            "records/track_3_optimization/train_gpt_simple.py",
            "--log-dir", log_dir,
            "--sweep-name", sweep_name,
        ]
        for name, value in combo.items():
            if value is None:
                continue
            flag = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                cmd.append(flag if value else "--no-" + name.replace("_", "-"))
            else:
                cmd.extend([flag, str(value)])
        print(f"run {index}/{len(combos)}: {combo}", flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
