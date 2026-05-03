"""
Launch a serial MuonFOOF hparam sweep.

Each dict in grid_blocks is one Cartesian-product grid. Missing keys, or values
set to None, are omitted from the train_gpt_simple.py command.
"""

import itertools
import json
import subprocess
from datetime import datetime
from pathlib import Path


nproc_per_node = 8
log_dir = "logs"
default_hparams = {
    "train_steps": 3250,
    "cooldown_frac": 0.7,
    "fw_alpha_cooldown_frac": 0.0,
    "fw_alpha_final_val": 1.0,
    "lr": 0.02,
    "beta_1": 0.95,
    "nesterov": True,
    "fw_alpha_method": "mean_iso",
    "fw_alpha_mult": 10.0,
    "fw_steps": 3,
    "fw_gamma_method": "line_search",
    "weight_decay": 0.02,
    "mbs": 64,
    "val_mbs": 64,
}

grid_blocks = [
    # default: step 3375, loss 3.28350
    # {"fw_alpha_method": ["first_fw"], "fw_alpha_mult": [1.0]},  # logs/20260503_033646, step 3375, loss 3.28643
    # {"cooldown_frac": [0.5]},  # logs/20260503_035759_598139, step 3375, loss 3.27764
    # {"beta_1": [0.9, 0.99]},   # logs/20260503_041609_179584, step 3375, loss [3.28809, 3.30654]
    {"fw_alpha_mult": [5.0], "fw_alpha_cooldown_frac": [0.5], "cooldown_frac": [0.5]}
    # {"lr": [0.01, 0.03]},
]


def main():
    repo_root = Path(__file__).resolve().parents[2]
    resolved_log_dir = repo_root / log_dir
    block_runs = []
    for grid in grid_blocks:
        keys = list(grid)
        combos = list(
            dict(zip(keys, values))
            for values in itertools.product(*(grid[key] for key in keys))
        )
        block_runs.append((grid, combos))
    print(f"num_blocks={len(block_runs)}")
    print(f"num_runs={sum(len(combos) for _, combos in block_runs)}")
    for grid, combos in block_runs:
        block_sweep_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        block_dir = resolved_log_dir / block_sweep_name
        block_dir.mkdir(parents=True, exist_ok=True)
        with (block_dir / "grid_block.json").open("w") as f:
            runs = []
            for combo in combos:
                hparams = default_hparams | {
                    key: value for key, value in combo.items() if value is not None
                }
                hparams["val_mbs"] = hparams["mbs"] if hparams["val_mbs"] is None else hparams["val_mbs"]
                runs.append(hparams)
            json.dump({"grid_block": grid, "runs": runs}, f, indent=2)
            f.write("\n")
        print(f"{block_sweep_name}: {len(combos)} runs", flush=True)
        for index, combo in enumerate(combos, start=1):
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={nproc_per_node}",
                "records/track_3_optimization/train_gpt_simple.py",
                "--log-dir", str(resolved_log_dir),
                "--sweep-name", block_sweep_name,
            ]
            for name, value in combo.items():
                if value is None:
                    continue
                flag = "--" + name.replace("_", "-")
                if isinstance(value, bool):
                    cmd.append(flag if value else "--no-" + name.replace("_", "-"))
                else:
                    cmd.extend([flag, str(value)])
            print(f"{block_sweep_name} run {index}/{len(combos)}: {combo}", flush=True)
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
