"""Run the approved 2--6 box showcase configurations sequentially."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "final_showcase"
OUTPUT_DIR = ROOT / "outputs" / "final_showcase_real_20260817"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    counts = [int(value) for value in sys.argv[1:]] or list(range(2, 7))
    for count in counts:
        if count not in range(2, 7):
            raise ValueError(f"unsupported box count: {count}")
        name = f"case_{count}_boxes"
        destination = OUTPUT_DIR / name
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            str(ROOT / "run_experiment.py"),
            "--config",
            str(CONFIG_DIR / f"{name}.yaml"),
            "--output",
            str(destination),
        ]
        print(f"SHOWCASE_START {name}", flush=True)
        with (destination / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(f"SHOWCASE_END {name} exit={result.returncode}", flush=True)
        if result.returncode:
            failures.append(name)
    if failures:
        print("SHOWCASE_FAILURES " + ",".join(failures), flush=True)
        return 1
    print("SHOWCASE_ALL_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
