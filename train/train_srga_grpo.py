import argparse
import os
import shlex
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Launch SRGA/GRPO alignment training.")
    parser.add_argument("--config", default="configs/srga_grpo.yaml")
    parser.add_argument("--backend_cmd", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = Path(args.config)
    if not config.exists():
        raise SystemExit(f"config not found: {config}")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))

    backend = args.backend_cmd or os.environ.get("SRGA_GRPO_CMD")
    if not backend:
        print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        message = "Set --backend_cmd or SRGA_GRPO_CMD to the verl/GRPO launcher used in your environment."
        if args.dry_run:
            print(message)
            return
        raise SystemExit(message)

    command = shlex.split(backend) + ["--config", str(config)]
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
