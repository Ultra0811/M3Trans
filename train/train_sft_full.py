import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch full-parameter SFT with LLaMA-Factory.")
    parser.add_argument("--config", default="configs/sft_full_qwen25vl_7b.yaml")
    parser.add_argument("--launcher", default="llamafactory-cli")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = Path(args.config)
    if not config.exists():
        raise SystemExit(f"config not found: {config}")
    command = [args.launcher, "train", str(config)]
    print(" ".join(command))
    if args.dry_run:
        return
    if shutil.which(args.launcher) is None:
        raise SystemExit(f"launcher not found in PATH: {args.launcher}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
