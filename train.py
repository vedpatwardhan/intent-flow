import os
import yaml
import argparse
from trainers.stage1_pretrain import train_stage1
from trainers.stage2_sft import train_stage2
from trainers.stage3_rl import train_stage3


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def update_config(config, opts):
    if not opts:
        return
    for i in range(0, len(opts), 2):
        if i + 1 >= len(opts):
            print(f"Warning: Option '{opts[i]}' is missing a value. Skipping.")
            break
        key = opts[i].lstrip("-")
        val = opts[i + 1]

        # Convert values to correct type
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        else:
            try:
                if "." in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass

        # Traversal for nested dictionary update (e.g., stage1.lr)
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
        print(f"[Config Override] Set {key} = {val}")


def main():
    parser = argparse.ArgumentParser(
        description="LatentFlow: Modular Humanoid Control Training Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Select training stage: 1 (Dynamics Pre-training), 2 (SFT Action Grounding), 3 (RL Alignment)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to global YAML configuration file",
    )
    parser.add_argument(
        "--use_subset",
        action="store_true",
        help="Limit pre-training dataset to first 5 files for dry-run validation",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify config options using list of key-value pairs (e.g., stage1.epochs 50 model.latent_dim 256)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command line overrides
    update_config(config, args.opts)

    # Resolve directories
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)

    if args.stage == 1:
        train_stage1(config, use_subset=args.use_subset)
    elif args.stage == 2:
        train_stage2(config)
    elif args.stage == 3:
        train_stage3(config)


if __name__ == "__main__":
    main()
