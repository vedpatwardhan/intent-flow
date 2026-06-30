import os
import yaml
import argparse
from trainers.stage1_pretrain import train_stage1
from trainers.stage2_sft import train_stage2
from trainers.stage3_rl import train_stage3


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

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
