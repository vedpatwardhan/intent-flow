import os
import hydra
from omegaconf import OmegaConf
from trainers.stage1_pretrain import train_stage1
from trainers.stage2_sft import train_stage2
from trainers.stage3_rl import train_stage3


@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(cfg):
    # Resolve all configurations dynamically using OmegaConf
    config = OmegaConf.to_container(cfg, resolve=True)

    # Resolve directories
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)

    stage = config.get("stage")
    use_subset = config.get("use_subset", False)

    if stage is None:
        raise ValueError("Please specify the training stage (e.g., stage=1)")

    if stage == 1:
        train_stage1(config, use_subset=use_subset)
    elif stage == 2:
        train_stage2(config)
    elif stage == 3:
        train_stage3(config)


if __name__ == "__main__":
    main()
