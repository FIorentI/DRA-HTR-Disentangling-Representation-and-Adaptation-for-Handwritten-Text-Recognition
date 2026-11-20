from __future__ import print_function
import torch
import wandb
import sys
import os

# Get the path to the project root (3 levels up from this script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.args_demo.args_da import get_args_da as get_args
from src.train.domain_adaptation.trainer import Trainer


def resolve_resume_checkpoint_file(checkpoint_path):
    if os.path.isdir(checkpoint_path):
        return os.path.join(checkpoint_path, "training_last.pth")
    return checkpoint_path


def resolve_wandb_run_id(opt):
    if opt.wandb_run_id:
        return opt.wandb_run_id

    environment_run_id = os.environ.get("WANDB_RUN_ID")
    if environment_run_id:
        return environment_run_id

    checkpoint_file = resolve_resume_checkpoint_file(opt.resume_checkpoint)
    checkpoint_directory = os.path.dirname(os.path.abspath(checkpoint_file))
    wandb_id_file = os.path.join(checkpoint_directory, "wandb_run_id.txt")
    if os.path.isfile(wandb_id_file):
        with open(wandb_id_file, "r", encoding="utf-8") as file_handle:
            run_id = file_handle.read().strip()
        if run_id:
            return run_id

    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    run_id = checkpoint.get("wandb_run_id")
    del checkpoint
    if run_id:
        return run_id

    raise ValueError(
        "The resume checkpoint predates automatic W&B run-ID storage. "
        "Resume it once with --wandb_run_id <existing-run-id>; subsequent "
        "checkpoints will store the ID automatically."
    )


if __name__ == '__main__':
    opt = get_args()

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    opt.device = device

    if opt.debug_pc == 0:
        # Initialize wandb
        wandb_init_args = {
            "project": opt.project_wandb,
            "config": vars(opt),
        }
        if opt.wandb_name is not None:
            wandb_init_args["name"] = opt.wandb_name
        if opt.resume_checkpoint is not None:
            wandb_init_args["id"] = resolve_wandb_run_id(opt)
            wandb_init_args["resume"] = "must"
        wandb.init(**wandb_init_args)

        print("run name wand : " + str(wandb.run.name))
        print("run ID wand : " + str(wandb.run.id))

    trainer = Trainer(opt)
    trainer.train()

    if opt.debug_pc == 0:
        wandb.finish()
