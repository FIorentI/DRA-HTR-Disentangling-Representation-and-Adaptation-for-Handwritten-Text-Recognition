import argparse
from html import parser
import os
import random


def check_dirs(dirs):
    dirs = [dirs] if type(dirs) not in [list, tuple] else dirs
    for d in dirs:
        try:
            os.makedirs(d)
        except OSError:
            pass
    return


def get_args_da():
    parser = argparse.ArgumentParser()

    # Mandotory
    # parser.add_argument("config_file")

    ## Common Parameters ##
    parser.add_argument('-T', '--task', default='clf', help='clf | seg')  # Classification or Segmentation
    # parser.add_argument('-D','--datasets', type=str, nargs='+', required=True, help='clf: M/MM/U (MNIST/MNIST-M/USPS) '
    #                                                                            'seg: G/C (GTA5/Cityscapes)')
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--dataset_dir_subdir", default="", type=str)
    parser.add_argument("--dataset_source", required=True, type=str)
    parser.add_argument("--dataset_target", required=True, type=str)
    parser.add_argument("--charset", required=True, default="", type=str, help="Name of the charset.txt file.")
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--imsize', type=int, help='the height of the input image')
    parser.add_argument('--iter', type=int, default=10000000, help='total training iterations')
    parser.add_argument('--manualSeed', type=int, default=None)
    parser.add_argument('--ex', help='Experiment name', default="not_set")
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--tensor_freq', type=int, help='frequency of showing results on tensorboard during training.')
    parser.add_argument('--eval_freq', type=int, help='frequency of evaluation during training.')
    parser.add_argument('--CADT', type=bool, default=False)
    parser.add_argument('--load_step', type=int, help="iteration of trained networks")
    parser.add_argument('--debug_images', type=bool, default=False)
    parser.add_argument('--invert_images', type=int, default=1, choices=[0, 1], help="If 1, invert grayscale inputs to white ink on a black background. If 0, preserve the original polarity and use white padding.")
    parser.add_argument('--generator_output_activation', type=str, default="tanh", choices=["tanh", "tanh01", "sigmoid"], help="Generator output activation. tanh (default) preserves the reference training behavior in [-1, 1]; tanh01 remaps tanh outputs to [0, 1].")
    parser.add_argument('--generator_vertical_refine_steps', type=int, default=0, help="Number of learned vertical x2 refinement blocks after the generator image head.")
    parser.add_argument('--generator_vertical_refine_channels', type=int, default=32, help="Hidden channels used by each vertical generator refinement block.")
    parser.add_argument('--eval_GANimg', type=bool, default=False)
    parser.add_argument('--T_both_directions', type=int, default=1, choices=[0, 1], help="If 1, train T in both directions (source->target and target->source). If 0, train T only in the source->target direction.")
    parser.add_argument('--T_training_start', type=int, default=0, help="Iteration to start training T.")
    parser.add_argument('--T_loss_combo', type=str, default="real_and_fake_loss", choices=["real_and_fake_loss", "fake_loss"], help="Loss combination for T. 'real_and_fake_loss' (default) uses both the real and fake task losses. 'fake_loss' uses only the fake task loss.")

    parser.add_argument('--latent_content_loss', type=str, default="L1", choices=["L1", "cosine"], help="Loss type for converted content consistency")
    parser.add_argument('--lambda_latent_content', type=float, default=0.0)
    parser.add_argument('--lambda_latent_style', type=float, default=0.0) 

    parser.add_argument('--lambda_task', type=float, default=0.0, help="make the value higher than 0 to enable the task loss")
    parser.add_argument('--T_ESG_loss_start', type=int, default=0, help="Iteration to start Task loss for ESG.")
    parser.add_argument("--task_deepcopy", default="no", type=str, choices=["no", "yes"])

    parser.add_argument("--discriminator", default="MNIST", type=str, choices=["MNIST", "PatchGAN"])
    parser.add_argument("--slice_patches", default=1, type=int, choices=[0, 1])
    parser.add_argument("--dis_last_layer_stride", default=2, type=int, choices=[1, 2])

    parser.add_argument("--dra_arch", default="legacy", type=str, choices=["legacy", "style_film"])

    parser.add_argument("--P_net", default="VGG19", type=str, choices=["VGG19", "convnextv2_large", "convnextv2_base", "convnextv2_tiny"])
    parser.add_argument("--E_net", default="CRNN", type=str, choices=["CRNN", "convnextv2_large", "convnextv2_tiny"])
    parser.add_argument("--convnextv2_model_path", default="", type=str)
    parser.add_argument("--E_net_swap_epoch", default=0, type=int, help="Epoch to swap the E_net from CRNN to convnextv2. If 0, no swap is performed.")
    parser.add_argument("--perceptual_convnextv2_model_path", default="", type=str)

    parser.add_argument("--dynamic_loss_weighting", type=int, default=0,  help="If 1, dynamically rescales ESG lambdas so each weighted loss keeps the same relative contribution observed on the first ESG training batch.")
    parser.add_argument("--dynamic_loss_weighting_eps", type=float, default=1e-8, help="Small value used to avoid division by zero when dynamic ESG loss weighting is enabled.")
    parser.add_argument("--dynamic_loss_weighting_max_scale", type=float, default=10.0,  help="Maximum multiplicative change allowed for a dynamic ESG lambda relative to its initial value. Use <=0 to disable clamping.")

    parser.add_argument("--load_path", default=None, type=str, help="Path to a saved model checkpoints.")
    parser.add_argument("--checkpoint_stage", default="last", type=str, choices=["last", "best"], help="Stage of the checkpoint to load.")
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        type=str,
        help=(
            "Path to training_last.pth (or its containing directory) for a full "
            "training resume, including optimizers and the global step."
        ),
    )
    parser.add_argument(
        "--wandb_run_id",
        default=None,
        type=str,
        help=(
            "W&B run ID fallback for resuming a checkpoint created before the "
            "run ID was stored automatically. This is not the visible run name."
        ),
    )

    parser.add_argument('--debug_pc', default=1, type=int)  # set to 0 to activate wandb log

    ## Optimizers Parameters ##
    parser.add_argument('--lr_dra', type=float, default=0.0002)
    parser.add_argument('--lr_clf', type=float, default=4e-4)
    parser.add_argument('--lr_seg', type=float, default=6e-4)
    parser.add_argument('--lambda_g', type=float, default=0.12)
    parser.add_argument('--lambda_content', type=float, default=1.0)
    parser.add_argument('--lambda_style', type=float, default=0.5)
    parser.add_argument('--lambda_consis', type=float, default=2.0)
    parser.add_argument('--lambda_recon', type=float, default=0.2)
    parser.add_argument('--recon_foreground_weight', type=float, default=0.0)
    parser.add_argument('--recon_foreground_mode', type=str, default="auto", choices=["auto", "dark", "bright"])
    parser.add_argument('--recon_edge_weight', type=float, default=0.0)
    parser.add_argument('--clip_grad_dra', type=float, default=0.0, help="Maximum gradient norm for E/S/G and D. Use 0 to disable clipping; non-finite gradients are always rejected.")
    parser.add_argument('--clip_grad_task', type=float, default=0.0, help="Maximum gradient norm for task networks. Use 0 to disable clipping; non-finite gradients are always rejected.")
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_step', type=int, default=20000)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--weight_decay_dra', type=float, default=1e-5)
    parser.add_argument('--weight_decay_task', type=float, default=5e-4)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--esg_updates_per_step', type=int, default=1, help="Number of encoder/separator/generator updates per discriminator update.")
    parser.add_argument("--project_wandb", default="DRANet_READ", type=str)
    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--path_log", default='checkpoint/', type=str)

    # CRNN
    parser.add_argument("--path_model", default="", type=str)
    parser.add_argument("--path_optimizer", default="", help="", type=str)
    parser.add_argument('--height_max', default=160, type=int)
    parser.add_argument('--width_max', default=1570, type=int)
    parser.add_argument('--pad_left', default=64, type=int)
    parser.add_argument('--pad_right', default=64, type=int)
    parser.add_argument('--apply_blur', default=0, type=int)
    parser.add_argument("--padding_strategy", default="center", type=str, choices=["center", "random", "top_left"])
    parser.add_argument("--model_type", default="crnn", type=str, choices=["crnn", "cnn"])
    parser.add_argument("--weights_to_load", default="", help="", type=str)
    parser.add_argument("--freeze_shortcut", default="No", type=str)
    parser.add_argument("--freeze_encoder", default="No", type=str)

    args = parser.parse_args()
    check_dirs([args.path_log + args.task + '/' + args.ex])
    args.logfile = args.path_log  + args.task + '/' + args.ex + '/' + args.ex + '.log'
    if args.task == 'seg':
        args.CADT = True
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    if args.batch is None:
        if args.task == 'clf':
            args.batch = 8
        elif args.task == 'seg':
            args.batch = 2
    if args.imsize is None:
        if args.task == 'clf':
            # args.imsize = (160, 1698)
            args.imsize = (args.height_max, args.width_max)
        elif args.task == 'seg':
            args.imsize = 512
    if args.tensor_freq is None:
        if args.task == 'clf':
            args.tensor_freq = 1000
        elif args.task == 'seg':
            args.tensor_freq = 100
    if args.eval_freq is None:
        if args.task == 'clf':
            args.eval_freq = 5000
        elif args.task == 'seg':
            args.eval_freq = 1000

    # To refactor
    args.datasets = [args.dataset_source, args.dataset_target]

    return args
