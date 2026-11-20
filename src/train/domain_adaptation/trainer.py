from __future__ import print_function

import json
import math
import os
import random
import sys
import copy
from logging import Formatter, StreamHandler, getLogger, FileHandler
from random import seed
from typing import Dict, Tuple

import editdistance
import numpy as np
import torch.backends.cudnn
import wandb
from prettytable import PrettyTable

from src.models.domain_adaptation.loss_functions_READ import *
from src.models.domain_adaptation.model_READ import *
from src.data.collate.collate_batch import CollateImageLabelHTR
from src.data.datasets.htr_dataset_lmdb import HTRDataset
from src.data.global_values.text_global_values import BLANK_STR_TOKEN, CTC_PAD
from src.data.image.augmentations.img_augmentation import get_img_data_augmentation
from src.data.readers.reader_config import read_json_config, read_labels_files
from src.data.text.best_path_ctc import ctc_best_path_one
from src.data.text.charset_token import CharsetToken
from src.evaluate.evaluate_recognition import nb_chars_from_list
from src.evaluate.metrics_counter import MetricLossCERWER
from src.models.crnn import CRNN
from src.models.model_utils import load_pretrained_model, load_pretrained_model_ConvNext
from src.train.domain_adaptation.utils_da import *
from src.models.domain_adaptation.DRANet_READ import *
from src.models.domain_adaptation.convnextv2 import convnextv2_large, convnextv2_base, convnextv2_tiny, DRANetConvNeXtEncoder

from src.models.domain_adaptation.cadt import *
from src.models.domain_adaptation.drn import drn26
import torchvision.utils as vutils



def set_converts(datasets, task):
    training_converts, test_converts = [], []
    center_dset = datasets[0]
    for source in datasets:  # source
        if not center_dset == source:
            training_converts.append(center_dset + '_2_' + source)
            training_converts.append(source + '_2_' + center_dset)
        if task == 'clf':
            for target in datasets:  # target
                if not source == target:
                    test_converts.append(source + '_2_' + target)
    if task == 'clf':
        tensorboard_converts = test_converts
    elif task == 'seg':
        test_converts.append('G2C')
        tensorboard_converts = training_converts
    else:
        raise Exeception("Does not support the task")

    return training_converts, test_converts, tensorboard_converts


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.training_converts, self.test_converts, self.tensorboard_converts = set_converts(args.datasets, args.task)
        if args.task == 'seg':
            self.imsize = (2 * args.imsize, args.imsize)  # (width, height)
            self.best_miou = 0.
        elif args.task == 'clf':
            self.imsize = args.imsize
            self.best_cer = {cv: float('inf') for cv in self.test_converts}

            self.best_cer_shortcut = {cv: float('inf') for cv in self.test_converts}
            
        self.dynamic_loss_weighting = int(getattr(args, "dynamic_loss_weighting", 0)) == 1
        self.dynamic_loss_eps = float(getattr(args, "dynamic_loss_weighting_eps", 1e-8))
        self.dynamic_loss_max_scale = float(getattr(args, "dynamic_loss_weighting_max_scale", 10.0))
        self.dynamic_loss_reference_ratios: Dict[str, float] | None = None
        self.dynamic_loss_reference_step: int | None = None

        # Paths
        # dbs_info, charsets_path, all_labels_files, _ = read_json_config(args.config_file)
        # all_labels = read_labels_files(all_labels_files) # Uncomment if you use the htr_dataset.py
        all_labels = None   # Comment if you use the htr_dataset.py

        # Alphabet
        charsets_path = os.path.join(self.args.dataset_root, self.args.datasets[0], self.args.dataset_dir_subdir, self.args.charset)
        print(f"Charset path: {charsets_path}")
        self.charset = CharsetToken([charsets_path], use_blank=True)
        self.char_list = self.charset.get_charset_list()
        self.char_dict = self.charset.get_charset_dictionary()

        # Data
        fixed_size_img = (args.height_max, args.width_max)
        aug_transforms = get_img_data_augmentation()

        # Pad img with black = 0
        c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
        collate_fn = c_collate_fn.collate_fn

        width_divisor = 8  # Relative to the model

        # data loader
        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        for dset in self.args.datasets:
            print('Dataset: ' + dset)

            # info_one_db = dbs_info[dset]
            # train_info = [[dset, info_one_db["train"]["img_dir"], info_one_db["train"]["extension_img"]]]
            # Merge train db
            directory_train = os.path.join(self.args.dataset_root, dset, self.args.dataset_dir_subdir, "train")
            train_dataset = HTRDataset(directory_train,
                                       fixed_size_img,
                                       width_divisor,
                                       args.padding_strategy,
                                       args.pad_left,
                                       args.pad_right,
                                       all_labels,
                                       self.char_dict,
                                       None,
                                       apply_blur=args.apply_blur,
                                       apply_noise=0,
                                       is_trainset=True,
                                       invert_images=bool(args.invert_images))

            #test_info = [[dset + "_test", info_one_db["train"]["img_dir"], info_one_db["train"]["extension_img"]]]
            directory_test = os.path.join(self.args.dataset_root, dset, self.args.dataset_dir_subdir, "test")
            test_dataset = HTRDataset(directory_test,
                                      fixed_size_img,
                                      width_divisor,
                                      args.padding_strategy,
                                      args.pad_left,
                                      args.pad_right,
                                      all_labels,
                                      self.char_dict,
                                      None,
                                      apply_blur=args.apply_blur,  # No blur on test
                                      apply_noise=0,        # No noise on test
                                      is_trainset=False,
                                      invert_images=bool(args.invert_images))

            print("Train dataset size:", len(train_dataset))
            print("Test dataset size:", len(test_dataset))

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.args.batch,
                shuffle=True,
                drop_last=True,
                num_workers=int(self.args.workers),
                pin_memory=True,
                collate_fn=collate_fn)

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.args.batch * 4,
                shuffle=False,
                num_workers=int(self.args.workers),
                collate_fn=collate_fn)

            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader

        self.nets, self.optims, self.losses = dict(), dict(), dict()
        self.loss_fns = Loss_Functions(args)

        self.logger = getLogger()
        # self.checkpoint = './checkpoint/%s/%s' % (args.task, args.ex)
        self.checkpoint = os.path.join(args.path_log, "checkpoint")
        os.makedirs(self.checkpoint, exist_ok=True)

        self.step = 0
        self.accumulation_counts = {'D': 0, 'ESG': 0}
        self.task_accumulation_counts = {cv: 0 for cv in self.test_converts}

        self.name_classes_19 = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "trafflight", "traffsign", "vegetation",
            "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
        ]


    def _collect_dynamic_loss_values(self, loss_tensors: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {key: float(value.detach().float().item()) for key, value in loss_tensors.items()}

    def _dynamic_esg_loss_total(
        self,
        loss_tensors: Dict[str, torch.Tensor],
        base_lambdas: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        active_keys = [
            key
            for key in loss_tensors
            if abs(float(base_lambdas.get(key, 0.0))) > 0.0
        ]
        if not self.dynamic_loss_weighting or len(active_keys) <= 1:
            weights = {key: float(base_lambdas.get(key, 0.0)) for key in loss_tensors}
            total = sum(weights[key] * loss_tensors[key] for key in loss_tensors)
            return total, weights, {}

        current_values = self._collect_dynamic_loss_values({key: loss_tensors[key] for key in active_keys})
        current_contribs = {
            key: abs(float(base_lambdas[key]) * current_values[key])
            for key in active_keys
            if math.isfinite(current_values[key])
        }
        current_total = sum(current_contribs.values())

        if self.dynamic_loss_reference_ratios is None:
            if current_total <= self.dynamic_loss_eps:
                lambda_total = sum(abs(float(base_lambdas[key])) for key in active_keys)
                if lambda_total <= self.dynamic_loss_eps:
                    ratios = {key: 1.0 / len(active_keys) for key in active_keys}
                else:
                    ratios = {key: abs(float(base_lambdas[key])) / lambda_total for key in active_keys}
            else:
                ratios = {key: current_contribs.get(key, 0.0) / current_total for key in active_keys}
            self.dynamic_loss_reference_ratios = ratios
            self.dynamic_loss_reference_step = self.step
            if hasattr(self, "logger"):
                ratio_desc = ", ".join(f"{key}={value:.4f}" for key, value in sorted(ratios.items()))
                self.logger.info("Dynamic ESG loss reference set at step %d: %s" % (self.step, ratio_desc))

        ratios = self.dynamic_loss_reference_ratios
        weights = {key: float(base_lambdas.get(key, 0.0)) for key in loss_tensors}

        if current_total > self.dynamic_loss_eps:
            for key in active_keys:
                ratio = ratios.get(key)
                if ratio is None:
                    continue
                raw_abs = abs(current_values.get(key, 0.0))
                target_contrib = ratio * current_total
                weight = target_contrib / max(raw_abs, self.dynamic_loss_eps)
                max_scale = self.dynamic_loss_max_scale
                if max_scale > 0:
                    base = abs(float(base_lambdas[key]))
                    if base > 0:
                        weight = min(max(weight, base / max_scale), base * max_scale)
                weights[key] = weight

        total = sum(weights[key] * loss_tensors[key] for key in loss_tensors)
        dynamic_ratios = {
            key: abs(weights[key] * current_values.get(key, 0.0)) / max(current_total, self.dynamic_loss_eps)
            for key in active_keys
        }
        return total, weights, dynamic_ratios
    
    def set_default(self):
        if self.args.accum_steps < 1:
            raise ValueError("accum_steps must be at least 1")
        if self.args.clip_grad_dra < 0 or self.args.clip_grad_task < 0:
            raise ValueError("gradient clipping limits must be non-negative")

        # Enable cuDNN benchmark only if CUDA is available
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        ## Random Seed ##
        print("Random Seed: ", self.args.manualSeed)
        seed(self.args.manualSeed)
        torch.manual_seed(self.args.manualSeed)

        # Only set CUDA seeds if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.manualSeed)

        ## Logger ##
        file_log_handler = FileHandler(self.args.logfile)
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = StreamHandler(sys.stdout)
        self.logger.addHandler(stderr_log_handler)
        self.logger.setLevel('INFO')
        formatter = Formatter()
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

        if self.dynamic_loss_weighting:
            self.logger.info(
                "Dynamic ESG loss weighting: enabled eps=%g max_scale=%g"
                % (self.dynamic_loss_eps, self.dynamic_loss_max_scale)
            )

    def save_networks(self):
        if not os.path.exists(self.checkpoint + '/%d' % self.step):
            os.mkdir(self.checkpoint + '/%d' % self.step)
        for key in self.nets.keys():
            if key == 'D':
                for dset in self.args.datasets:
                    torch.save(self.nets[key][dset].state_dict(),
                               self.checkpoint + '/%d/net%s_%s.pth' % (self.step, key, dset))
            elif key == 'T':
                for cv in self.test_converts:
                    torch.save(self.nets[key][cv].state_dict(),
                               self.checkpoint + '/%d/net%s_%s.pth' % (self.step, key, cv))
            else:
                torch.save(self.nets[key].state_dict(), self.checkpoint + '/%d/net%s.pth' % (self.step, key))

    def save_networks_last(self):
        if not os.path.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        for key in self.nets.keys():
            if key == 'D':
                for dset in self.args.datasets:
                    torch.save(self.nets[key][dset].state_dict(),
                               self.checkpoint + '/net_last_%s_%s.pth' % (key, dset))
            elif key == 'T':
                for cv in self.test_converts:
                    torch.save(self.nets[key][cv].state_dict(),
                               self.checkpoint + '/net_last_%s_%s.pth' % (key, cv))
            else:
                torch.save(self.nets[key].state_dict(), self.checkpoint + '/net%s.pth' % (key))

    @staticmethod
    def _state_dict_tree(objects):
        return {
            key: (
                {subkey: subobject.state_dict() for subkey, subobject in value.items()}
                if isinstance(value, dict)
                else value.state_dict()
            )
            for key, value in objects.items()
        }

    @staticmethod
    def _gradient_state_tree(networks):
        def module_gradients(module):
            return {
                name: parameter.grad.detach().cpu()
                for name, parameter in module.named_parameters()
                if parameter.grad is not None
            }

        return {
            key: (
                {subkey: module_gradients(submodule) for subkey, submodule in value.items()}
                if isinstance(value, dict)
                else module_gradients(value)
            )
            for key, value in networks.items()
        }

    def _step_optimizers(self, optimizers, group_name, max_grad_norm=0.0):
        """Step one accumulated optimizer group without applying invalid gradients."""
        optimizers = list(optimizers)
        parameters = []
        seen_parameters = set()
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                for parameter in param_group['params']:
                    if parameter.grad is None or id(parameter) in seen_parameters:
                        continue
                    seen_parameters.add(id(parameter))
                    parameters.append(parameter)

        if parameters:
            effective_max_norm = max_grad_norm if max_grad_norm > 0 else float("inf")
            total_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=effective_max_norm,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(total_norm).item():
                self.logger.warning(
                    "Skipping %s optimizer update at step %d because its accumulated "
                    "gradient norm is NaN or Inf."
                    % (group_name, self.step)
                )
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                return False

        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return True

    @staticmethod
    def _load_state_dict_tree(objects, states, object_name):
        if set(objects) != set(states):
            raise ValueError(
                f"{object_name} keys differ: current={sorted(objects)}, "
                f"checkpoint={sorted(states)}"
            )
        for key, value in objects.items():
            if isinstance(value, dict):
                if set(value) != set(states[key]):
                    raise ValueError(
                        f"{object_name}['{key}'] keys differ: "
                        f"current={sorted(value)}, checkpoint={sorted(states[key])}"
                    )
                for subkey, subobject in value.items():
                    subobject.load_state_dict(states[key][subkey])
            else:
                value.load_state_dict(states[key])

    def _resume_metadata(self):
        return {
            "task": self.args.task,
            "datasets": list(self.args.datasets),
            "charset": self.args.charset,
            "model_type": self.args.model_type,
            "freeze_shortcut": self.args.freeze_shortcut,
            "E_net": self.args.E_net,
            "E_net_swap_epoch": self.args.E_net_swap_epoch,
            "P_net": self.args.P_net,
            "dra_arch": self.args.dra_arch,
            "discriminator": self.args.discriminator,
            "generator_output_activation": self.args.generator_output_activation,
            "generator_vertical_refine_steps": self.args.generator_vertical_refine_steps,
            "generator_vertical_refine_channels": self.args.generator_vertical_refine_channels,
            "height_max": self.args.height_max,
            "width_max": self.args.width_max,
            "pad_left": self.args.pad_left,
            "pad_right": self.args.pad_right,
            "accum_steps": self.args.accum_steps,
        }

    def save_training_checkpoint(self):
        checkpoint_file = os.path.join(self.checkpoint, "training_last.pth")
        temporary_file = checkpoint_file + ".tmp"
        wandb_run_id = wandb.run.id if wandb.run is not None else None
        state = {
            "version": 1,
            "step": self.step,
            "wandb_run_id": wandb_run_id,
            "networks": self._state_dict_tree(self.nets),
            "optimizers": self._state_dict_tree(self.optims),
            "gradients": self._gradient_state_tree(self.nets),
            "accumulation_counts": dict(self.accumulation_counts),
            "task_accumulation_counts": dict(self.task_accumulation_counts),
            "best_cer": dict(self.best_cer) if self.args.task == "clf" else None,
            "best_cer_shortcut": (
                dict(self.best_cer_shortcut) if self.args.task == "clf" else None
            ),
            "best_miou": getattr(self, "best_miou", None),
            "dynamic_loss_reference_ratios": self.dynamic_loss_reference_ratios,
            "dynamic_loss_reference_step": self.dynamic_loss_reference_step,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "metadata": self._resume_metadata(),
        }
        torch.save(state, temporary_file)
        os.replace(temporary_file, checkpoint_file)
        if wandb_run_id is not None:
            wandb_id_file = os.path.join(self.checkpoint, "wandb_run_id.txt")
            temporary_wandb_id_file = wandb_id_file + ".tmp"
            with open(temporary_wandb_id_file, "w", encoding="utf-8") as file_handle:
                file_handle.write(wandb_run_id + "\n")
            os.replace(temporary_wandb_id_file, wandb_id_file)
        self.logger.info(
            "Saved resumable training checkpoint at step %d: %s"
            % (self.step, checkpoint_file)
        )

    def load_training_checkpoint(self, checkpoint_path):
        checkpoint_file = checkpoint_path
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(checkpoint_file, "training_last.pth")
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(checkpoint_file)

        checkpoint = torch.load(
            checkpoint_file,
            map_location=self.device,
            weights_only=False,
        )
        if checkpoint.get("version") != 1:
            raise ValueError(
                f"Unsupported training checkpoint version: {checkpoint.get('version')}"
            )
        saved_wandb_run_id = checkpoint.get("wandb_run_id")
        active_wandb_run_id = wandb.run.id if wandb.run is not None else None
        if (
            saved_wandb_run_id is not None
            and active_wandb_run_id is not None
            and saved_wandb_run_id != active_wandb_run_id
        ):
            raise ValueError(
                "W&B run ID mismatch: "
                f"active={active_wandb_run_id!r}, checkpoint={saved_wandb_run_id!r}"
            )

        current_metadata = self._resume_metadata()
        print(current_metadata)
        for key, saved_value in checkpoint["metadata"].items():
            if current_metadata.get(key) != saved_value:
                raise ValueError(
                    f"Resume configuration mismatch for '{key}': "
                    f"current={current_metadata.get(key)!r}, checkpoint={saved_value!r}"
                )

        self._load_state_dict_tree(self.nets, checkpoint["networks"], "Network")
        self._load_state_dict_tree(self.optims, checkpoint["optimizers"], "Optimizer")

        # Deliberately start a fresh accumulation window on resume. The saved
        # model and optimizer history are kept, but pending gradients are not.
        for value in self.optims.values():
            optimizers = value.values() if isinstance(value, dict) else (value,)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
        self.accumulation_counts = {'D': 0, 'ESG': 0}
        self.task_accumulation_counts = {cv: 0 for cv in self.test_converts}

        self.step = int(checkpoint["step"])
        if self.args.task == "clf":
            self.best_cer = dict(checkpoint["best_cer"])
            self.best_cer_shortcut = dict(checkpoint["best_cer_shortcut"])
        elif checkpoint["best_miou"] is not None:
            self.best_miou = checkpoint["best_miou"]
        self.dynamic_loss_reference_ratios = checkpoint["dynamic_loss_reference_ratios"]
        self.dynamic_loss_reference_step = checkpoint["dynamic_loss_reference_step"]

        rng_state = checkpoint["rng_state"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and rng_state["cuda"] is not None:
            for device_index, cuda_state in enumerate(
                rng_state["cuda"][:torch.cuda.device_count()]
            ):
                torch.cuda.set_rng_state(cuda_state.cpu(), device_index)

        self.logger.info(
            "Resumed complete training state from step %d: %s"
            % (self.step, checkpoint_file)
        )
                
    def load_networks(self, checkpoint_stage="last"):
        device = self.device
        print(f"\nLoading checkpoints from: {self.args.load_path}\n")
        for key in self.nets.keys():
            if key == "D":
                for dset in self.args.datasets:
                    file = f"{self.args.load_path}/net_{checkpoint_stage}_{key}_{dset}.pth"
                    if not os.path.isfile(file):
                        raise FileNotFoundError(file)
                    msg = self.nets[key][dset].load_state_dict(torch.load(file, map_location=device, weights_only=True), strict=True)
                    print(f"[OK] {os.path.basename(file)} - {msg}")
            elif key == "T":
                if self.args.task == "clf":
                    for cv in self.test_converts:
                        file = f"{self.args.load_path}/net_{checkpoint_stage}_{key}_{cv}.pth"
                        if not os.path.isfile(file):
                            raise FileNotFoundError(file)
                        msg = self.nets[key][cv].load_state_dict(torch.load(file, map_location=device, weights_only=True), strict=True)
                        print(f"[OK] {os.path.basename(file)} - {msg}")
            else:
                file = f"{self.args.load_path}/net{key}.pth"
                if not os.path.isfile(file):
                    raise FileNotFoundError(file)
                msg = self.nets[key].load_state_dict(torch.load(file, map_location=device, weights_only=True), strict=True)
                print(f"[OK] {os.path.basename(file)} - {msg}")
        print("\nAll checkpoints loaded successfully.\n")


    def _create_encoder_optimizer(self):
        return optim.Adam(
            self.nets['E'].parameters(),
            lr=self.args.lr_dra,
            betas=(self.args.beta1, 0.999),
            weight_decay=self.args.weight_decay_dra,
        )

    def swap_E_net(self):
        if self.args.E_net == 'convnextv2_large':
            new_encoder = convnextv2_large(net="E", out_index=1)
            load_pretrained_model_ConvNext(self.args, new_encoder, self.device)
            print("E_Net: ConvNeXt-V2_large + weights")
        elif self.args.E_net == 'convnextv2_tiny':
            new_encoder = convnextv2_tiny(net="E", out_index=1)
            load_pretrained_model_ConvNext(self.args, new_encoder, self.device)
            print("E_Net: ConvNeXt-V2_tiny + weights")
        else:
            raise ValueError(f"Unsupported E_net type: {self.args.E_net}")

        # Finish an in-progress ESG window using the old encoder before its
        # parameters and optimizer become unreachable.
        accumulated_steps = self.accumulation_counts['ESG']
        if accumulated_steps > 0:
            self._step_partial_accumulation(
                [self.optims[net] for net in ['E', 'S', 'G']],
                accumulated_steps,
            )
            self.accumulation_counts['ESG'] = 0

        self.nets['E'] = new_encoder
        self.nets['E'].train()
        self.optims['E'] = self._create_encoder_optimizer()
        print("Replaced E optimizer with one bound to the new encoder parameters")

    def set_networks(self):
        # Model
        cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
        head_cfg = (256, 3)  # (hidden dimension, num_layers blstm)
        # width_divisor = 8

        model_reco = CRNN(
            cnn_cfg,
            head_cfg,
            self.charset.get_nb_char(),
            head_type=self.args.model_type,
        )
        
        # after_blstm = model_reco.top

        # Init model
        print("Initializing model weights kaiming")
        for p in model_reco.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p, nonlinearity="relu")

        if os.path.isfile(self.args.path_model):
            load_pretrained_model(self.args, model_reco, self.device)
        else:
            print("No pretrained model load")

        if self.args.freeze_shortcut == "Yes":
            # For head_type="cnn", top.cnn is the recognition classifier. Keep
            # the source-trained character classifier fixed while adapting the
            # feature extractor to the new domains.
            for param in model_reco.top.cnn.parameters():
                param.requires_grad_(False)
            print("Frozen task-network shortcut/classifier: top.cnn")

        print(f"Transferring model to {str(self.device)}...")

        model_reco = model_reco.to(self.device)

        # Replace DRANet Encoder with your custom Extractor
        if self.args.E_net == 'convnextv2_large' and self.args.E_net_swap_epoch == 0:
            self.nets['E'] = convnextv2_large(net="E", out_index=1)
            load_pretrained_model_ConvNext(self.args, self.nets['E'], self.device)         
            print("E_Net: ConvNeXt-V2_large + weights")
        elif self.args.E_net == 'convnextv2_tiny' and  self.args.E_net_swap_epoch == 0:
            self.nets['E'] = convnextv2_tiny(net="E", out_index=1)
            load_pretrained_model_ConvNext(self.args, self.nets['E'], self.device)         
            print("E_Net: ConvNeXt-V2_tiny + weights")
        else:
            # before_blstm = model_reco.features
            before_blstm = copy.deepcopy(model_reco.features)
            self.nets['E'] = before_blstm.to(self.device)
            print("E_Net: CRNN-Before-BLSTM")
        with torch.no_grad():
            # dummy = torch.zeros(1, 1, self.imsize[0], self.imsize[1]).to(self.device)  # 1 channel input
            dummy = torch.zeros(1, 1, self.args.height_max,
                                self.args.width_max + self.args.pad_left + self.args.pad_right).to(self.device)  # 1 channel input
            feat = self.nets['E'](dummy)
            encoder_out_channels = feat.size(1)
            print(feat.shape)
            print(f"Encoder output channels: {encoder_out_channels}")
        # dsafsfsdf
        
        if self.args.dra_arch == "style_film":
            print("DRA architecture: style_film (Separator_FL + Generator_FL)")
            self.nets['G'] = Generator_FL(
                out_height=self.args.height_max,
                out_width_with_pad=self.args.width_max + self.args.pad_left + self.args.pad_right,
                input_channels=encoder_out_channels,
                style_dim=128,
                output_activation=self.args.generator_output_activation,
                vertical_refine_steps=self.args.generator_vertical_refine_steps,
                vertical_refine_channels=self.args.generator_vertical_refine_channels,
            )
            self.nets['S'] = Separator_FL(ch=encoder_out_channels, style_dim=128)
        else:
            print(f"DRA architecture: {self.args.dra_arch} (Separator + Generator)")
            self.nets['G'] = Generator(
                out_height=self.args.height_max,
                out_width_with_pad=self.args.width_max + self.args.pad_left + self.args.pad_right,
                input_channels=encoder_out_channels,
                output_activation=self.args.generator_output_activation,
                vertical_refine_steps=self.args.generator_vertical_refine_steps,
                vertical_refine_channels=self.args.generator_vertical_refine_channels,
            )
            self.nets['S'] = Separator((self.args.height_max,
                                        self.args.width_max + self.args.pad_left + self.args.pad_right),
                                        self.training_converts)

        self.nets['D'] = dict()
        for dset in self.args.datasets:
            if self.args.task == 'clf':
                if dset == 'U':
                    self.nets['D'][dset] = Discriminator_USPS()
                else:
                    if self.args.discriminator == "PatchGAN":
                        self.nets['D'][dset] = PatchGAN_Discriminator(dis_last_layer_stride=self.args.dis_last_layer_stride)
                    else:
                        self.nets['D'][dset] = Discriminator_MNIST()
            else:
                self.nets['D'][dset] = PatchGAN_Discriminator()
        self.nets['T'] = dict()
        for cv in self.test_converts:
            if self.args.task == 'clf':
                # Each conversion gets independent parameters, optimizer state,
                # and gradient-accumulation state.
                self.nets['T'][cv] = copy.deepcopy(model_reco) if self.args.task_deepcopy == "yes" else model_reco
            elif self.args.task == 'seg':
                self.nets['T'][cv] = drn26()

        # Initialisation des paramètres (sauf E et T[cv] si clf)
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    init_params(self.nets[net][dset])
            elif net == 'T':
                if self.args.task == 'clf':
                    for cv in self.test_converts:
                        continue  # Ne pas initialiser T[cv] si clf
                elif self.args.task == 'seg':
                    for cv in self.test_converts:
                        init_params(self.nets[net][cv])  # Initialiser uniquement pour segmentation
            elif net == 'E':
                continue  # Ne pas initialiser E
            else:
                init_params(self.nets[net])
        if self.args.P_net == 'convnextv2_large':
            self.nets['P'] = convnextv2_large(net="P")
            load_pretrained_model_ConvNext(
                self.args,
                self.nets['P'],
                self.device,
                pretrained_model_file=self.args.perceptual_convnextv2_model_path or self.args.convnextv2_model_path,
            )
            print("P_Net: ConvNeXt-V2_large + weights")
            # checkpoint = torch.load(self.args.convnextv2_model_path, map_location="cpu")
            # msg = self.nets['P'].load_state_dict(checkpoint["model"], strict=True)
            # print("P_Net: ConvNeXt-V2 weights loading:")
            # print(msg)
            for p in self.nets['P'].parameters():
                p.requires_grad = False
            self.nets['P'].eval() # Put in eval mode
        elif self.args.P_net == 'convnextv2_base':
            self.nets['P'] = convnextv2_base(net="P")
            load_pretrained_model_ConvNext(
                self.args,
                self.nets['P'],
                self.device,
                pretrained_model_file=self.args.perceptual_convnextv2_model_path or self.args.convnextv2_model_path,
            )
            print("P_Net: ConvNeXt-V2_base + weights")
            for p in self.nets['P'].parameters():
                p.requires_grad = False
            self.nets['P'].eval()
        elif self.args.P_net == 'convnextv2_tiny':
            self.nets['P'] = convnextv2_tiny(net="P")
            load_pretrained_model_ConvNext(
                self.args,
                self.nets['P'],
                self.device,
                pretrained_model_file=self.args.perceptual_convnextv2_model_path or self.args.convnextv2_model_path,
            )
            print("P_Net: ConvNeXt-V2_tiny + weights")
            # checkpoint = torch.load(self.args.convnextv2_model_path, map_location="cpu")
            # msg = self.nets['P'].load_state_dict(checkpoint["model"], strict=True)
            # print("P_Net: ConvNeXt-V2 weights loading:")
            # print(msg)
            for p in self.nets['P'].parameters():
                p.requires_grad = False
            self.nets['P'].eval() # Put in eval mode
        else:
            self.nets['P'] = VGG19()

        # Move nets to device
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset] = self.nets[net][dset].to(self.device)
            elif net == 'T':
                for cv in self.test_converts:
                    self.nets[net][cv] = self.nets[net][cv].to(self.device)
            else:
                self.nets[net] = self.nets[net].to(self.device)

    def set_optimizers(self):
        self.optims['E'] = self._create_encoder_optimizer()

        self.optims['D'] = dict()
        for dset in self.args.datasets:
            print(f"dataset {dset}")
            self.optims['D'][dset] = optim.Adam(self.nets['D'][dset].parameters(), lr=self.args.lr_dra,
                                                betas=(self.args.beta1, 0.999),
                                                weight_decay=self.args.weight_decay_dra)

        self.optims['G'] = optim.Adam(self.nets['G'].parameters(), lr=self.args.lr_dra,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_dra)

        self.optims['S'] = optim.Adam(self.nets['S'].parameters(), lr=self.args.lr_dra,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_dra)

        self.optims['T'] = dict()
        for convert in self.test_converts:
            if self.args.task == 'clf':
                trainable_parameters = [
                    param for param in self.nets['T'][convert].parameters()
                    if param.requires_grad
                ]
                self.optims['T'][convert] = optim.SGD(trainable_parameters, lr=self.args.lr_clf,
                                                      momentum=0.9,
                                                      weight_decay=self.args.weight_decay_task)
            elif self.args.task == 'seg':
                self.optims['T'][convert] = optim.SGD(self.nets['T'][convert].parameters(), lr=self.args.lr_seg,
                                                      momentum=0.9,
                                                      weight_decay=self.args.weight_decay_task)

    def set_zero_grad(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].zero_grad()
            elif net == 'T':
                for convert in self.test_converts:
                    self.nets[net][convert].zero_grad()
            else:
                self.nets[net].zero_grad()

    def set_train(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].train()
            elif net == 'T':
                for convert in self.test_converts:
                    self.nets[net][convert].train()
            else:
                self.nets[net].train()

    def set_eval(self):
        for convert in self.test_converts:
            self.nets['T'][convert].eval()

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.args.datasets:
            try:
                batch_data[dset] = next(batch_data_iter[dset])
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = next(batch_data_iter[dset])
        return batch_data

    def train_dis(self, imgs):  # Train Discriminators (D)
        # self.set_zero_grad()
        features, converted_imgs, D_outputs_fake, D_outputs_real = dict(), dict(), dict(), dict()

        # Real
        for dset in self.args.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            if self.args.task == 'clf':
                if self.args.discriminator == "PatchGAN":
                    if self.args.slice_patches == 1:
                        D_outputs_real[dset] = self.nets['D'][dset](slice_patches(imgs[dset]))
                    else:
                        D_outputs_real[dset] = self.nets['D'][dset](imgs[dset])
                else:
                    D_outputs_real[dset] = self.nets['D'][dset](imgs[dset])
            else:
                D_outputs_real[dset] = self.nets['D'][dset](slice_patches(imgs[dset]))

            

        contents, styles = self.nets['S'](features, self.training_converts)

        # CADT
        if self.args.CADT:
            for convert in self.training_converts:
                source, target = convert.split("_2_", 1)
                # target = "read_" + target
                _, styles[target] = cadt(contents[source], contents[target], styles[target])

        # Fake
        for convert in self.training_converts:
            source, target = convert.split("_2_", 1)
            # target = "read_" + target
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target]).detach() # Detach to avoid backprop through G
            if self.args.task == 'clf':
                if self.args.discriminator == "PatchGAN":
                    if self.args.slice_patches == 1:
                        D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))
                    else:
                        D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
                else:
                    D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
            else:
                D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))

            
        errD = self.loss_fns.dis(D_outputs_real, D_outputs_fake)
        (errD / self.args.accum_steps).backward()
        self.accumulation_counts['D'] += 1

        for name, p in self.nets['D'][target].named_parameters():
            if p.grad is not None:
                print(name, p.grad.abs().mean().item())
                break

        print("errD =", errD.item())

        if self.accumulation_counts['D'] == self.args.accum_steps:
            self._step_optimizers(
                self.optims['D'].values(),
                "D",
                max_grad_norm=self.args.clip_grad_dra,
            )
            self.accumulation_counts['D'] = 0
        self.losses['D'] = errD.item()

    def train_save_gen_images(self, imgs, converted_imgs, source, target, convert, interval=500, debug_dir="debug_images"):  # Train Generators (G, S)
        if self.step % interval == 0:
            save_dir = os.path.join(self.args.path_log, debug_dir)
            os.makedirs(save_dir, exist_ok=True)

            source_img = imgs[source][:4]
            target_img = imgs[target][:4]
            adapted_img = converted_imgs[convert][:4]

            print("Adapted Image Stats: min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(
                adapted_img.min().item(),
                adapted_img.max().item(),
                adapted_img.mean().item()
                )      
            )

            # Inputs use [0, 1]. Legacy generators configured with plain tanh
            # produce [-1, 1], so remap only their debug preview before saving.
            # This does not alter the tensor used for training.
            adapted_preview = adapted_img
            if self.args.generator_output_activation == "tanh":
                adapted_preview = (adapted_preview + 1.0) * 0.5

            grid = torch.cat(
                [source_img, adapted_preview, target_img],
                dim=0
            ).clamp(0.0, 1.0)
                                
            vutils.save_image(
                grid,
                os.path.join(
                    save_dir,
                    f"{convert}_triplet_step_{self.step}.png"
                ),
                # One row per group (source, adapted, target), independently
                # of the configured micro-batch size.
                nrow=source_img.size(0),
                normalize=False
            )

    def train_task(self, imgs, y_enc, x_reduced_len, y_len_enc, print_target_source):  # Train Task Networks (T)
        # print("Training task networks...##################################################")
        # self.set_zero_grad()
        features = dict()
        converted_imgs = dict()
        pred = dict()
        ctc_loss = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")
        converts = self.training_converts if self.args.task == 'clf' else self.test_converts

        def maybe_blur_converted(batch):
            if self.args.apply_blur != 1:
                return batch
            blurred = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(batch, (1, 1, 1, 1), mode="reflect"),
                kernel_size=3,
                stride=1,
            )
            blur_mask = torch.rand(
                (batch.size(0), 1, 1, 1),
                device=batch.device,
            ) < 0.5
            return torch.where(blur_mask, blurred, batch)

        with torch.no_grad():
            for dset in self.args.datasets:
                features[dset] = self.nets['E'](imgs[dset])
            contents, styles = self.nets['S'](features, converts)
            for convert in converts:
                source, target = convert.split("_2_", 1)
                # target = "read_" + target
                if self.args.CADT:
                    _, styles[target] = cadt(contents[source], contents[target], styles[target])
                converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
                converted_imgs[convert] = torch.clamp(converted_imgs[convert], 0.0, 1.0)
                converted_imgs[convert] = maybe_blur_converted(converted_imgs[convert])
                print("############## Gnerated Shape ##############")
                print(converted_imgs[convert].shape)

                # Visualize the converted images for debugging
                if self.args.debug_images:
                    self.train_save_gen_images(imgs, converted_imgs, source, target, convert)

            for convert in list(set(self.test_converts) - set(self.training_converts)):
                features_mid = dict()
                source, target = convert.split("_2_", 1)
                # target = "read_" + target
                if print_target_source:
                    print("T-source: " + source)
                    print("T-target: " + target)
                mid = list(set(self.args.datasets) - {source, target})[0]
                convert1 = source + '_2_' + mid
                convert2 = mid + '_2_' + target
                features_mid[convert1] = self.nets['E'](converted_imgs[convert1])
                contents_mid, _ = self.nets['S'](features_mid, [convert2])
                converted_imgs[convert] = self.nets['G'](contents_mid[convert2], styles[target])
                converted_imgs[convert] = torch.clamp(converted_imgs[convert], 0.0, 1.0)
                converted_imgs[convert] = maybe_blur_converted(converted_imgs[convert])

        for convert in self.test_converts:
            if convert.startswith(self.args.dataset_source + "_2_") or self.args.T_both_directions == 1:
                task_loss = None
                optimizer = self.optims["T"][convert]
                # print("#######################################")
                # print(f"Training task network for convert: {convert}")
                pred = self.nets['T'][convert](converted_imgs[convert])
                source, target = convert.split("_2_", 1)
                loss_fake = self.loss_fns.single_task_CRNN(pred, source, ctc_loss, y_enc, x_reduced_len, y_len_enc)

                pred = self.nets['T'][convert](imgs[source])
                loss_real = self.loss_fns.single_task_CRNN(pred, source, ctc_loss, y_enc, x_reduced_len, y_len_enc)

                task_loss = loss_fake if self.args.T_loss_combo == "fake_loss" else loss_fake + loss_real

                errT = task_loss / self.args.accum_steps

                if self.step > self.args.T_training_start:
                    print("Train Net-T")
                    errT.backward()
                    self.task_accumulation_counts[convert] += 1
                    if self.task_accumulation_counts[convert] == self.args.accum_steps:
                        self._step_optimizers(
                            [optimizer],
                            f"T/{convert}",
                            max_grad_norm=self.args.clip_grad_task,
                        )
                        self.task_accumulation_counts[convert] = 0

                self.losses[f"T/{convert}"] = task_loss.item()



    def train_esg(self, imgs, y_enc, x_reduced_len, y_len_enc):  # Train Encoder(E), Separator(S), Generator(G)
        features, converted_imgs, recon_imgs, D_outputs_fake = dict(), dict(), dict(), dict()
        features_converted = dict()
        perceptual, style_gram = dict(), dict()
        perceptual_converted, style_gram_converted = dict(), dict()
        con_sim = dict()
        Latent_consistency_loss_style, Latent_consistency_loss_content = 0, 0
        Task_loss = torch.tensor(0.0, device=self.device)
        ctc_loss = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")

        # D participates in the forward pass so its output can guide G, but its
        # parameters must not receive gradients from the generator objective.
        discriminator_grad_state = [
            (parameter, parameter.requires_grad)
            for discriminator in self.nets['D'].values()
            for parameter in discriminator.parameters()
        ]
        for parameter, _ in discriminator_grad_state:
            parameter.requires_grad_(False)

        for dset in self.args.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            with torch.no_grad():
                perceptual[dset] = self.nets['P'](imgs[dset])

            style_gram[dset] = [gram(fmap) for fmap in perceptual[dset][:-1]]
        contents, styles = self.nets['S'](features, self.training_converts)

        # Reconstruction now follows the same content/style interface as
        # translation instead of bypassing S with a scalar style placeholder.
        for dset in self.args.datasets:
            recon_imgs[dset] = self.nets['G'](contents[dset], styles[dset])

        for convert in self.training_converts:
            source, target = convert.split("_2_", 1)
            # target = "read_" + target
            if self.args.CADT:
                con_sim[convert], styles[target] = cadt(contents[source], contents[target], styles[target])
                style_gram[target] = cadt_gram(style_gram[target], con_sim[convert])
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])

            if self.args.task == 'clf':
                if self.args.discriminator == "PatchGAN":
                    if self.args.slice_patches == 1:
                        D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))
                    else:
                        D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
                else:
                    D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
                # print("############## Discriminator Shape ##############")
                # print(D_outputs_fake[convert].shape)

                if convert.startswith(self.args.dataset_source + "_2_") and self.args.lambda_task > 0 and self.step >= self.args.T_ESG_loss_start:
                    task_parameter_states = [
                        (param, param.requires_grad)
                        for param in self.nets['T'][convert].parameters()
                    ]
                    for p, _ in task_parameter_states:
                        p.requires_grad_(False)
                    pred = self.nets['T'][convert](converted_imgs[convert])
                    source, target = convert.split("_2_", 1)
                    Task_loss = self.loss_fns.single_task_CRNN(pred, source, ctc_loss, y_enc, x_reduced_len, y_len_enc)
                                
                    # pred = self.nets['T'][convert](imgs[source])
                    # Task_loss += self.loss_fns.single_task_CRNN(pred, source, ctc_loss, y_enc, x_reduced_len, y_len_enc)
                    for p, requires_grad in task_parameter_states:
                        p.requires_grad_(requires_grad)
            else:
                D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))
            features_converted[convert] = self.nets['E'](converted_imgs[convert])
            perceptual_converted[convert] = self.nets['P'](converted_imgs[convert])
            style_gram_converted[convert] = [gram(fmap) for fmap in perceptual_converted[convert][:-1]]
        contents_converted, styles_converted = self.nets['S'](features_converted)

        ############################################################################
        if self.args.lambda_latent_content > 0 or self.args.lambda_latent_style > 0:
            for convert in self.training_converts:
                source, target = convert.split("_2_", 1)

                Latent_consistency_loss_style += self.loss_fns.latent_consistency(styles[target].detach(), styles_converted[convert], "l_style")

                if self.args.latent_content_loss == "cosine":
                    Latent_consistency_loss_content +=  self.loss_fns.latent_cosine_consistency(contents[source].detach(), contents_converted[convert], "l_content")
                else:
                    Latent_consistency_loss_content += self.loss_fns.latent_consistency(contents[source].detach(), contents_converted[convert], "l_content")        

            # print(
            #     f"LatentStyle={Latent_consistency_loss_style.item():.4f}",
            #     f"LatentContent={Latent_consistency_loss_content.item():.4f}"
            # )       
        ############################################################################

        Content_loss = self.loss_fns.content_perceptual(perceptual, perceptual_converted)
        Style_loss = self.loss_fns.style_perceptual(style_gram, style_gram_converted)
        Consistency_loss = self.loss_fns.consistency(contents, styles, contents_converted, styles_converted,
                                                     self.training_converts)
        G_loss = self.loss_fns.gen(D_outputs_fake)
        Recon_loss = self.loss_fns.recon(imgs, recon_imgs)


        loss_tensors = {
            "G": G_loss,
            "Recon": Recon_loss,
            "Consis": Consistency_loss,
            "Content": Content_loss,
            "Style": Style_loss,
            "Task": Task_loss,
            "Latent_Consistency_Style": Latent_consistency_loss_style,
            "Latent_Consistency_Content": Latent_consistency_loss_content 
        }
        base_lambdas = {
            "G": float(self.args.lambda_g),
            "Recon": float(self.args.lambda_recon),
            "Consis": float(self.args.lambda_consis),
            "Content": float(self.args.lambda_content),
            "Style": float(self.args.lambda_style),
            "Task": float(self.args.lambda_task),
            "Latent_Consistency_Style": float(self.args.lambda_latent_style),
            "Latent_Consistency_Content": float(self.args.lambda_latent_content)
        }
        errESG, effective_lambdas, dynamic_ratios = self._dynamic_esg_loss_total(loss_tensors, base_lambdas)

        print(f"Effective Lambdas: {effective_lambdas}")

        print("errESG : ", errESG.item())
        print("self.args.accum_steps : ", self.args.accum_steps)

        for name, loss in loss_tensors.items():
            if torch.is_tensor(loss):
                value = loss.detach()
                value_text = value.item()
                finite = torch.isfinite(value).all().item()
                requires_grad = loss.requires_grad
            else:
                value_text = float(loss)
                finite = math.isfinite(value_text)
                requires_grad = False
            print(
                f"{name}: value={value_text}, "
                f"finite={finite}, "
                f"requires_grad={requires_grad}"
            )

        (errESG / self.args.accum_steps).backward()
        for parameter, requires_grad in discriminator_grad_state:
            parameter.requires_grad_(requires_grad)

        self.accumulation_counts['ESG'] += 1
        if self.accumulation_counts['ESG'] == self.args.accum_steps:
            self._step_optimizers(
                [self.optims[net] for net in ['E', 'S', 'G']],
                "E/S/G",
                max_grad_norm=self.args.clip_grad_dra,
            )
            self.accumulation_counts['ESG'] = 0

        self.losses['G'] = G_loss.item()
        self.losses['Recon'] = Recon_loss.item()
        self.losses['Consis'] = Consistency_loss.item()
        self.losses['Content'] = Content_loss.item()
        self.losses['Style'] = Style_loss.item()
        self.losses['wG'] = effective_lambdas['G'] * G_loss.item()
        self.losses['wRecon'] = effective_lambdas['Recon'] * Recon_loss.item()
        self.losses['wConsis'] = effective_lambdas['Consis'] * Consistency_loss.item()
        self.losses['wContent'] = effective_lambdas['Content'] * Content_loss.item()
        self.losses['wStyle'] = effective_lambdas['Style'] * Style_loss.item()
        self.losses['wESG'] = errESG.item()
        if self.args.lambda_task > 0:
            self.losses['ESG_Task'] = Task_loss.item()
            self.losses['wESG_Task'] = effective_lambdas['Task'] * Task_loss.item()
        if self.args.lambda_latent_content > 0 or self.args.lambda_latent_style > 0:
            self.losses['Latent_Consistency_Style'] = Latent_consistency_loss_style.item()
            self.losses['Latent_Consistency_Content'] = Latent_consistency_loss_content.item()
            self.losses['wLatent_Consistency_Style'] = effective_lambdas['Latent_Consistency_Style'] * Latent_consistency_loss_style.item()
            self.losses['wLatent_Consistency_Content'] = effective_lambdas['Latent_Consistency_Content'] * Latent_consistency_loss_content.item()



        if self.dynamic_loss_weighting:
            for key, value in effective_lambdas.items():
                self.losses[f'dynLambda_{key}'] = value
            for key, value in dynamic_ratios.items():
                self.losses[f'dynRatio_{key}'] = value

        print(
            f"G={G_loss.item():.4f}",
            f"Recon={Recon_loss.item():.4f}",
            f"Content={Content_loss.item():.4f}",
            f"Style={Style_loss.item():.4f}",
            f"Consis={Consistency_loss.item():.4f}"
        )


    def eval(self, cv):
        source, target = cv.split("_2_", 1)
        # target = "read_" + target
        # print("source: " + source)
        # print("target: " + target)
        self.set_eval()
        token_blank = self.char_dict[BLANK_STR_TOKEN]

        if self.args.task == 'clf':
            metrics_main = MetricLossCERWER("Main")
            metrics_shortcut = MetricLossCERWER("Shortcut")
            ctc_loss_fn = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(self.test_loader[target]):
                    x = batch_data["imgs"]
                    x_reduced_len = batch_data["w_reduce"]
                    y_enc = batch_data["label_ind"]
                    y_len_enc = batch_data["label_ind_length"]
                    y_gt_txt = batch_data["label_str"]

                    # if self.args.eval_GANimg:
                    #     features = dict()
                    #     features[target] = self.nets['E'](x.to(self.device))
                    #     contents, styles = self.nets['S'](features)
                    #     converted_imgs = self.nets['G'](contents[target], styles[target])
                    #     converted_imgs = torch.clamp(converted_imgs, 0.0, 1.0)
                    #     x = converted_imgs

                    x, y_enc = x.to(self.device), y_enc.to(self.device)

                    # Remove text padding
                    y_gt_txt = [t.strip() for t in y_gt_txt]
                    # Remove double spaces
                    y_gt_txt = [t.replace("  ", " ") for t in y_gt_txt]

                    nb_item_batch = x.shape[0]

                    y_pred, _, _ = self.nets['T'][cv](x)
                    output, aux_output = y_pred

                    # Main head
                    output_log = torch.nn.functional.log_softmax(output, dim=-1)

                    ctc_loss = ctc_loss_fn(output_log, y_enc, x_reduced_len, y_len_enc)

                    metrics_main.add_loss(ctc_loss.item(), nb_item_batch)

                    # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
                    output_log = output_log.transpose(0, 1)

                    top = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                           enumerate(output_log)]
                    predictions_text = [ctc_best_path_one(p, self.char_list, token_blank) for p in top]

                    predictions_text = [t.strip() for t in predictions_text]  # Remove text padding
                    predictions_text = [t.replace("  ", " ") for t in predictions_text]

                    cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text)]
                    metrics_main.add_cer(sum(cers), nb_chars_from_list(y_gt_txt))

                    if aux_output is not None:
                        # Auxiliary shortcut exists only for head_type="crnn".
                        output_log = torch.nn.functional.log_softmax(aux_output, dim=-1)

                        ctc_loss = ctc_loss_fn(output_log, y_enc, x_reduced_len, y_len_enc)
                        metrics_shortcut.add_loss(ctc_loss.item(), nb_item_batch)

                        # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
                        output_log = output_log.transpose(0, 1)

                        top_aux = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                                   enumerate(output_log)]

                        predictions_text_aux = [ctc_best_path_one(p, self.char_list, token_blank) for p in top_aux]
                        predictions_text_aux = [t.strip() for t in predictions_text_aux]  # Remove text padding
                        predictions_text_aux = [t.replace("  ", " ") for t in predictions_text_aux]

                        cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_aux)]
                        metrics_shortcut.add_cer(sum(cers), nb_chars_from_list(y_gt_txt))

            # Main head
            print(f"metrics_main.nb_letters: {metrics_main.nb_letters}")
            if metrics_main.nb_letters != 0:
                cer_main = metrics_main.cer / metrics_main.nb_letters
            else:
                cer_main = 1.0  # fallback if no letters
            print(f"Main CER for {cv}: {cer_main:.4f}")

            cer_shortcut = None
            if metrics_shortcut.nb_letters != 0:
                cer_shortcut = metrics_shortcut.cer / metrics_shortcut.nb_letters
                print(f"Shortcut CER for {cv}: {cer_shortcut:.4f}")
                if cer_shortcut < self.best_cer_shortcut[cv]:
                    self.best_cer_shortcut[cv] = cer_shortcut

            if self.args.debug_pc == 0:
                eval_metrics = {f"Main/CER_{cv}": cer_main}
                if cer_shortcut is not None:
                    eval_metrics[f"Shortcut/CER_{cv}"] = cer_shortcut
                wandb.log(eval_metrics, step=self.step)

            if cer_main < self.best_cer[cv]:
                self.best_cer[cv] = cer_main
                if self.args.debug_pc == 0:
                    wandb.log({f'Best_Main_CER/{cv}': cer_main}, step=self.step)
                    torch.save(
                        self.nets["T"][cv].state_dict(),
                        self.checkpoint + f'/net_best_T_{cv}.pth'
                    )

        self.set_train()


    def eval_GAN_img(self):
        self.set_eval()
        for net in ['E', 'S', 'G']:
            self.nets[net].eval()

        token_blank = self.char_dict[BLANK_STR_TOKEN]
        source, target = self.args.datasets[0], self.args.datasets[1]
        source_2_target = source + "_2_" + target
        target_2_source = target + "_2_" + source
        metrics_by_style = {
            style_case: MetricLossCERWER("Main")
            for style_case in [source, target]
        }

        try:
            source_batch = next(iter(self.test_loader[source]))
        except StopIteration as exc:
            raise RuntimeError(f"The source test loader '{source}' is empty") from exc

        source_imgs = source_batch["imgs"].to(self.device)
        first_batch_outputs = {}
        diagnostic_content = None
        diagnostic_source_style = None

        with torch.no_grad():
            source_features = self.nets['E'](source_imgs)
            source_contents, source_styles = self.nets['S'](
                {source: source_features},
                debug=False,
            )
            source_style_prototype = source_styles[source].mean(dim=0, keepdim=True)
            source_content_prototype = source_contents[source].mean(dim=0, keepdim=True)

            for batch_idx, batch_data in enumerate(self.test_loader[target]):
                target_imgs = batch_data["imgs"].to(self.device)
                x_reduced_len = batch_data["w_reduce"]
                y_gt_txt = [text.strip().replace("  ", " ") for text in batch_data["label_str"]]

                target_features = self.nets['E'](target_imgs)
                contents, styles = self.nets['S'](
                    {target: target_features},
                    [target_2_source],
                    debug=False,
                )

                batch_size = target_imgs.size(0)
                source_style = source_style_prototype.expand(
                    batch_size,
                    *source_style_prototype.shape[1:],
                ).contiguous()
                if self.args.CADT:
                    source_content = source_content_prototype.expand(
                        batch_size, -1, -1, -1
                    ).contiguous()
                    _, source_style = cadt(
                        contents[target],
                        source_content,
                        source_style,
                    )

                styles_for_evaluation = {
                    source: source_style,
                    target: styles[target],
                }

                for style_case, style in styles_for_evaluation.items():
                    raw_converted = self.nets['G'](contents[target_2_source], style)
                    converted_imgs = raw_converted.clamp(0.0, 1.0)

                    # Both variants retain target-domain content. Evaluate them
                    # with the forward task model trained for source -> target.
                    y_pred, _, _ = self.nets['T'][source_2_target](converted_imgs)
                    output, _ = y_pred
                    output_log = torch.nn.functional.log_softmax(output, dim=-1).transpose(0, 1)
                    top = [
                        torch.argmax(log_probs, dim=1).detach().cpu().numpy()[:x_reduced_len[j]]
                        for j, log_probs in enumerate(output_log)
                    ]
                    predictions_text = [ctc_best_path_one(prediction, self.char_list, token_blank) for prediction in top]
                    predictions_text = [text.strip().replace("  ", " ")  for text in predictions_text]
                    cers = [editdistance.eval(reference, prediction)  for reference, prediction in zip(y_gt_txt, predictions_text) ]
                    metrics_by_style[style_case].add_cer(sum(cers), nb_chars_from_list(y_gt_txt))

                    if batch_idx == 0:
                        first_batch_outputs[style_case] = raw_converted
                        preview_imgs = {source: source_imgs, target: target_imgs}
                        preview_converted_imgs = {target_2_source: converted_imgs}
                        if self.args.debug_images:  
                            self.train_save_gen_images(
                                preview_imgs,
                                preview_converted_imgs,
                                target,
                                source,
                                target_2_source,
                                interval=1,
                                debug_dir=f"debug_GAN_inference_images-Style_{style_case}",
                            )

                if batch_idx == 0:
                    diagnostic_content = contents[target_2_source]
                    diagnostic_source_style = source_style

            for style_case, metrics_main in metrics_by_style.items():
                cer_main = (
                    metrics_main.cer / metrics_main.nb_letters
                    if metrics_main.nb_letters != 0
                    else 1.0
                )
                print(f"GAN-image test CER for {source_2_target}, target content, style {style_case}: {cer_main:.4f}")
                if self.args.debug_pc == 0:
                    wandb.log({
                        f"Main/CER_{source_2_target}-GAN_IMG_style-{style_case}": cer_main
                    }, step=self.step)

            if source in first_batch_outputs and target in first_batch_outputs:
                print(
                    "style swap output difference:",
                    (first_batch_outputs[source] - first_batch_outputs[target]).abs().mean().item(),
                )

            if diagnostic_content is not None:
                out_normal = self.nets['G'](diagnostic_content, diagnostic_source_style)
                out_zero = self.nets['G'](
                    diagnostic_content,
                    torch.zeros_like(diagnostic_source_style),
                )
                out_10x = self.nets['G'](diagnostic_content, 10.0 * diagnostic_source_style)
                out_100x = self.nets['G'](diagnostic_content, 100.0 * diagnostic_source_style)
                print("\n############## STYLE SENSITIVITY TEST ##############")
                print("normal vs zero:", (out_normal - out_zero).abs().mean().item())
                print("normal vs 10x:", (out_normal - out_10x).abs().mean().item())
                print("normal vs 100x:", (out_normal - out_100x).abs().mean().item())
                print("zero vs 100x:", (out_zero - out_100x).abs().mean().item())
                print("normal output abs mean:", out_normal.abs().mean().item())
                print("style abs mean:", diagnostic_source_style.abs().mean().item())
                print("####################################################\n")

        self.set_train()





    def print_loss(self):
        best = ''
        wandb_log_data = {}

        if self.args.task == 'clf':
            for cv in self.test_converts:
                cer_main = self.best_cer[cv]
                best += f'{cv}: Main_CER={cer_main:.4f}'
                wandb_log_data[f'best_main_cer/{cv}'] = cer_main
                if self.args.model_type == "crnn":
                    cer_shortcut = self.best_cer_shortcut[cv]
                    best += f', Shortcut_CER={cer_shortcut:.4f}'
                    wandb_log_data[f'best_shortcut_cer/{cv}'] = cer_shortcut
                best += ' |'

        # Log current losses
        losses = ''
        for key, value in self.losses.items():
            losses += f'{key}: {value:.2f}|'
            wandb_log_data[f'loss/{key}'] = value

        # Terminal log
        self.logger.info('[%d/%d] %s| %s %s' % (self.step, self.args.iter, losses, best, self.args.ex))

        return wandb_log_data

    def _step_partial_accumulation(
        self,
        optimizers,
        accumulated_steps,
        group_name,
        max_grad_norm=0.0,
    ):
        if accumulated_steps == 0:
            return

        optimizers = list(optimizers)
        # Each micro-loss was divided by the full accumulation size. Correct
        # the final partial window so it is averaged over the steps it contains.
        gradient_scale = self.args.accum_steps / accumulated_steps
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                for parameter in param_group['params']:
                    if parameter.grad is not None:
                        parameter.grad.mul_(gradient_scale)
        self._step_optimizers(
            optimizers,
            group_name,
            max_grad_norm=max_grad_norm,
        )

    def flush_partial_accumulation(self):
        self._step_partial_accumulation(
            self.optims['D'].values(),
            self.accumulation_counts['D'],
            "D (partial)",
            max_grad_norm=self.args.clip_grad_dra,
        )
        self.accumulation_counts['D'] = 0

        self._step_partial_accumulation(
            [self.optims[net] for net in ['E', 'S', 'G']],
            self.accumulation_counts['ESG'],
            "E/S/G (partial)",
            max_grad_norm=self.args.clip_grad_dra,
        )
        self.accumulation_counts['ESG'] = 0

        for convert, accumulated_steps in self.task_accumulation_counts.items():
            self._step_partial_accumulation(
                [self.optims['T'][convert]],
                accumulated_steps,
                f"T/{convert} (partial)",
                max_grad_norm=self.args.clip_grad_task,
            )
            self.task_accumulation_counts[convert] = 0

    def train(self):
        self.set_default()
        self.set_networks()
        self.set_optimizers()
        if self.args.resume_checkpoint is not None:
            if self.args.load_path is not None:
                raise ValueError(
                    "Use either --resume_checkpoint for a full resume or "
                    "--load_path for network weights only, not both."
                )
            self.load_training_checkpoint(self.args.resume_checkpoint)
        elif self.args.load_path is not None:
            self.load_networks(self.args.checkpoint_stage)

        # self.eval_GAN_img()
        self.set_train()
        self.logger.info(self.loss_fns.alpha)

        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])

        # eval epoch 0
        print("Eval start:")
        for cv in self.test_converts:
            self.eval(cv)
        self.save_training_checkpoint()
        self.save_networks_last()

        for i in range(self.step, self.args.iter):
            self.step += 1

            if self.args.E_net_swap_epoch > 0 and self.step == self.args.E_net_swap_epoch:
                print(f"Swapping E network at step {self.step}")
                self.swap_E_net()

            # get batch data
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels, x_reduced_len, y_len_enc, y_enc = dict(), dict(), dict(), dict(), dict()
            # min_batch = self.args.batch

            for dset in self.args.datasets:
                x = batch_data[dset]["imgs"]
                y_gt_txt = batch_data[dset]["label_str"]
                x_reduced_len[dset] = batch_data[dset]["w_reduce"]
                y_enc0 = batch_data[dset]["label_ind"]
                y_len_enc[dset] = batch_data[dset]["label_ind_length"]
                # Remove text padding
                y_gt_txt = [t.strip() for t in y_gt_txt]
                # Remove double spaces
                y_gt_txt = [t.replace("  ", " ") for t in y_gt_txt]

                imgs[dset], labels[dset], y_enc[dset] = x.to(self.device), y_gt_txt, y_enc0.to(self.device)

                if self.args.task == 'seg':
                    labels[dset] = labels[dset].long()

            # training steps
            self.train_dis(imgs)

            for t in range(1 if self.args.discriminator == "PatchGAN" else 2):
                self.train_esg(imgs, y_enc, x_reduced_len, y_len_enc)

            print_target_source = False
            if i < 2:
                print_target_source = True
            print(f"y_len_enc: {y_len_enc}")
            self.train_task(imgs, y_enc, x_reduced_len, y_len_enc, print_target_source)

            if i == self.args.iter - 1:
                self.flush_partial_accumulation()

            # evaluation
            if self.step % self.args.eval_freq == 0:
                for cv in self.test_converts:
                    self.eval(cv)
                # Save the restart state before the more expensive GAN-image
                # evaluation so a scheduler timeout cannot delay checkpointing.
                self.save_training_checkpoint()
                self.save_networks_last()
                self.eval_GAN_img()

            losses = self.print_loss()
            if self.args.debug_pc == 0:
                if isinstance(losses, dict):
                    wandb.log(losses, step=self.step)

        # Also preserve a final state when --iter is not a multiple of
        # --eval_freq, or when a resumed job has already reached --iter.
        self.save_training_checkpoint()
        self.save_networks_last()

    def test(self):
        self.set_default()
        self.set_networks()
        self.load_networks(self.args.load_step)
        for cv in self.test_converts:
            self.eval(cv)
