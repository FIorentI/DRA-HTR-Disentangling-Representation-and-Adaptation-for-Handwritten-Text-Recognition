# DRA-HTR: Disentangling Representation and Adaptation for Handwritten Text Recognition

This repository provides the preprocessing, training, and evaluation code for **DRA-HTR**, an unsupervised domain adaptation pipeline for line-level handwritten text recognition (HTR), and **DRAL-HTR**, its extension with language-model post-correction.



### Key Features

- Visual adaptation from labeled source data to a target domain.
- State-of-the-art performance on Handwritten Text Recognition Domain Adaptation
- Increased computation time only during training, not inference
- Optional Qwen post-correction of recognized text.


## Method

The pipeline has three stages:

1. **Source pretraining:** train a CRNN on labeled source lines.
2. **Visual adaptation:** learn to generate source lines in the target style.
3. **Recognizer training:** train the HTR model with the generated lines and their source transcriptions.

DRAL-HTR adds Qwen post-correction after recognition. 

## Citation

If you use this work, please cite the [arXiv preprint](https://arxiv.org/abs/2611.18143):

```bibtex
@misc{pathirage2026drahtr,
  title={{DRA-HTR}: Disentangling Representation and Adaptation for Handwritten Text Recognition},
  author={Pathirage, Gayan H. and Imbert, Florent and Coquenet, Denis and Barney Smith, Elisa H.},
  year={2026},
  eprint={2611.18143},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.18143}
}
```

## Installation

The reference environment uses Python 3.11.8, PyTorch 2.6.0, torchvision 0.21.0, and CUDA 12.4. Install the corresponding PyTorch wheels, then the project dependencies:

```bash
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

`requirements.txt` pins the project dependencies from the reference environment; it is not a complete environment lock. It includes Transformers, accelerate, safetensors, and bitsandbytes for local Qwen inference and 4-bit loading.

The examples below use greedy CTC decoding, which does not require `pyctcdecode`. Optional beam decoding needs a separately resolved dependency set: `pyctcdecode==0.5.0` declares `numpy<2`, whereas the reference environment uses NumPy 2.2.3. This optional package is therefore excluded from installation.

## Data

### Datasets and LMDB format

The evaluation scripts use the following dataset folder names:

| Collection        | Folder name |
|-------------------| --- |
| IAM               | `IAM` |
| Bentham           | `Bentham` |
| READ 2016         | `READ_2016` |
| NorHand 72702     | `NorHandV3_72702` |
| NorHand 331861    | `NorHandV3_331861` |
| Riksarkivet BK 1  | `Riksarkivet_Bergskollegium` |
| Riksarkivet GBG 2 | `Riksarkivet_Goteborgs2` |

Prepare line images and their transcriptions as LMDB datasets. The [LMDB Creator notebook](https://github.com/gayan68/ToolBox/blob/main/LMDB_Creator.ipynb) provides a conversion workflow.

The loader, `src/data/datasets/htr_dataset_lmdb.py`, expects this layout for each dataset:

```text
DATASETS/
└── IAM/
    └── polygon_lines_dataset1/
        ├── charset2.txt
        ├── train/
        │   └── lmdb/
        ├── val/
        │   └── lmdb/
        └── test/
            └── lmdb/
```

Each LMDB contains encoded line images under `image-<id>` and UTF-8 transcriptions under matching `label-<id>` keys. The current loader requires both keys for every sample, including target-domain samples.

`charset2.txt` contains one character per line, including a newline after the final character. Use the same alphabet and character order for the source, target, and corresponding checkpoints. Domain-adaptation training reads the source dataset's charset; evaluation reads the evaluated dataset's charset. `data/charset2.txt` is provided as a reference alphabet.

### Label preprocessing

`src/data_pre-processing.py` applies predefined character replacements and removals. Set `lmdb_root` in the script to the dataset directory and review its normalization rules before running:

```bash
python src/data_pre-processing.py
```

This script updates the train, validation, and test LMDB labels **in place** and writes character counts to `char_counter2.json`. Image entries are unchanged. Charset export is currently commented out, so provide `charset2.txt` separately.

## Usage

Run all commands from the repository root. Dataset, checkpoint, and output paths are resolved from the current working directory. Replace the example paths with your own locations.

### Step 1: Pretrain the CRNN on source data

The following example trains an IAM baseline. The first positional argument is the output **directory**, where checkpoints such as `crnn_best.torch` are saved:

```bash
python -u src/train/supervised/train_crnn.py \
    "../../logs/DomainAdaptation/CRNN/CRNN-041_run1" \
    --dataset_path "../../DATASETS/IAM/polygon_lines_dataset1" \
    --charset "charset2.txt" \
    --batch_size 32 \
    --num_workers 4 \
    --nb_epochs_max 1500 \
    --lr_drop_patience 100 \
    --early_stop_patience 200 \
    --use_regularization 0 \
    --epoch_start_regularization 20000 \
    --height_max 160 \
    --width_max 1570 \
    --pad_left 64 \
    --pad_right 64 \
    --model_type "crnn"
```

### Steps 2 and 3: Domain adaptation and recognizer training

Adapt from IAM to Bentham using the source CRNN checkpoint from step 1. Supply the pretrained ConvNeXtV2 weights through `--convnextv2_model_path`. The `--path_log` argument is an output directory; adaptation checkpoints are saved in its `checkpoint/` subdirectory.

```bash
python -u src/train/domain_adaptation/train_DRA-HTR.py \
    -T clf \
    --dataset_root "../../DATASETS" \
    --dataset_source "IAM" \
    --dataset_target "Bentham" \
    --dataset_dir_subdir "polygon_lines_dataset1" \
    --charset "charset2.txt" \
    --batch 4 \
    --accum_steps 8 \
    --lr_dra 0.0002 \
    --lr_clf 0.0004 \
    --lr_seg 0.0006 \
    --clip_grad_dra 1.0 \
    --clip_grad_task 1.0 \
    --lambda_g 0.1 \
    --lambda_content 0.0002 \
    --lambda_style 1.0 \
    --lambda_consis 0.12 \
    --lambda_recon 0.2 \
    --lambda_task 0.5 \
    --T_ESG_loss_start 130000 \
    --recon_foreground_weight 2.0 \
    --recon_foreground_mode bright \
    --recon_edge_weight 0.5 \
    --generator_output_activation sigmoid \
    --discriminator PatchGAN \
    --slice_patches 1 \
    --height_max 160 \
    --width_max 1570 \
    --pad_left 64 \
    --pad_right 64 \
    --eval_freq 500 \
    --T_training_start 100000 \
    --T_both_directions 0 \
    --dra_arch style_film \
    --E_net convnextv2_tiny \
    --convnextv2_model_path "../../logs/DomainAdaptation/pretrain_models/convnextv2_tiny_22k_384_ema.pt" \
    --path_log "../../logs/DomainAdaptation/DA_Convnext/run1/DA-101_run1/" \
    --path_model "../../logs/DomainAdaptation/CRNN/CRNN-041_run1/crnn_best.torch" \
    --debug_pc 1
```

`--T_both_directions 0` restricts recognizer training to the source-to-target direction. The default value, `1`, also trains a reverse recognizer using target transcriptions. Target test transcriptions are used for periodic CER evaluation and selection of the best checkpoints, including in the example above. `--debug_pc 1` disables Weights & Biases logging.

### Evaluation

The evaluators select datasets and checkpoints from their experiment tables. Review these tables before running: the baseline and DRA-HTR scripts construct paths under `../../DATASETS` and `../../logs/DomainAdaptation`, and `--run_name` selects the checkpoint run. The correction script also accepts `--dataset_root`, `--dataset_subdir`, and `--experiment_ids`.

#### CRNN baseline

Evaluate the CRNN baseline checkpoints from `run1`:

```bash
python -u src/evaluate/supervised/eval_7_dataset-CRNN-baseline.py \
    --results_log ../../logs/DomainAdaptation/test.txt \
    --run_name run1
```

#### DRA-HTR

Evaluate the domain-adapted recognizer checkpoints from `run1`:

```bash
python -u src/evaluate/supervised/eval_7_dataset-DRA-HTR.py \
    --results_log ../../logs/DomainAdaptation/test.txt \
    --run_name run1
```

These two commands iterate over all configured experiments and require the corresponding checkpoints. The training example above produces the IAM-to-Bentham checkpoint for experiment `101`; update the tables if evaluating only that experiment.

#### DRAL-HTR: Qwen post-correction

Evaluate experiment `101` (IAM to Bentham) from `run1` with greedy CTC decoding and local Qwen correction. Set `CUDA_VISIBLE_DEVICES` to the GPU you want to use; `cuda:0` refers to the first visible GPU.

```bash
python -u src/evaluate/supervised/eval_7_dataset-DRA-HTRL.py \
    --qwen_model "Qwen/Qwen3.8-27B" \
    --qwen_load_in_4bit \
    --qwen_device cuda:0 \
    --qwen_max_new_tokens 16384 \
    --qwen_temperature 0.0 \
    --results_log ../../logs/DomainAdaptation/test.txt \
    --qwen_cache_path "" \
    --ctc_decoding greedy \
    --context_lines 0 \
    --run_name run1 \
    --predictions_dir ../../logs/DomainAdaptation/QwenV3_predictions/DA-101_run1 \
    --print_qwen_debug 1 \
    --experiment_ids 101
```

## Models used in this work

### CRNN

The recognizer builds on [HTR best practices](https://github.com/georgeretsi/HTR-best-practices/). Its supervised training entry point is `src/train/supervised/train_crnn.py`.

```bibtex
@inproceedings{retsinas2022best,
  title={Best practices for a handwritten text recognition system},
  author={Retsinas, George and Sfikas, Giorgos and Gatos, Basilis and Nikou, Christophoros},
  booktitle={International Workshop on Document Analysis Systems},
  pages={247--259},
  year={2022},
  organization={Springer}
}
```

### DRANet

The adaptation framework builds on [DRANet-pytorch](https://github.com/FIorentI/DRANet-pytorch) and the earlier [domain adaptation pipeline for character classification and HTR](https://github.com/simon-corbi/DRANet-for-classification-and-htr).

```bibtex
@misc{lee2021dranetdisentanglingrepresentationadaptation,
  title={{DRANet}: Disentangling Representation and Adaptation Networks for Unsupervised Cross-Domain Adaptation},
  author={Seunghun Lee and Sunghyun Cho and Sunghoon Im},
  year={2021},
  eprint={2103.13447},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2103.13447}
}
```
