import argparse
import faulthandler
import os
import time
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate.collate_batch import CollateImageLabelHTR
from src.data.datasets.htr_dataset_lmdb import HTRDataset
from src.data.global_values.text_global_values import CTC_PAD, BLANK_STR_TOKEN
from src.data.image.augmentations.aug_htr import get_img_data_augmentation_htr
from src.data.text.charset_token import CharsetToken
from src.evaluate.metrics_counter import MetricLossCERWER
from src.evaluate.supervised.eval_one_epoch_crnn import evaluate_one_epoch_crnn
from src.models.crnn import CRNN
from src.models.losses.compute_center_cluster import compute_center_coordinates_crnn
from src.models.model_utils import load_pretrained_model, load_pretrained_model_shortcut, load_pretrained_model_encoder_and_shortcut

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path", default="", type=str)
parser.add_argument("--charset", default="", type=str)

parser.add_argument('--batch_size', default=32, type=int)  # To update
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument("--path_model", default="", type=str)
parser.add_argument('--height_max', default=160, type=int)
parser.add_argument('--width_max', default=1570, type=int)
parser.add_argument('--pad_left', default=64, type=int)
parser.add_argument('--pad_right', default=64, type=int)
parser.add_argument("--model_type", default="crnn", type=str, choices=["crnn", "cnn"])
parser.add_argument("--freeze_shortcut", default="No", type=str)
parser.add_argument("--freeze_encoder", default="No", type=str)

parser.add_argument('--test_data_exist', default=1, type=int)
parser.add_argument("--run_name", default="run1", type=str,
                    help="Run directory to evaluate (for example: run2).")
parser.add_argument("--results_log",  default="../../logs/DomainAdaptation/results_CRNN_Baseline.txt",  type=str)


# Regularization


print("===============================================================================")

def evaluate_test(args, device, char_list, char_dict, ctc_loss_fn):
    # Model
    cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
    head_cfg = (256, 3)  # (hidden dimension, num_layers blstm)
    width_divisor = 8

    model_reco = CRNN(cnn_cfg, head_cfg, charset.get_nb_char(), args.model_type)

    # Init model
    print("Initializing model weights kaiming")
    for p in model_reco.parameters():
        if p.dim() > 1:
            torch.nn.init.kaiming_normal_(p, nonlinearity="relu")

    if os.path.isfile(args.path_model):
        load_pretrained_model(args, model_reco, device)


    print(f"Transferring model to {str(device)}...")
    model_reco = model_reco.to(device)
    
    # Separate test db
    directory_test = os.path.join(args.dataset_path, "test")
    test_db = HTRDataset(directory_test,
                            fixed_size_img,
                            width_divisor,
                            "center",
                            args.pad_left,
                            args.pad_right,
                            all_labels,
                            char_dict)

    print('Nb samples test {}:'.format(len(test_db)))

    test_dataloader = DataLoader(test_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                    collate_fn=collate_fn, shuffle=False)
    
    dict_result = evaluate_one_epoch_crnn(args,
                                      test_dataloader,
                                    model_reco,
                                    device,
                                    char_list,
                                    char_dict[BLANK_STR_TOKEN],
                                    ctc_loss_fn)

    out = dict_result["metrics_main"].get_cer()
    return out



trained_on = {
    "IAM": {
        "trained_tag" : "041",
        "crnn_model": "CRNN/CRNN-041"
    },
    "Bentham": {
        "trained_tag" : "042",
        "crnn_model": "CRNN/CRNN-042"
    },
    "READ_2016": {
        "trained_tag" : "033",
        "crnn_model": "CRNN/CRNN-033"
    },
    "NorHandV3_72702": {
        "trained_tag" : "044",
        "crnn_model": "CRNN/CRNN-044"
    },
    "NorHandV3_331861": {
        "trained_tag" : "043",
        "crnn_model": "CRNN/CRNN-043"
    },
    "Riksarkivet_Bergskollegium": {
        "trained_tag" : "045",
        "crnn_model": "CRNN/CRNN-045"
    },
    "Riksarkivet_Goteborgs2": {
        "trained_tag" : "046",
        "crnn_model": "CRNN/CRNN-046"
    },
}

validate_on = {
    "IAM": {
        "dataset_sufix": "IAM",
        "dataset_folder": "../../DATASETS/IAM/polygon_lines_dataset1",
        "charset_file": "../../DATASETS/IAM/polygon_lines_dataset1/charset2.txt",
    },
    "Bentham": {
        "dataset_sufix": "Bentham",
        "dataset_folder": "../../DATASETS/Bentham/polygon_lines_dataset1",
        "charset_file": "../../DATASETS/Bentham/polygon_lines_dataset1/charset2.txt",
    },
    "READ_2016": {
        "dataset_sufix": "READ_2016",
        "dataset_folder": "../../DATASETS/READ_2016/polygon_lines_dataset1",
        "charset_file": "../../DATASETS/READ_2016/polygon_lines_dataset1/charset2.txt",
    },
    "NorHandV3_72702": {
        "dataset_sufix": "NorHandV3_72702",
        "dataset_folder": "../../DATASETS/NorHandV3_72702/polygon_lines_dataset1",
        "charset_file": "../../DATASETS/NorHandV3_72702/polygon_lines_dataset1/charset2.txt",
    },
    "NorHandV3_331861": {
        "dataset_sufix": "NorHandV3_331861",
        "dataset_folder": "../../DATASETS/NorHandV3_331861/polygon_lines_dataset1",
        "charset_file": "../../DATASETS/NorHandV3_331861/polygon_lines_dataset1/charset2.txt",
    },
    "Riksarkivet_Bergskollegium": {
        "dataset_sufix": "Riksarkivet_Bergskollegium",
        "dataset_folder": "../../DATASETS/Riksarkivet_Bergskollegium/polygon_lines_dataset1",
        "charset_file": "../../DATASETS/Riksarkivet_Bergskollegium/polygon_lines_dataset1/charset2.txt",
    },
    "Riksarkivet_Goteborgs2": {
        "dataset_sufix": "Riksarkivet_Goteborgs2",
        "dataset_folder": "../../DATASETS/Riksarkivet_Goteborgs2/polygon_lines_dataset1",
        "charset_file": "../../DATASETS/Riksarkivet_Goteborgs2/polygon_lines_dataset1/charset2.txt",
    },
}

if __name__ == '__main__':

    args = parser.parse_args()
    faulthandler.enable()

    results_log = args.results_log
    args.charset = "charset2.txt"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device :")
    print(device)
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    print("torch.cuda.device_count(): " + str(torch.cuda.device_count()))

    # Paths
    # train_info, val_info, test_info, charsets_path, all_labels_files, _ = read_json_config_crnn_supervised(args.config_file)
    # all_labels = read_labels_files(all_labels_files)
    all_labels = None   # Comment if you use the htr_dataset.py

    # Data
    fixed_size_img = (args.height_max, args.width_max)

    # Pad img with black = 0
    c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
    collate_fn = c_collate_fn.collate_fn

    ctc_loss_fn = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")


    for key_train, value_train in trained_on.items():
        print(f"CRNN model Trained on: {key_train}")
        args.path_model = f"../../logs/DomainAdaptation/{value_train['crnn_model']}_{args.run_name}/crnn_best.torch"

        for key_validate, value_validate in validate_on.items():
            print(f"Inference dataset: {key_validate}")
            args.dataset_path = value_validate['dataset_folder']

            # Alphabet
            charsets_path = os.path.join(args.dataset_path, args.charset)
            charset = CharsetToken([charsets_path], use_blank=True)
            char_list = charset.get_charset_list()
            char_dict = charset.get_charset_dictionary()
            

            out_cer = evaluate_test(args, device, char_list, char_dict, ctc_loss_fn)
            out_cer_presentage = f"{100 * out_cer:.2f}"
            print(out_cer_presentage)


            with open(results_log, "a") as f:
                f.write(f"{key_train};{value_train['trained_tag']};{value_validate['dataset_sufix']};{out_cer_presentage}\n")


