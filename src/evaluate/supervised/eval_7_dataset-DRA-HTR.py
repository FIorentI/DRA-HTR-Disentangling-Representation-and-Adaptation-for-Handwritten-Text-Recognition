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
parser.add_argument("--run_name", default="run2", type=str,
                    help="Run directory to evaluate (for example: run2).")
parser.add_argument('--height_max', default=160, type=int)
parser.add_argument('--width_max', default=1570, type=int)
parser.add_argument("--padding_strategy", default="center", choices=["center", "random", "top_left"])
parser.add_argument('--pad_left', default=64, type=int)
parser.add_argument('--pad_right', default=64, type=int)
parser.add_argument("--model_type", default="crnn", type=str, choices=["crnn", "cnn"])
parser.add_argument("--freeze_shortcut", default="No", type=str)
parser.add_argument("--freeze_encoder", default="No", type=str)
parser.add_argument("--results_log",  default="../../logs/DomainAdaptation/results_DRA-HTR.txt",  type=str)

parser.add_argument('--test_data_exist', default=1, type=int)

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
                            args.padding_strategy,
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
    "101": {
        "DA_adaptation": "IAM-Bentham",
        "crnn_model": "net_best_T_IAM_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "102": {
        "DA_adaptation": "IAM-NorHandV3_72702",
        "crnn_model": "net_best_T_IAM_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "103": {
        "DA_adaptation": "IAM-NorHandV3_331861",
        "crnn_model": "net_best_T_IAM_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "104": {
        "DA_adaptation": "IAM-Riksarkivet_Bergskollegium",
        "crnn_model": "net_best_T_IAM_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "105": {
        "DA_adaptation": "IAM-Riksarkivet_Goteborgs2",
        "crnn_model": "net_best_T_IAM_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
    "106": {
        "DA_adaptation": "IAM-READ_2016",
        "crnn_model": "net_best_T_IAM_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "107": {
        "DA_adaptation": "Bentham-IAM",
        "crnn_model": "net_best_T_Bentham_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "108": {
        "DA_adaptation": "Bentham-NorHandV3_72702",
        "crnn_model": "net_best_T_Bentham_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "109": {
        "DA_adaptation": "Bentham-NorHandV3_331861",
        "crnn_model": "net_best_T_Bentham_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "110": {
        "DA_adaptation": "Bentham-Riksarkivet_Bergskollegium",
        "crnn_model": "net_best_T_Bentham_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "111": {
        "DA_adaptation": "Bentham-Riksarkivet_Goteborgs2",
        "crnn_model": "net_best_T_Bentham_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
    "112": {
        "DA_adaptation": "Bentham-READ_2016",
        "crnn_model": "net_best_T_Bentham_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "113": {
        "DA_adaptation": "NorHandV3_72702-IAM",
        "crnn_model": "net_best_T_NorHandV3_72702_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "114": {
        "DA_adaptation": "NorHandV3_72702-Bentham",
        "crnn_model": "net_best_T_NorHandV3_72702_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "115": {
        "DA_adaptation": "NorHandV3_72702-NorHandV3_331861",
        "crnn_model": "net_best_T_NorHandV3_72702_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "116": {
        "DA_adaptation": "NorHandV3_72702-Riksarkivet_Bergskollegium",
        "crnn_model": "net_best_T_NorHandV3_72702_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "117": {
        "DA_adaptation": "NorHandV3_72702-Riksarkivet_Goteborgs2",
        "crnn_model": "net_best_T_NorHandV3_72702_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
    "118": {
        "DA_adaptation": "NorHandV3_72702-READ_2016",
        "crnn_model": "net_best_T_NorHandV3_72702_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "119": {
        "DA_adaptation": "NorHandV3_331861-IAM",
        "crnn_model": "net_best_T_NorHandV3_331861_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "120": {
        "DA_adaptation": "NorHandV3_331861-Bentham",
        "crnn_model": "net_best_T_NorHandV3_331861_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "121": {
        "DA_adaptation": "NorHandV3_331861-NorHandV3_72702",
        "crnn_model": "net_best_T_NorHandV3_331861_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "122": {
        "DA_adaptation": "NorHandV3_331861-Riksarkivet_Bergskollegium",
        "crnn_model": "net_best_T_NorHandV3_331861_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "123": {
        "DA_adaptation": "NorHandV3_331861-Riksarkivet_Goteborgs2",
        "crnn_model": "net_best_T_NorHandV3_331861_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
    "124": {
        "DA_adaptation": "NorHandV3_331861-READ_2016",
        "crnn_model": "net_best_T_NorHandV3_331861_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "125": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-IAM",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "126": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-Bentham",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "127": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-NorHandV3_72702",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "128": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-NorHandV3_331861",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "129": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-Riksarkivet_Goteborgs2",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
    "130": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-READ_2016",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "131": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-IAM",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "132": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-Bentham",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "133": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-NorHandV3_72702",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "134": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-NorHandV3_331861",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "135": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-Riksarkivet_Bergskollegium",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "136": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-READ_2016",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "137": {
        "DA_adaptation": "READ_2016-IAM",
        "crnn_model": "net_best_T_READ_2016_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "138": {
        "DA_adaptation": "READ_2016-Bentham",
        "crnn_model": "net_best_T_READ_2016_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "139": {
        "DA_adaptation": "READ_2016-NorHandV3_72702",
        "crnn_model": "net_best_T_READ_2016_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "140": {
        "DA_adaptation": "READ_2016-NorHandV3_331861",
        "crnn_model": "net_best_T_READ_2016_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "141": {
        "DA_adaptation": "READ_2016-Riksarkivet_Bergskollegium",
        "crnn_model": "net_best_T_READ_2016_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "142": {
        "DA_adaptation": "READ_2016-Riksarkivet_Goteborgs2",
        "crnn_model": "net_best_T_READ_2016_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
}

if __name__ == '__main__':

    args = parser.parse_args()
    faulthandler.enable()
    args.charset = "charset2.txt"

    results_log = args.results_log

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


    for exp_id, value_train in trained_on.items():
        print(f"CRNN model Trained on: {exp_id}")
        experiment_name = f"DA-{exp_id}_{args.run_name}"
        args.path_model = os.path.join(
            "../../logs/DomainAdaptation/DA_Convnext",
            args.run_name,
            experiment_name,
            "checkpoint",
            value_train["crnn_model"],
        )

        args.dataset_path = f"../../DATASETS/{value_train['target_dataset']}/polygon_lines_dataset1"

        # Alphabet
        charsets_path = os.path.join(args.dataset_path, args.charset)
        charset = CharsetToken([charsets_path], use_blank=True)
        char_list = charset.get_charset_list()
        char_dict = charset.get_charset_dictionary()
        

        out_cer = evaluate_test(args, device, char_list, char_dict, ctc_loss_fn)
        out_cer_presentage = f"{100 * out_cer:.2f}"
        print(out_cer_presentage)


        with open(results_log, "a") as f:
            f.write(f"{exp_id};{value_train['DA_adaptation']};{out_cer_presentage}\n")


