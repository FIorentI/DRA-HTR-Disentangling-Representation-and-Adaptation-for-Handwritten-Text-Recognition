import argparse
import faulthandler
import os
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.collate.collate_batch import CollateImageLabelHTR
from src.data.datasets.htr_dataset_lmdb import HTRDataset
from src.data.global_values.text_global_values import CTC_PAD, BLANK_STR_TOKEN
from src.data.image.augmentations.aug_htr import get_img_data_augmentation_htr
from src.data.readers.reader_config import read_json_config, read_labels_files, read_json_config_crnn_supervised
from src.data.text.charset_token import CharsetToken
from src.evaluate.metrics_counter import MetricLossCERWER
from src.evaluate.supervised.eval_one_epoch_crnn import evaluate_one_epoch_crnn
from src.models.crnn import CRNN
from src.models.losses.compute_center_cluster import compute_center_coordinates_crnn
from src.models.model_utils import load_pretrained_model, load_pretrained_model_shortcut, load_pretrained_model_encoder_and_shortcut
from src.train.supervised.train_crnn_one_epoch import train_crnn_one_epoch
import wandb

parser = argparse.ArgumentParser()

# parser.add_argument("config_file")
parser.add_argument("log_dir")
parser.add_argument("--dataset_path", default="", type=str)

parser.add_argument("--charset", default="", type=str)

parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--batch_size', default=4, type=int)  # To update
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--nb_epochs_max', default=2, type=int)  # To update
parser.add_argument("--path_model", default="", type=str)
parser.add_argument("--path_optimizer", default="", help="", type=str)
parser.add_argument('--height_max', default=160, type=int)
parser.add_argument('--width_max', default=1570, type=int)
parser.add_argument('--pad_left', default=64, type=int)
parser.add_argument('--pad_right', default=64, type=int)

parser.add_argument('--LR_drop', default=False, type=bool)
parser.add_argument('--lr_decay_1', default=10, type=float)
parser.add_argument('--lr_drop_patience', default=50, type=int)
parser.add_argument('--early_stop_patience', default=50, type=int)
parser.add_argument('--test_data_exist', default=1, type=int)

# Regularization
parser.add_argument('--use_regularization', default=1, type=int)
parser.add_argument('--epoch_start_regularization', default=1, type=int) # To update
parser.add_argument('--weight_loss_regularization_ok', default=0.55, type=float)

# Special Training
parser.add_argument("--model_type", default="crnn", type=str, choices=["crnn", "cnn"])
parser.add_argument("--weights_to_load", default="", help="", type=str)
parser.add_argument("--freeze_shortcut", default="No", type=str)
parser.add_argument("--freeze_encoder", default="No", type=str)

# Wandb
parser.add_argument('--wandb', default=False, type=bool)
parser.add_argument("--wandb_proj_name", default="", help="", type=str)
parser.add_argument("--wandb_name", default="", help="", type=str)
print("===============================================================================")

begin = time.time()
args = parser.parse_args()
print(args)

faulthandler.enable()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device :")
print(device)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
print("torch.cuda.device_count(): " + str(torch.cuda.device_count()))

os.makedirs(args.log_dir, exist_ok=True)
# Paths
# train_info, val_info, test_info, charsets_path, all_labels_files, _ = read_json_config_crnn_supervised(args.config_file)
# all_labels = read_labels_files(all_labels_files)
all_labels = None   # Comment if you use the htr_dataset.py

# Alphabet
charsets_path = os.path.join(args.dataset_path, args.charset)
print(f"Charset path: {charsets_path}")
charset = CharsetToken([charsets_path], use_blank=True)
char_list = charset.get_charset_list()
char_dict = charset.get_charset_dictionary()

# Model    
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
head_cfg = (256, 3)  # (hidden dimension, num_layers blstm)
width_divisor = 8

model_reco = CRNN(cnn_cfg, head_cfg, charset.get_nb_char(), args.model_type)

# Data
fixed_size_img = (args.height_max, args.width_max)
aug_transforms = get_img_data_augmentation_htr()

# Pad img with black = 0
c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
collate_fn = c_collate_fn.collate_fn

# Merge train db
directory_train = os.path.join(args.dataset_path, "train")
train_db = HTRDataset(directory_train,
                      fixed_size_img,
                      width_divisor,
                      "center",
                      args.pad_left,
                      args.pad_right,
                      all_labels,
                      char_dict,
                      aug_transforms,
                      apply_noise=1,
                      is_trainset=True)
print('Nb samples train {}:'.format(len(train_db)))
train_dataloader = DataLoader(train_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                              collate_fn=collate_fn, shuffle=True)



directory_val = os.path.join(args.dataset_path, "val")
val_db = HTRDataset(directory_val,
                    fixed_size_img,
                    width_divisor,
                    "center",
                    args.pad_left,
                    args.pad_right,
                    all_labels,
                    char_dict)
print('Nb samples val {}:'.format(len(val_db)))

val_dataloader = DataLoader(val_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                            collate_fn=collate_fn, shuffle=False)



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

### --- Initialize Wandb ---
print(f"self.args.wandb: {args.wandb}")

if args.wandb:
    if args.wandb_name == "":
        wandb.init(
            project=args.wandb_proj_name,
            config={
            "architecture": "CRNN",
            "head_type": args.model_type,
            }
        )
    else:
        wandb.init(
            project=args.wandb_proj_name,
            name=args.wandb_name,
            config={
            "architecture": "CRNN",
            "head_type": args.model_type,
            }
        )
    print("run name WANDB : " + str(wandb.run.name))   


# Init model
print("Initializing model weights kaiming")
for p in model_reco.parameters():
    if p.dim() > 1:
        torch.nn.init.kaiming_normal_(p, nonlinearity="relu")

if os.path.isfile(args.path_model):
    if args.model_type == "cnn":
        if args.weights_to_load == "shortcut":
            load_pretrained_model_shortcut(args, model_reco, device)
        else:
            load_pretrained_model(args, model_reco, device)
    elif args.model_type == "crnn":
        if args.weights_to_load == "EncoderAndShortcut":
            load_pretrained_model_encoder_and_shortcut(args, model_reco, device)
        else:
            load_pretrained_model(args, model_reco, device)

print("\nTrainable parameters:")
for name, param in model_reco.named_parameters():
    if param.requires_grad:
        print(name)


print(f"Transferring model to {str(device)}...")
model_reco = model_reco.to(device)

number_parameters = sum(p.numel() for p in model_reco.parameters() if p.requires_grad)
print(f"Model has {number_parameters:,} trainable parameters.")

print_summary = ""

ctc_loss_fn = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")

optimizer = torch.optim.Adam(model_reco.parameters(), lr=args.learning_rate)

if os.path.isfile(args.path_optimizer):
    try:
        checkpoint = torch.load(args.path_optimizer, map_location=device)
        optimizer.load_state_dict(checkpoint)
        print("Load optimizer")
    except:
        print("Error load optimizer")
        optimizer = torch.optim.Adam(model_reco.parameters(), lr=args.learning_rate)

best_cer = 1.0
best_epoch = 0

path_save_model_best = os.path.join(args.log_dir, "crnn_best.torch")
path_save_model_last = os.path.join(args.log_dir, "crnn_last.torch")

path_save_optimizer_best = os.path.join(args.log_dir, "optimizer_best.torch")
path_save_optimizer_last = os.path.join(args.log_dir, "optimizer_last.torch")

lr = args.learning_rate
# For center loss
index_class_to_filter = [char_dict[BLANK_STR_TOKEN], char_dict[" "]]
loss_reg = torch.nn.MSELoss(reduction="mean")

conf_reg = {
    "index_class_to_filter": index_class_to_filter,
    "loss_reg": loss_reg,
    "weight_loss_regularization_ok": args.weight_loss_regularization_ok
}


    

ceneters_value = []
compute_loss_reg = False

val_stable_count = 0
epoch2 = 0
LR_drop = args.LR_drop

print(f"LR_drop: {LR_drop}")

begin_train = time.time()
# Training
for epoch in range(0, args.nb_epochs_max):
    begin_time_epoch = time.time()
    print('EPOCH {}:'.format(epoch))

    if (val_stable_count >= args.lr_drop_patience) and (LR_drop == False):
        lr = args.learning_rate / args.lr_decay_1
        LR_drop = True
        val_stable_count = 0
        best_cer = 1.0
        print(f'LR Drop at: {epoch}')

    if LR_drop:
        epoch2 += 1

    # Learning rate values
    # if epoch < args.milestones_lr_1:
    #     lr = args.learning_rate
    # # decay
    # else:
    #     lr = args.learning_rate / args.lr_decay_2

    for g in optimizer.param_groups:
        g['lr'] = lr
    print("lr:" + str(lr))

    # Compute clusters coordinates if activate
    if args.use_regularization == 1 and LR_drop:
        if epoch2 >= args.epoch_start_regularization:
            print("Compute prototype")
            compute_loss_reg = True

            ceneters_value = compute_center_coordinates_crnn(args,
                                                        train_dataloader,
                                                        model_reco,
                                                        device,
                                                        char_list,
                                                        char_dict["<BLANK>"],
                                                        index_class_to_filter)
            
    # Training
    dict_losses = train_crnn_one_epoch(args,
                                       train_dataloader,
                                       optimizer,
                                       model_reco,
                                       device,
                                       ctc_loss_fn,
                                       compute_loss_reg,
                                       conf_reg,
                                       ceneters_value,
                                       char_list,
                                       char_dict[BLANK_STR_TOKEN])

    print('train_loss_main {}'.format(dict_losses["loss_main"]))

    if compute_loss_reg:
        print('train_loss_reg_epoch {}'.format(dict_losses["loss_reg_epoch"]))


    print("--------------Evaluate-------------------------------")
    # current_db_name = all_val_dataset[i_db].name_db
    # print(current_db_name)

    dict_result = evaluate_one_epoch_crnn(args,
                                            val_dataloader,
                                            model_reco,
                                            device,
                                            char_list,
                                            char_dict[BLANK_STR_TOKEN],
                                            ctc_loss_fn)

    dict_result["metrics_main"].print_cer_wer()
    if args.wandb :
        wandb.log({
            "training_loss": dict_losses["loss_main"],
            "validation_loss" : dict_result["metrics_main"].get_loss(),
            "Val_CER": dict_result["metrics_main"].get_cer(),
            "learning_rate": optimizer.param_groups[0]['lr']
            })

    # Save best model
    # -> to update if multiple validation datasets
    if dict_result["metrics_main"].get_cer() < best_cer:
        best_cer = dict_result["metrics_main"].get_cer()
        best_epoch = epoch
        val_stable_count = 0
        print("Best cer, save model.")

        torch.save(model_reco.state_dict(), path_save_model_best)
        torch.save(optimizer.state_dict(), path_save_optimizer_best)
    else:
        val_stable_count += 1

    if (val_stable_count >= args.early_stop_patience) and LR_drop :
        print('Early Stopping at: %d. Best epoch is: %d' % (epoch, best_epoch))
        torch.save(model_reco.state_dict(), path_save_model_last)
        torch.save(optimizer.state_dict(), path_save_optimizer_last)
        if args.wandb:
            wandb.log({"Early Stopping at": epoch, "Best Epoch": best_epoch })
            if args.test_data_exist != 1:
                wandb.finish()
        break



    end_time_epoch = time.time()
    print("Time one epoch (s): " + str((end_time_epoch - begin_time_epoch)))
    print("")

    torch.save(model_reco.state_dict(), path_save_model_last)
    torch.save(optimizer.state_dict(), path_save_optimizer_last)

end_train = time.time()
print("best_epoch: " + str(best_epoch))
print("best_cer val: " + str(best_cer))
print("Time all (s): " + str((end_train - begin_train)))
print("End training")

print_summary += "best_epoch: " + str(best_epoch) + "\n"
print_summary += "best_cer val: "
print_summary += f"{100 * best_cer:.2f}% \n"
print_summary += "\n"

metrics_test_last = MetricLossCERWER("All Tests last")
metrics_test_best = MetricLossCERWER("All Tests best")

if args.test_data_exist == 1:
    print("--------------Testing-------------------------------")
    # Load last model -> case multiple test datasets
    # if os.path.isfile(path_save_model_last):
    #     load_pretrained_model(args, model_reco, device, print_load_ok=False)

    print()
    # current_db_name = all_test_dataset[i_db].name_db
    # print(current_db_name)
    # print_summary += current_db_name
    # print_summary += " \n"

    print()
    print("--------Begin Testing last-----------")
    dict_result = evaluate_one_epoch_crnn(args,
                                            test_dataloader,
                                        model_reco,
                                        device,
                                        char_list,
                                        char_dict[BLANK_STR_TOKEN],
                                        ctc_loss_fn)

    dict_result["metrics_main"].print_cer_wer()

    metrics_test_last.add_metric(dict_result["metrics_main"])

    print_summary += "Testing last \n"
    str_cer_wer = dict_result["metrics_main"].str_cer_wer()
    print_summary += str_cer_wer + "\n"
    final_test_cer = dict_result["metrics_main"].get_cer()



    print("--------Begin Testing best cer val-----------")
    # Load best model
    if os.path.isfile(path_save_model_best):
        args.path_model = path_save_model_best
        load_pretrained_model(args, model_reco, device, print_load_ok=False)

    dict_result = evaluate_one_epoch_crnn(args,
                                        val_dataloader,
                                        model_reco,
                                        device,
                                        char_list,
                                        char_dict[BLANK_STR_TOKEN],
                                        ctc_loss_fn)

    dict_result["metrics_main"].print_cer_wer()
    metrics_test_best.add_metric(dict_result["metrics_main"])

    print_summary += "\n"
    print_summary += "Testing best val cer \n"
    str_cer_wer = dict_result["metrics_main"].str_cer_wer()
    print_summary += str_cer_wer + "\n"
    final_val_cer = dict_result["metrics_main"].get_cer()

if args.wandb :
    wandb.log({
        "Final_Val_CER": final_val_cer,
        "Final_Test_CER": final_test_cer
        })
    wandb.finish()
print()
# for all tests -> norm by all characters
metrics_test_last.print_cer()
metrics_test_best.print_cer()

print()
print("Summary:")
print(print_summary)

