import os
import torch


def load_pretrained_model(args, model, device, print_load_ok=True):
    pretrained_model_file = args.path_model
    print("Loading pretrained model (from provided location: " + pretrained_model_file + ")...")
    if os.path.isfile(pretrained_model_file):
        pretrained_dict = torch.load(pretrained_model_file, map_location=device)

        model_dict = model.state_dict()

        pretrained_keys = []
        skipped_keys = []
        scratch_keys = []
        weights_to_freeze = []
        for k in model_dict.keys():
            key = k

            if key in pretrained_dict:
                if model_dict[k].shape == pretrained_dict[key].shape:
                    pretrained_keys.append(k)
                    if "top.cnn" in key and args.freeze_shortcut == "Yes":
                        weights_to_freeze.append(k)
                    if "features.features" in key and args.freeze_encoder == "Yes":
                        weights_to_freeze.append(k)
                else:
                    skipped_keys.append(k)
            else:
                scratch_keys.append(k)

        if print_load_ok:
            print('-' * 80)
            print("Loading following pretrained weights:")
        for k in pretrained_keys:
            key = k
            if print_load_ok:
                print(k)
            model_dict[k] = pretrained_dict[key]

        print('-' * 80)
        print("Freezing loaded pre-trained weights:")
        for name, param in model.named_parameters():
            if name in weights_to_freeze:
                param.requires_grad = False
                print(name)

        print('-' * 80)
        print("Training following weights from scratch:")
        for k in scratch_keys:
            print(k)

        print('-' * 80)
        print("Skipping following pretrained weights, because shapes mismatch:")
        for k in skipped_keys:
            key = k
            print(k)
            print(f"Model shape: '{model_dict[k].shape}'")
            print(f"Pretrained model shape: '{pretrained_dict[key].shape}'")

        model.load_state_dict(model_dict)

        print('-' * 80)
        print("Pretrained weights loaded.")

    else:
        print("Cannot load pretrained model from provided location: " + pretrained_model_file + " ...")

    model.to(device)

def load_pretrained_model_shortcut(args, model, device, print_load_ok=True):
    pretrained_model_file = args.path_model
    print("### load_pretrained_model_shortcut ###")
    print("Loading pretrained model only for the SHORTCUT (from provided location: " + pretrained_model_file + ")...")
    if os.path.isfile(pretrained_model_file):
        pretrained_dict = torch.load(pretrained_model_file, map_location=device)

        model_dict = model.state_dict()

        pretrained_keys = []
        skipped_keys = []
        scratch_keys = []
        weights_to_freeze = []
        for k in model_dict.keys():
            key = k

            if "top.cnn" in key:
                if key in pretrained_dict:
                    if model_dict[k].shape == pretrained_dict[key].shape:
                        # print(f"## Loaded Wights for layer: {key}")
                        pretrained_keys.append(k)
                        if args.freeze_shortcut == "Yes":
                            weights_to_freeze.append(k)
                    else:
                        skipped_keys.append(k)
            else:
                scratch_keys.append(k)

        if print_load_ok:
            print('-' * 80)
            print("Loading following pretrained weights:")
        for k in pretrained_keys:
            key = k
            if print_load_ok:
                print(k)
            model_dict[k] = pretrained_dict[key]

        print('-' * 80)
        print("Freezing loaded pre-trained weights:")
        for name, param in model.named_parameters():
            if name in weights_to_freeze:
                param.requires_grad = False
                print(name)

        print('-' * 80)
        print("Training following weights from scratch:")
        for k in scratch_keys:
            print(k)

        print('-' * 80)
        print("Skipping following pretrained weights, because shapes mismatch:")
        for k in skipped_keys:
            key = k
            print(k)
            print(f"Model shape: '{model_dict[k].shape}'")
            print(f"Pretrained model shape: '{pretrained_dict[key].shape}'")

        model.load_state_dict(model_dict)

        print('-' * 80)
        print("Pretrained weights loaded.")

    else:
        print("Cannot load pretrained model from provided location: " + pretrained_model_file + " ...")

    model.to(device)


def load_pretrained_model_encoder_and_shortcut(args, model, device, print_load_ok=True):
    pretrained_model_file = args.path_model
    print("Loading pretrained model for both ENCODER and SHORTCUT (from provided location: " + pretrained_model_file + ")...")
    if os.path.isfile(pretrained_model_file):
        checkpoint = torch.load(pretrained_model_file, map_location=device)

        pretrained_dict = checkpoint
        model_dict = model.state_dict()

        pretrained_keys = []
        skipped_keys = []
        scratch_keys = []
        weights_to_freeze = []
        for k in model_dict.keys():
            key = k

            if "top.cnn" in key or "features.features" in key: #top.cnn_top
                if key in pretrained_dict:
                    if model_dict[k].shape == pretrained_dict[key].shape:
                        # print(f"## Loaded Wights for layer: {key}")
                        pretrained_keys.append(k)
                        if "top.cnn" in key and args.freeze_shortcut == "Yes":
                            weights_to_freeze.append(k)
                        if "features.features" in key and args.freeze_encoder == "Yes":
                            weights_to_freeze.append(k)
                    else:
                        skipped_keys.append(k)
            else:
                scratch_keys.append(k)

        if print_load_ok:
            print('-' * 80)
            print("Loading following pretrained weights:")
        for k in pretrained_keys:
            key = k
            if print_load_ok:
                print(k)
            model_dict[k] = pretrained_dict[key]

        print('-' * 80)
        print("Freezing loaded pre-trained weights:")
        for name, param in model.named_parameters():
            if name in weights_to_freeze:
                param.requires_grad = False
                print(name)

        print('-' * 80)
        print("Training following weights from scratch:")
        for k in scratch_keys:
            print(k)

        print('-' * 80)
        print("Skipping following pretrained weights, because shapes mismatch:")
        for k in skipped_keys:
            key = k
            print(k)
            print(f"Model shape: '{model_dict[k].shape}'")
            print(f"Pretrained model shape: '{pretrained_dict[key].shape}'")

        model.load_state_dict(model_dict)

        print('-' * 80)
        print("Pretrained weights loaded.")

    else:
        print("Cannot load pretrained model from provided location: " + pretrained_model_file + " ...")

    model.to(device)


def load_pretrained_model_LSTM_swap(pretrained_model_file_cnn, pretrained_model_file_lstm, model, device, print_load_ok=True):
    print("Loading pretrained model (from provided location: " + pretrained_model_file_cnn + ")...")
    if os.path.isfile(pretrained_model_file_cnn) and os.path.isfile(pretrained_model_file_lstm):
        checkpoint_cnn = torch.load(pretrained_model_file_cnn, map_location=device)
        checkpoint_lstm = torch.load(pretrained_model_file_lstm, map_location=device)

        pretrained_dict_cnn = checkpoint_cnn
        pretrained_dict_lstm = checkpoint_lstm
        model_dict = model.state_dict()

        pretrained_keys = []
        skipped_keys = []
        scratch_keys = []
        for k in model_dict.keys():
            key = k

            if key in pretrained_dict_cnn and ("features.features" in k ):
                print("CNN key: " + key)
                if model_dict[k].shape == pretrained_dict_cnn[key].shape:
                    pretrained_keys.append(k)
                else:
                    skipped_keys.append(k)
            elif key in pretrained_dict_lstm and ("top.rec" in k or "top.fnl" in k or "top.cnn" in k):
                print("LSTM key: " + key)
                if model_dict[k].shape == pretrained_dict_lstm[key].shape:
                    pretrained_keys.append(k)
                else:
                    skipped_keys.append(k)
            else:
                scratch_keys.append(k)


        if print_load_ok:
            print('-' * 80)
            print("Loading following pretrained weights:")
        for k in pretrained_keys:
            key = k
            if "top.rec" in k or "top.fnl" in k:
                model_dict[k] = pretrained_dict_lstm[key]
                if print_load_ok:
                    print(f"Loading LSTM weight: {k}")
            else:
                model_dict[k] = pretrained_dict_cnn[key]
                if print_load_ok:
                    print(f"Loading CNN weight: {k}")
            

        print('-' * 80)
        print("Training following weights from scratch:")
        for k in scratch_keys:
            print(k)

        print('-' * 80)
        print("Skipping following pretrained weights, because shapes mismatch:")
        for k in skipped_keys:
            key = k
            print(k)
            print(f"Model shape: '{model_dict[k].shape}'")
            print(f"Pretrained model shape: '{pretrained_dict_cnn[key].shape}'")

        model.load_state_dict(model_dict)

        print('-' * 80)
        print("Pretrained weights loaded.")

    else:
        print("Cannot load pretrained model from provided location: " + pretrained_model_file_cnn + " OR " + pretrained_model_file_lstm + " ...")

    model.to(device)


def load_pretrained_model_shortcut_swap(pretrained_model_file_cnn, pretrained_model_file_lstm, model, device, print_load_ok=True):
    print("Loading pretrained model (from provided location: " + pretrained_model_file_cnn + ")...")
    if os.path.isfile(pretrained_model_file_cnn) and os.path.isfile(pretrained_model_file_lstm):
        checkpoint_cnn = torch.load(pretrained_model_file_cnn, map_location=device)
        checkpoint_lstm = torch.load(pretrained_model_file_lstm, map_location=device)

        pretrained_dict_cnn = checkpoint_cnn
        pretrained_dict_lstm = checkpoint_lstm
        model_dict = model.state_dict()

        pretrained_keys = []
        skipped_keys = []
        scratch_keys = []
        for k in model_dict.keys():
            key = k

            if key in pretrained_dict_cnn:
                if "features.features" in k:
                    if model_dict[k].shape == pretrained_dict_cnn[key].shape:
                        pretrained_keys.append(k)
                    else:
                        skipped_keys.append(k)
            else:
                scratch_keys.append(k)

            if key in pretrained_dict_lstm:
                if "top.cnn" in k:
                    if model_dict[k].shape == pretrained_dict_lstm[key].shape:
                        pretrained_keys.append(k)
                    else:
                        skipped_keys.append(k)
            else:
                scratch_keys.append(k)

        if print_load_ok:
            print('-' * 80)
            print("Loading following pretrained weights:")
        for k in pretrained_keys:
            key = k
            if "top.cnn" in k:
                model_dict[k] = pretrained_dict_lstm[key]
                if print_load_ok:
                    print(f"Loading LSTM weight: {k}")
            else:
                model_dict[k] = pretrained_dict_cnn[key]
                if print_load_ok:
                    print(f"Loading CNN weight: {k}")
            

        print('-' * 80)
        print("Training following weights from scratch:")
        for k in scratch_keys:
            print(k)

        print('-' * 80)
        print("Skipping following pretrained weights, because shapes mismatch:")
        for k in skipped_keys:
            key = k
            print(k)
            print(f"Model shape: '{model_dict[k].shape}'")
            print(f"Pretrained model shape: '{pretrained_dict_cnn[key].shape}'")

        model.load_state_dict(model_dict)

        print('-' * 80)
        print("Pretrained weights loaded.")

    else:
        print("Cannot load pretrained model from provided location: " + pretrained_model_file_cnn + " OR " + pretrained_model_file_lstm + " ...")

    model.to(device)


def load_pretrained_model_ConvNext(args, model, device, print_load_ok=True, pretrained_model_file=None):
    if pretrained_model_file is None:
        pretrained_model_file = args.convnextv2_model_path
    print("Loading pretrained model (from provided location: " + pretrained_model_file + ")...")
    if os.path.isfile(pretrained_model_file):
        pretrained_dict = torch.load(pretrained_model_file, map_location=device)["model"]
        model_dict = model.state_dict()

        pretrained_keys = []
        skipped_keys = []
        scratch_keys = []
        weights_to_freeze = []
        for k in model_dict.keys():
            key = k

            if key in pretrained_dict:
                if model_dict[k].shape == pretrained_dict[key].shape:
                    pretrained_keys.append(k)
                else:
                    skipped_keys.append(k)
            else:
                scratch_keys.append(k)

        if print_load_ok:
            print('-' * 80)
            print("Loading following pretrained weights:")
        for k in pretrained_keys:
            key = k
            if print_load_ok:
                print(k)
            model_dict[k] = pretrained_dict[key]

        print('-' * 80)
        print("Freezing loaded pre-trained weights:")
        for name, param in model.named_parameters():
            if name in weights_to_freeze:
                param.requires_grad = False
                print(name)

        print('-' * 80)
        print("Training following weights from scratch:")
        for k in scratch_keys:
            print(k)

        print('-' * 80)
        print("Skipping following pretrained weights, because shapes mismatch:")
        for k in skipped_keys:
            key = k
            print(k)
            print(f"Model shape: '{model_dict[k].shape}'")
            print(f"Pretrained model shape: '{pretrained_dict[key].shape}'")

        model.load_state_dict(model_dict)

        print('-' * 80)
        print("Pretrained weights loaded.")

    else:
        print("Cannot load pretrained model from provided location: " + pretrained_model_file + " ...")

    model.to(device)
