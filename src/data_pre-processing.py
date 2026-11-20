import lmdb
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json
from pathlib import Path


######################################################
# This script will in place update the LMDB dataset by replacing certain characters in the labels and removing unwanted characters.
# However, use the dataset/charset2.txt as it has the union of characters from all datasets and is used for training the model.
# Goodluck!
######################################################

def letter_replace(in_txt):
    replacements = {
        "Z": "z",
        "Q": "q",
        "X": "x",
        "ß": "ss",
        "Æ": "Ä",
        "æ": "ä",
        "ā": "a",
        "ē": "e",
        "ō": "o",
        "ū": "u",
        "ȳ": "y",
        "ÿ": "y",
        "Ø": "Ö",
        "ø": "ö",
        "ç": "c",
        "–": "-",
        "—": "-",
        "_": "-",
        "=": "-",
        "”": '"',
        "[": "(",
        "]": ")",
        "<": "(",
        ">": ")",
        "à": "a",
        "á": "a",
        "â": "a",
        "è": "e",
        "é": "e",
        "ê": "e",
        "÷": "/",
        "℔": "lb",
    }
    removals = [
        "¬",
        "§",
        "^-",
        "^..",
        "¾",
        "¼",
        "½",
        "°",
        "º",
        "⊥",
    ]
    
    # removals by ASCII/Unicode codes
    removal_codes = [
        772,
        776
    ]
    
    removals.extend(chr(c) for c in removal_codes)
    
    # Apply longer removal patterns first
    for old in sorted(removals, key=len, reverse=True):
        in_txt = in_txt.replace(old, "")

    for old, new in replacements.items():
        in_txt = in_txt.replace(old, new)

    return in_txt


def get_lmdb_size(path):
    env = lmdb.open(path, readonly=True, lock=False)
    try:
        info = env.info()
        stat = env.stat()

        page_size = stat["psize"]
        num_pages = info["last_pgno"] + 1  # pages are 0-indexed

        size_bytes = num_pages * page_size
        size_bytes = round(size_bytes*1.01)
        return size_bytes

    finally:
        env.close()

lmdb_root = f"../../DATASETS/READ_2016/polygon_lines_dataset_tmp"

lmdb_size = get_lmdb_size(f"{lmdb_root}/train/lmdb")
train_env = lmdb.open(f"{lmdb_root}/train/lmdb", readonly=False, lock=True, map_size=lmdb_size)
lmdb_size = get_lmdb_size(f"{lmdb_root}/val/lmdb")
val_env = lmdb.open(f"{lmdb_root}/val/lmdb", readonly=False, lock=True, map_size=lmdb_size)
lmdb_size = get_lmdb_size(f"{lmdb_root}/test/lmdb")
test_env = lmdb.open(f"{lmdb_root}/test/lmdb", readonly=False, lock=True, map_size=lmdb_size)


unique_characters = set()
max_img_width = 0
max_text_length = 0
char_counter = Counter()

keys = []
widths, heights = [], []
for env in [train_env, val_env, test_env]:
# for env in [test_env]:
    with env.begin(write=True) as txn:

        info = env.info()
        stat = env.stat()
        
        page_size = stat["psize"]
        num_pages = info["last_pgno"] + 1  # pages are 0-indexed
        
        current_size = num_pages * page_size
        print(f"Current LMDB size: {current_size / (1024**3):.2f} GB")

        cursor = txn.cursor()  # create a cursor to iterate

        for key, value in cursor:
            key_str = key.decode()  # convert bytes -> string
            if key_str.startswith("label-"):
                # print(key_str)
                # label = txn.get(key_str.encode()).decode()
                label = txn.get(key).decode("utf-8")
                new_label = letter_replace(label)
                if new_label != label:
                    print(f"{label} ->  {new_label}")
                unique_characters.update(set(new_label))
                char_counter.update(new_label)

                # write back to LMDB
                txn.put(key, new_label.encode("utf-8"))



# unique_char_path = f"{lmdb_root}/charset3.txt"
# with open(unique_char_path, 'w', encoding='utf-8') as f:
#     for ch in sorted(unique_characters):
#         f.write(ch + "\n")


save_path = f"{lmdb_root}/char_counter2.json"

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(char_counter, f, ensure_ascii=False, indent=2)
