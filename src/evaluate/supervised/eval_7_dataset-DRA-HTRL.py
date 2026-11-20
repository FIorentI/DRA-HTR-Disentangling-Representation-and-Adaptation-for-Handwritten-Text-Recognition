"""Evaluate ConvNext domain-adapted HTR models with Qwen post-correction (V3).

Uses the same HTR evidence and correction rules as QwenV2, but loads Qwen
in this Python process through Hugging Face Transformers; no Ollama server is
needed. --qwen_batch_size controls lines grouped in one prompt (default: 1),
following GeminiV2's independent-line batch format for larger batches.

Install the optional inference dependencies with:
    pip install -U "transformers>=4.57" accelerate safetensors
Optional 4-bit CUDA loading also requires bitsandbytes.
For Qwen/Qwen3.8-27B, use Transformers 5.21 (or a compatible newer version)
with a supported PyTorch build (5.21 requires PyTorch >=2.6). The model uses
the qwen3_5 architecture and AutoProcessor; Transformers 4.57 is insufficient.

Example (paths are relative to the working directory, as in QwenV2):
    python src/evaluate/supervised/eval_7_dataset-DRA-HTRL.py \
        --run_name run2 --qwen_batch_size 1 --experiment_ids 114

Only recognizer text is sent to Qwen. Target labels are used only for metrics;
optional few-shot examples come from source validation. Qwen loads lazily after
HTR inference and unloads after correction, avoiding simultaneous GPU residency
across experiments. Weights downloaded by Transformers remain cached on disk.

"""

import argparse
import difflib
import faulthandler
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import xml.etree.ElementTree as ET

# Allow direct execution from the repository root, for example:
# python -u src/evaluate/supervised/eval_7_dataset-DRA-HTRL.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import editdistance
import torch
from torch.utils.data import DataLoader

from src.data.collate.collate_batch import CollateImageLabelHTR
from src.data.datasets.htr_dataset_lmdb import HTRDataset
from src.data.global_values.text_global_values import CTC_PAD, BLANK_STR_TOKEN
from src.data.text.best_path_ctc import ctc_best_path_one
from src.data.text.charset_token import CharsetToken
from src.evaluate.evaluate_recognition import nb_chars_from_list
from src.models.crnn import CRNN
from src.models.model_utils import load_pretrained_model

# Number of CTC candidates shown to Qwen. Override with --nbest.
NBEST = 10

UNCERTAIN_OPEN = "⟨"
UNCERTAIN_CLOSE = "⟩"

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path", default="", type=str)
parser.add_argument("--charset", default="charset2.txt", type=str)
parser.add_argument("--dataset_root", default="../../DATASETS", type=str)
parser.add_argument("--dataset_subdir", default="polygon_lines_dataset1", type=str)

parser.add_argument('--batch_size', default=32, type=int)  # To update
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument("--path_model", default="", type=str)
parser.add_argument('--height_max', default=160, type=int)
parser.add_argument('--width_max', default=1570, type=int)
parser.add_argument("--padding_strategy", default="center", choices=["center", "random", "top_left"])
parser.add_argument('--pad_left', default=64, type=int)
parser.add_argument('--pad_right', default=64, type=int)
parser.add_argument("--model_type", default="crnn", type=str, choices=["crnn", "cnn"])
parser.add_argument("--freeze_shortcut", default="No", type=str)
parser.add_argument("--freeze_encoder", default="No", type=str)

parser.add_argument("--run_name", default="run1", type=str,
                    help="Run directory to evaluate (for example: run2).")

# N-best candidates
parser.add_argument(
    "--ctc_decoding",
    default="beam",
    choices=["greedy", "beam"],
    help="Use exact CTC greedy text or CTC prefix-beam candidates as Qwen input.",
)
parser.add_argument("--nbest", default=NBEST, type=int, help="Number of CTC candidates given to Qwen.")
parser.add_argument("--beam_width", default=50, type=int, help="Active beams in CTC prefix beam search.")

# Confidence marks
parser.add_argument(
    "--use_confidence_marks",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Mark low-confidence characters of the greedy reading as ⟨...⟩.",
)
parser.add_argument(
    "--confidence_threshold",
    default=0.6,
    type=float,
    help="Characters whose max CTC posterior is below this value are marked uncertain.",
)

# Neighbouring lines
parser.add_argument(
    "--context_lines",
    default=1,
    type=int,
    help="Recognized lines before and after the current line given as context (0 disables).",
)
parser.add_argument(
    "--page_xml_subdir",
    default="test/gt_xml",
    type=str,
    help="PAGE XML folder below {dataset_root}/{target_dataset}, used only for line reading order.",
)

# Few-shot error patterns from the source domain
parser.add_argument(
    "--use_fewshot",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Add recognizer error patterns measured on the source validation split (off by default).",
)
parser.add_argument("--fewshot_split", default="val", type=str)
parser.add_argument("--fewshot_examples", default=4, type=int, help="Number of example line pairs.")
parser.add_argument("--confusion_top_k", default=15, type=int, help="Number of frequent confusions listed.")
parser.add_argument("--fewshot_seed", default=0, type=int)

# Qwen inference in this process (independent of the HTR --batch_size).
parser.add_argument("--qwen_model", default="Qwen/Qwen3-VL-8B-Instruct", type=str,
                    help="Hugging Face model ID or local model directory, not an Ollama tag.")
parser.add_argument("--qwen_batch_size", default=1, type=int,
                    help="Independent lines grouped into one Qwen prompt (default: 1).")
parser.add_argument("--qwen_device", default="auto", type=str,
                    help="auto selects cuda:0 when available, otherwise cpu; also accepts cuda:N.")
parser.add_argument("--qwen_dtype", default="auto",
                    choices=["auto", "float32", "float16", "bfloat16"])
parser.add_argument("--qwen_load_in_4bit", action="store_true",
                    help="Load Qwen in 4-bit on CUDA; requires bitsandbytes and accelerate.")
parser.add_argument("--qwen_local_files_only", action="store_true",
                    help="Use only locally cached model/tokenizer files.")
parser.add_argument("--qwen_prompt", default="new", choices=["new", "new-no-CM", "extra-context", "detailed-restr"], type=str,
                    help="Qwen prompts, (default: %(default)s).")
parser.add_argument("--qwen_temperature", default=0.0, type=float)
parser.add_argument("--qwen_max_new_tokens", default=4096, type=int,
                    help="Output-token budget for a whole batch; raise for larger batches.")
parser.add_argument("--qwen_retries", default=2, type=int,
                    help="Additional attempts for malformed or incomplete correction JSON.")
parser.add_argument("--print_qwen_debug", default=0, type=int, choices=[0, 1],
                    help="Print raw Qwen responses before validation when set to 1 (default: 0).")
parser.add_argument("--qwen_cache_path",
                    default=str(PROJECT_ROOT / "outputs" / "qwen_v3_correction_cache.jsonl"),
                    help="Persistent correction cache; an empty string disables disk caching.")
parser.add_argument(
    "--target_language",
    "--target-language",
    default="auto",
    type=str,
    help=(
        "Language/domain hint for Qwen. 'auto' selects it from target_dataset "
        "for each experiment (default: %(default)s)."
    ),
)
llm_group = parser.add_mutually_exclusive_group()
llm_group.add_argument(
    "--llm_post_correction",
    "--llm-post-correction",
    dest="llm_post_correction",
    action="store_true",
    help="Enable Qwen post-correction (default).",
)
llm_group.add_argument(
    "--no_llm_post_correction",
    "--no-llm-post-correction",
    dest="llm_post_correction",
    action="store_false",
    help="Disable Qwen post-correction and report recognizer CERs only.",
)
parser.set_defaults(llm_post_correction=True)

parser.add_argument(
    "--experiment_ids",
    nargs="+",
    default=None,
    help="Optional experiment IDs to run, for example: 101 114 142. Default: all.",
)
parser.add_argument(
    "--results_log",
    default="../../logs/DomainAdaptation/results_DRA-HTRL.txt",
    type=str,
)
parser.add_argument(
    "--predictions_dir",
    default="../../logs/DomainAdaptation/QwenV3_predictions",
    type=str,
    help="Per-line JSONL dumps (ground truth, candidates, prompt inputs, Qwen output).",
)

parser.add_argument('--test_data_exist', default=1, type=int)


print("===============================================================================")


TEXT_CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {"corrected_text": {"type": "string"}},
    "required": ["corrected_text"],
    "additionalProperties": False,
}

BATCH_CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "corrected_text": {"type": "string"},
                },
                "required": ["sample_id", "corrected_text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["corrections"],
    "additionalProperties": False,
}

LINE_HEADER = "### LINE "

def batch_system_prompt(language_context, fewshot_block="", qwen_prompt="new"):
    """Qwen's per-line prompt, unchanged, preceded by the batch format.

    The batch preamble only explains the transport; every line is corrected
    with exactly the instructions and evidence that Qwen receives.
    """
    return (
        "You receive a batch of independent handwriting-recognition lines. Each line "
        f"starts with '{LINE_HEADER}<sample_id>' and is followed by its own input "
        "sections. Correct every line separately, exactly as if it were the only line "
        "in the request. Other lines of the batch are not context for it; use only its "
        "own PREVIOUS LINE / NEXT LINE sections. Return the corrections array with one "
        "item per input line, in input order, with the sample_id copied exactly and the "
        "corrected line in corrected_text. Never merge, split, skip, or reorder lines.\n\n"
        "Per-line instructions:\n"
        + text_correction_system_prompt(language_context, fewshot_block, qwen_prompt)
    )


def correction_system_prompt(language_context, fewshot_block, batch_size, qwen_prompt="new"):
    if batch_size == 1:
        base = text_correction_system_prompt(language_context, fewshot_block, qwen_prompt)
        schema = TEXT_CORRECTION_SCHEMA
    else:
        base = batch_system_prompt(language_context, fewshot_block, qwen_prompt)
        schema = BATCH_CORRECTION_SCHEMA
    return base + "\n\nReturn only valid JSON matching this schema, without Markdown fences:\n" + json.dumps(schema)


DATASET_LANGUAGE_CONTEXT = {
    "IAM": "modern English",
    "Bentham": "historical English from the 18th and 19th century",
    "NorHandV3_72702": "historical Norwegian from the 19th and early 20th century",
    "NorHandV3_331861": "historical Norwegian from the 19th and early 20th century",
    "Riksarkivet_Bergskollegium": "historical Swedish",
    "Riksarkivet_Goteborgs2": "historical Swedish",
    "READ_2016": "Early Modern German",
}


def text_correction_system_prompt(language_context, fewshot_block="", qwen_prompt="new"):
    prompt1 = (
        "You post-correct one line of handwriting-recognition (HTR) output. "
        f"The expected language and domain are {language_context}.\n"
        "The input may contain these sections:\n"
        "- CANDIDATES: the recognizer's best readings of the same line image, ranked by the "
        "recognizer's relative confidence. The correct text is often one of them, or a "
        "combination of parts of several of them, and it is frequently not the first one.\n"
        f"- UNCERTAIN: the recognizer's reading with low-confidence characters wrapped in "
        f"{UNCERTAIN_OPEN} {UNCERTAIN_CLOSE}. Characters outside the markers are usually correct; "
        "errors are concentrated inside them.\n"
        "- PREVIOUS LINE / NEXT LINE: recognized neighbouring lines of the same page. They may also "
        "contain errors. Use them only to understand the topic, names, and spelling conventions; "
        "never copy their text into the output.\n"
        "Follow these rules:\n"
        "1. Choose the candidate that is the most plausible text in the expected language, or "
        "combine parts of different candidates when each is correct in a different place. The "
        "ranking and percentages are only a weak hint: a lower-ranked candidate made of real words "
        "beats a higher-ranked one that is not.\n"
        "2. Go beyond the candidates only for plausible recognition errors, mainly in uncertain "
        "characters:\n"
        "   - character substitutions, insertions, or deletions\n"
        "   - incorrect word boundaries (splits or merges)\n"
        "   - punctuation recognition errors\n"
        "3. Use the expected language/domain only to choose between plausible readings of the "
        "recognized text.\n"
        "4. Preserve historical spelling, archaic forms, dialect, names, foreign words or phrases, "
        "abbreviations, numbers, and original meaning. Keep capitalization unless the candidates "
        "disagree on it or it is clearly a recognition error.\n"
        "5. Do not modernize spelling or grammar.\n"
        "6. Do not translate, paraphrase, explain, normalize, or stylistically improve the text.\n"
        "7. Do not add words based on what you think the surrounding context should say.\n"
        "8. Only when candidates are equally plausible, prefer the higher-ranked one.\n"
        "9. When going beyond the candidates, make the minimum number of edits necessary.\n"
        f"10. Return only the corrected line in the corrected_text field, without {UNCERTAIN_OPEN}"
        f"{UNCERTAIN_CLOSE} markers, numbering, or confidence values."
    )

    prompt2 = (
        "You post-correct one line of handwriting-recognition (HTR) output. "
        f"The expected language and domain are {language_context}.\n"
        "The input may contain these sections:\n"
        "- CANDIDATES: the recognizer's best readings of the same line image, ranked by the "
        "recognizer's relative confidence. The correct text is often one of them, or a "
        "combination of parts of several of them, and it is frequently not the first one.\n"
        "- PREVIOUS LINE / NEXT LINE: recognized neighbouring lines of the same page. They may also "
        "contain errors. Use them only to understand the topic, names, and spelling conventions; "
        "never copy their text into the output.\n"
        "Follow these rules:\n"
        "1. Choose the candidate that is the most plausible text in the expected language, or "
        "combine parts of different candidates when each is correct in a different place. The "
        "ranking and percentages are only a weak hint: a lower-ranked candidate made of real words "
        "beats a higher-ranked one that is not.\n"
        "2. Go beyond the candidates only for plausible recognition errors:\n"
        "   - character substitutions, insertions, or deletions\n"
        "   - incorrect word boundaries (splits or merges)\n"
        "   - punctuation recognition errors\n"
        "3. Use the expected language/domain only to choose between plausible readings of the "
        "recognized text.\n"
        "4. Preserve historical spelling, archaic forms, dialect, names, foreign words or phrases, "
        "abbreviations, numbers, and original meaning. Keep capitalization unless the candidates "
        "disagree on it or it is clearly a recognition error.\n"
        "5. Do not modernize spelling or grammar.\n"
        "6. Do not translate, paraphrase, explain, normalize, or stylistically improve the text.\n"
        "7. Do not add words based on what you think the surrounding context should say.\n"
        "8. Only when candidates are equally plausible, prefer the higher-ranked one.\n"
        "9. When going beyond the candidates, make the minimum number of edits necessary.\n"
        "10. Return only the corrected line in the corrected_text field, without numbering or confidence values."
    )

    prompt3 = (
        "You are a Post-OCR corrector. You correct mistakes in historic "
        "texts that are caused by errors in the Optical Character Recognition. "
        f"The text comes from {language_context} documents. "
        "Return exactly one corrected text line in the corrected_text JSON field."
    )

    prompt4 = (
        "You are a Post-OCR corrector. Your only task is to correct OCR-generated errors "
        "(misrecognized characters, broken words, missing or extra letters caused by scanning issues).\n"
        "Do NOT modernize spelling, fix grammar, rephrase sentences, or change punctuation if it reflects "
        "the original text style. \n"
        "Preserve original wording, spelling, capitalization style, and formatting as much as possible. \n"
        "Only correct errors that are clearly the result of OCR mistakes. \n"
        "Return exactly one corrected text line in the corrected_text JSON field."
    ) 
    
    prompt = {"new": prompt1, "new-no-CM": prompt2, "extra-context": prompt3, "detailed-restr": prompt4}[qwen_prompt]
    
    if fewshot_block:
        prompt += "\n\n" + fewshot_block
    return prompt


def language_context_for_dataset(dataset_name, configured_language):
    if configured_language.strip().lower() != "auto":
        return configured_language.strip()
    return DATASET_LANGUAGE_CONTEXT.get(dataset_name, "the language present in the input")


def normalize_text(text):
    # Match the normalization used by evaluate_one_epoch_crnn.
    return text.strip().replace("  ", " ")


def line_cer(truth, prediction):
    return editdistance.eval(truth, prediction) / max(1, len(truth))


def corpus_cer(ground_truths, predictions):
    total_characters = nb_chars_from_list(ground_truths)
    if total_characters == 0:
        return 1.0
    return sum(
        editdistance.eval(truth, prediction)
        for truth, prediction in zip(ground_truths, predictions)
    ) / total_characters


# ---------------------------------------------------------------------------
# Recognizer outputs: greedy text, per-character confidence, N-best candidates
# ---------------------------------------------------------------------------

def build_ctc_decoder(char_list, blank_index):
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError as exc:
        raise RuntimeError(
            "CTC N-best decoding requires pyctcdecode. Install the project requirements."
        ) from exc

    labels = char_list.copy()
    labels[blank_index] = ""
    # No KenLM model: Qwen is the only language model in this script.
    return build_ctcdecoder(labels)


def beam_text_and_score(beam):
    """Support both named OutputBeam objects and pyctcdecode 0.5 tuples."""
    if hasattr(beam, "text") and hasattr(beam, "logit_score"):
        return normalize_text(beam.text), float(beam.logit_score)
    return normalize_text(beam[0]), float(beam[3])


def decode_nbest(decoder, log_probs, beam_width, nbest):
    beams = decoder.decode_beams(log_probs, beam_width=beam_width, prune_history=False)
    unique_candidates = {}
    for beam in beams:
        text, ctc_score = beam_text_and_score(beam)
        if text not in unique_candidates or ctc_score > unique_candidates[text]:
            unique_candidates[text] = ctc_score

    candidates = sorted(unique_candidates.items(), key=lambda item: item[1], reverse=True)[:nbest]
    if not candidates:
        return []
    # Relative probability among the listed candidates.
    max_score = candidates[0][1]
    weights = [math.exp(score - max_score) for _, score in candidates]
    total = sum(weights)
    return [
        {"text": text, "ctc_score": score, "prob": weight / total}
        for (text, score), weight in zip(candidates, weights)
    ]


def greedy_chars_with_confidence(sample_log_probs, char_list, token_blank):
    """Collapse the greedy CTC path into (character, confidence) pairs.

    The confidence of a character is the maximum posterior over the frames of
    its run. The pairs follow the same normalization as normalize_text().
    """
    max_log_probs, path = torch.max(sample_log_probs, dim=1)
    probs = max_log_probs.exp().tolist()
    path = path.tolist()

    pairs = []
    previous_index = -1
    for index, prob in zip(path, probs):
        if index != previous_index:
            if index != token_blank:
                pairs.append([char_list[index], prob])
        elif index != token_blank:
            pairs[-1][1] = max(pairs[-1][1], prob)
        previous_index = index

    # Emulate str.strip() and a single non-overlapping replace("  ", " ").
    while pairs and pairs[0][0].isspace():
        pairs.pop(0)
    while pairs and pairs[-1][0].isspace():
        pairs.pop()
    normalized = []
    position = 0
    while position < len(pairs):
        if (
            pairs[position][0] == " "
            and position + 1 < len(pairs)
            and pairs[position + 1][0] == " "
        ):
            normalized.append([" ", min(pairs[position][1], pairs[position + 1][1])])
            position += 2
        else:
            normalized.append(pairs[position])
            position += 1
    return normalized


def mark_uncertain(char_confidences, threshold):
    """Wrap runs of characters below threshold in ⟨...⟩."""
    parts = []
    inside = False
    for char, confidence in char_confidences:
        uncertain = confidence < threshold
        if uncertain and not inside:
            parts.append(UNCERTAIN_OPEN)
        elif not uncertain and inside:
            parts.append(UNCERTAIN_CLOSE)
        inside = uncertain
        parts.append(char)
    if inside:
        parts.append(UNCERTAIN_CLOSE)
    return "".join(parts)


def run_recognizer(args, data_loader, model, device, char_list, token_blank, decoder=None):
    """Return one record per line. N-best candidates are added when decoder is given."""
    records = []
    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            x = batch_data["imgs"].to(device)
            x_reduced_len = batch_data["w_reduce"]

            predictions, _, _ = model(x)
            main_output, _ = predictions
            output_log = torch.nn.functional.log_softmax(main_output, dim=-1).transpose(0, 1)

            for index, sample_log_probs in enumerate(output_log):
                sample_log_probs = sample_log_probs[:int(x_reduced_len[index])].float().cpu()
                greedy_path = torch.argmax(sample_log_probs, dim=1).numpy()
                greedy_text = normalize_text(ctc_best_path_one(greedy_path, char_list, token_blank))
                record = {
                    "sample_id": batch_data["ids"][index],
                    "ground_truth": normalize_text(batch_data["label_str"][index]),
                    "greedy_text": greedy_text,
                }

                if decoder is not None:
                    char_confidences = greedy_chars_with_confidence(
                        sample_log_probs, char_list, token_blank
                    )
                    if "".join(char for char, _ in char_confidences) == greedy_text:
                        record["uncertain_text"] = mark_uncertain(
                            char_confidences, args.confidence_threshold
                        )
                    else:
                        record["uncertain_text"] = None

                    if args.ctc_decoding == "greedy":
                        candidates = [
                            {"text": greedy_text, "ctc_score": 0.0, "prob": 1.0}
                        ]
                    else:
                        candidates = decode_nbest(
                            decoder, sample_log_probs.numpy(), args.beam_width, args.nbest
                        )
                    if not candidates:
                        candidates = [{"text": greedy_text, "ctc_score": 0.0, "prob": 1.0}]
                    record["candidates"] = candidates

                records.append(record)

            print(
                f"Recognized {len(records)}/{len(data_loader.dataset)} samples",
                end="\r",
                flush=True,
            )
    print()
    return records


# ---------------------------------------------------------------------------
# Neighbouring lines
# ---------------------------------------------------------------------------

LINE_NUMBER_PATTERN = re.compile(r"^(.*?)(\d+)(\D*)$")
READING_ORDER_PATTERN = re.compile(r"readingOrder\s*\{\s*index\s*:\s*(\d+)")


def _local_tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _custom_reading_order(element, default):
    match = READING_ORDER_PATTERN.search(element.get("custom", ""))
    return int(match.group(1)) if match else default


PAGE_SIDE_SUFFIXES = {"_left": 0, "_right": 1}


def load_page_reading_order(page_xml_dir):
    """Map "{page}_{TextLine id}" to (page, order key) from PAGE XML layout.

    Only region/line reading order is read; the transcriptions are not used.
    Riksarkivet XML files are split into "{spread}_left.xml" and
    "{spread}_right.xml" while the LMDB IDs use "{spread}_{TextLine id}", so
    those lines are also registered under the spread, left side first.
    """
    line_order = {}
    if not os.path.isdir(page_xml_dir):
        return line_order
    for file_name in sorted(os.listdir(page_xml_dir)):
        if not file_name.lower().endswith(".xml"):
            continue
        page = os.path.splitext(file_name)[0]
        spread, side = page, 0
        for suffix, side_index in PAGE_SIDE_SUFFIXES.items():
            if page.endswith(suffix):
                spread, side = page[:-len(suffix)], side_index
        try:
            root = ET.parse(os.path.join(page_xml_dir, file_name)).getroot()
        except ET.ParseError as exc:
            print(f"WARNING: cannot parse {file_name}: {exc}")
            continue

        region_index = {
            element.get("regionRef"): int(element.get("index", 0))
            for element in root.iter()
            if _local_tag(element) == "RegionRefIndexed"
        }
        regions = [element for element in root.iter() if _local_tag(element) == "TextRegion"]
        for region_position, region in enumerate(regions):
            region_order = region_index.get(
                region.get("id"), _custom_reading_order(region, region_position)
            )
            lines = [element for element in region.iter() if _local_tag(element) == "TextLine"]
            for line_position, line in enumerate(lines):
                order_key = (
                    side, region_order, _custom_reading_order(line, line_position), line_position
                )
                line_order.setdefault(f"{page}_{line.get('id')}", (page, order_key))
                if spread != page:
                    line_order.setdefault(f"{spread}_{line.get('id')}", (spread, order_key))
    return line_order


def add_neighbour_context(records, context_lines, page_xml_dir=None):
    """Attach recognized previous/next lines of the same page.

    Line IDs are "{page}_{TextLine id}". The reading order comes from the PAGE
    XML files in page_xml_dir. IDs not found there fall back to grouping by
    everything except the last number in the ID, ordered by that number.
    """
    for record in records:
        record["previous_lines"] = []
        record["next_lines"] = []
    if context_lines <= 0:
        return

    line_order = load_page_reading_order(page_xml_dir) if page_xml_dir else {}
    pages = {}
    from_xml = 0
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in line_order:
            page, order_key = line_order[sample_id]
            pages.setdefault(("xml", page), []).append((order_key, record))
            from_xml += 1
            continue
        match = LINE_NUMBER_PATTERN.match(sample_id)
        if match is None:
            continue
        page_key = ("id", match.group(1), match.group(3))
        pages.setdefault(page_key, []).append(((int(match.group(2)),), record))
    print(
        f"Neighbour context: {from_xml}/{len(records)} lines ordered from PAGE XML "
        f"({page_xml_dir}), the rest by line-ID number."
    )

    for page_records in pages.values():
        page_records.sort(key=lambda item: item[0])
        ordered = [record for _, record in page_records]
        for position, record in enumerate(ordered):
            record["previous_lines"] = [
                other["candidates"][0]["text"]
                for other in ordered[max(0, position - context_lines):position]
            ]
            record["next_lines"] = [
                other["candidates"][0]["text"]
                for other in ordered[position + 1:position + 1 + context_lines]
            ]


# ---------------------------------------------------------------------------
# Few-shot error patterns from the labelled source validation split
# ---------------------------------------------------------------------------

def show_segment(segment):
    if segment == "":
        return "∅"
    return '"' + segment.replace(" ", "␣") + '"'


def build_fewshot_block(source_records, top_k, num_examples, seed):
    confusion_counts = {}
    for record in source_records:
        prediction, truth = record["greedy_text"], record["ground_truth"]
        matcher = difflib.SequenceMatcher(None, prediction, truth, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal" or (i2 - i1) > 3 or (j2 - j1) > 3:
                continue
            key = (prediction[i1:i2], truth[j1:j2])
            confusion_counts[key] = confusion_counts.get(key, 0) + 1

    lines = []
    frequent = sorted(confusion_counts.items(), key=lambda item: item[1], reverse=True)[:top_k]
    if frequent:
        lines.append(
            "Frequent errors of this recognizer (recognized → correct, count), measured on another "
            "collection whose language may differ. Learn the visual error patterns, not the "
            "vocabulary:"
        )
        for (recognized, correct), count in frequent:
            lines.append(f"- {show_segment(recognized)} → {show_segment(correct)} ({count})")

    example_pool = [
        record for record in source_records
        if len(record["ground_truth"]) >= 10
        and 0 < line_cer(record["ground_truth"], record["greedy_text"]) <= 0.3
    ]
    examples = random.Random(seed).sample(example_pool, min(num_examples, len(example_pool)))
    if examples:
        lines.append("Example lines from that collection (recognized → correct):")
        for record in examples:
            lines.append(f"- {record['greedy_text']} → {record['ground_truth']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Direct Python Qwen inference
# ---------------------------------------------------------------------------

def build_user_message(record, use_confidence_marks):
    sections = []
    if record.get("previous_lines"):
        sections.append("PREVIOUS LINE: " + " / ".join(record["previous_lines"]))
    if record.get("next_lines"):
        sections.append("NEXT LINE: " + " / ".join(record["next_lines"]))
    if use_confidence_marks and record.get("uncertain_text"):
        sections.append("UNCERTAIN: " + record["uncertain_text"])
    candidate_lines = ["CANDIDATES:"]
    for rank, candidate in enumerate(record["candidates"], start=1):
        percent = 100 * candidate["prob"]
        shown = f"{percent:.0f}%" if percent >= 1 else "<1%"
        candidate_lines.append(f"{rank}. [{shown}] {candidate['text']}")
    sections.append("\n".join(candidate_lines))
    return "\n".join(sections)


class QwenBatchError(RuntimeError):
    """Invalid output, excessive context, or an exhausted output budget."""


class LocalQwenCorrector:
    """Lazy Transformers inference with Gemini-style grouped line prompts."""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.processor = None
        self.is_vl = False
        self.cache = {}
        self.cache_path = Path(args.qwen_cache_path) if args.qwen_cache_path else None
        self.device = torch.device(
            ("cuda:0" if torch.cuda.is_available() else "cpu")
            if args.qwen_device == "auto" else args.qwen_device
        )
        if args.qwen_load_in_4bit and self.device.type != "cuda":
            raise ValueError("--qwen_load_in_4bit requires a CUDA device.")
        self._load_cache()

    def _cache_key(self, system_prompt, message):
        settings = {
            "model": self.args.qwen_model,
            "dtype": self.args.qwen_dtype,
            "four_bit": self.args.qwen_load_in_4bit,
            "temperature": self.args.qwen_temperature,
            "max_new_tokens": self.args.qwen_max_new_tokens,
            "batch_size": self.args.qwen_batch_size,
            "system_prompt": system_prompt,
            "user_message": message,
        }
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()

    def _load_cache(self):
        if self.cache_path is None or not self.cache_path.is_file():
            return
        with self.cache_path.open(encoding="utf-8") as cache_file:
            for line in cache_file:
                try:
                    entry = json.loads(line)
                    if isinstance(entry["key"], str) and isinstance(entry["corrected_text"], str):
                        self.cache[entry["key"]] = entry["corrected_text"]
                except (ValueError, KeyError, TypeError):
                    continue  # An interrupted final cache line must not prevent resuming.
        print(f"Loaded {len(self.cache)} cached Qwen correction(s) from {self.cache_path}")

    def _store(self, entries):
        for key, text in entries:
            self.cache[key] = text
        if self.cache_path is None or not entries:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as cache_file:
            for key, text in entries:
                cache_file.write(json.dumps({"key": key, "corrected_text": text}, ensure_ascii=False) + "\n")

    def load(self):
        if self.model is not None:
            return
        try:
            import transformers
            if not transformers.utils.is_torch_available():
                raise RuntimeError(
                    f"Transformers {transformers.__version__} cannot use installed PyTorch "
                    f"{torch.__version__}. Install a compatible PyTorch/torchvision pair "
                    "for your NVIDIA driver; Transformers 5.21 requires PyTorch >=2.6. "
                    "Qwen3.8 cannot use the older Transformers 4.57 fallback."
                )
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'Direct Qwen inference requires: pip install -U "transformers>=4.57" accelerate safetensors'
            ) from exc

        common = {"local_files_only": self.args.qwen_local_files_only}
        config = AutoConfig.from_pretrained(self.args.qwen_model, **common)
        # Multimodal Qwen checkpoints need conditional generation + AutoProcessor
        # even for text-only correction. Qwen3.8-27B uses the qwen3_5 architecture.
        multimodal_classes = {
            "qwen3_vl": "Qwen3VLForConditionalGeneration",
            "qwen3_5": "Qwen3_5ForConditionalGeneration",
        }
        self.is_vl = config.model_type in multimodal_classes
        if self.is_vl:
            try:
                model_class = getattr(transformers, multimodal_classes[config.model_type])
                processor_class = transformers.AutoProcessor
            except AttributeError as exc:
                raise RuntimeError(
                    f"Transformers {transformers.__version__} does not provide "
                    f"{multimodal_classes[config.model_type]}. Upgrade Transformers and "
                    "its supported PyTorch dependencies for this model."
                ) from exc
        else:
            model_class = AutoModelForCausalLM
            processor_class = AutoTokenizer

        dtype_name = self.args.qwen_dtype
        if dtype_name == "auto":
            dtype = (
                torch.float32 if self.device.type == "cpu" else
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            dtype = getattr(torch, dtype_name)
        # Transformers 5 calls this argument dtype; keep the 4.x spelling for
        # existing Qwen3/Qwen3-VL environments.
        dtype_argument = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        options = dict(common, low_cpu_mem_usage=True)
        options[dtype_argument] = dtype
        if self.args.qwen_load_in_4bit:
            from transformers import BitsAndBytesConfig
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            )
            options["device_map"] = {"": str(self.device)}
        print(f"Loading {self.args.qwen_model} on {self.device} ({dtype_name})...")
        try:
            self.processor = processor_class.from_pretrained(self.args.qwen_model, **common)
            self.model = model_class.from_pretrained(self.args.qwen_model, **options)
            if not self.args.qwen_load_in_4bit:
                self.model.to(self.device)
            self.model.eval()
        except Exception:
            self.unload()
            raise

    def unload(self):
        """Release Qwen before the next experiment's HTR model is built."""
        self.model = None
        self.processor = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _parse_response(content, expected_ids, batch_format):
        # Accept a JSON Markdown fence, but never guess corrections from prose.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            result = json.loads(content)
        except (ValueError, TypeError) as exc:
            # Recover only a complete, single-field response with unescaped
            # quotation marks in its text. Never infer batch IDs or repair
            # truncated objects, extra fields, prose, or invalid escapes.
            match = None if batch_format else re.fullmatch(
                r'\s*\{\s*"corrected_text"\s*:\s*"([^{}\r\n]*)"\s*\}\s*', content
            )
            if match is None:
                raise QwenBatchError("Qwen did not return valid correction JSON.") from exc
            text = match.group(1)
            repaired, escaped = [], False
            for index, char in enumerate(text):
                if char == '"' and not escaped:
                    if re.match(r'\s*[:,]', text[index + 1:]):
                        raise QwenBatchError("Qwen did not return valid correction JSON.") from exc
                    repaired.append('\\')
                repaired.append(char)
                escaped = char == '\\' and not escaped
            try:
                result = {"corrected_text": json.loads('"' + ''.join(repaired) + '"')}
            except (ValueError, TypeError) as repair_exc:
                raise QwenBatchError("Qwen did not return valid correction JSON.") from repair_exc
            print("Qwen response recovered: escaped quotation marks in corrected_text.", flush=True)
        if not isinstance(result, dict):
            raise QwenBatchError("Qwen correction JSON must be an object.")
        if batch_format:
            corrections = result.get("corrections")
            if not isinstance(corrections, list):
                raise QwenBatchError("Qwen response has no corrections array.")
        else:
            corrections = [{"sample_id": expected_ids[0], "corrected_text": result.get("corrected_text")}]
        accepted, seen = {}, set()
        for correction in corrections:
            if not isinstance(correction, dict):
                continue
            sample_id = correction.get("sample_id")
            text = correction.get("corrected_text")
            if not isinstance(sample_id, str) or sample_id not in expected_ids:
                continue
            if sample_id in seen:
                accepted.pop(sample_id, None)
                continue
            seen.add(sample_id)
            if not isinstance(text, str) or "\n" in text or "\r" in text:
                continue
            accepted[sample_id] = normalize_text(
                text.replace(UNCERTAIN_OPEN, "").replace(UNCERTAIN_CLOSE, "")
            )
        return accepted

    def _generate(self, system_prompt, batch, attempt=0):
        self.load()
        batch_format = self.args.qwen_batch_size > 1
        user_message = (
            "\n\n".join(f"{LINE_HEADER}{sample_id}\n{message}" for sample_id, message, _ in batch)
            if batch_format else batch[0][1]
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        if attempt:
            messages.append({"role": "user", "content":
                             'Return all requested corrections as valid JSON only. Include each requested ID exactly once. '
                             'Escape quotation marks inside corrected_text with a backslash: '
                             '{"corrected_text": "He said \\"hello\\"."}'})
        if self.is_vl:
            messages = [dict(m, content=[{"type": "text", "text": m["content"]}]) for m in messages]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            )
        else:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = self.processor(text, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.device)
        input_length = inputs["input_ids"].shape[-1]
        text_config = getattr(self.model.config, "text_config", self.model.config)
        context_limit = getattr(text_config, "max_position_embeddings", None)
        if context_limit and input_length + self.args.qwen_max_new_tokens > context_limit:
            raise QwenBatchError(
                f"Qwen context limit ({context_limit}) exceeded: {input_length} input tokens "
                f"plus {self.args.qwen_max_new_tokens} requested output tokens."
            )
        tokenizer = self.processor.tokenizer if self.is_vl else self.processor
        options = {
            "max_new_tokens": self.args.qwen_max_new_tokens,
            "do_sample": self.args.qwen_temperature > 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "return_dict_in_generate": False,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        }
        if self.args.qwen_temperature > 0:
            options["temperature"] = self.args.qwen_temperature
        with torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        generated = output[0, input_length:]
        content = tokenizer.decode(generated, skip_special_tokens=True)
        if getattr(self.args, "print_qwen_debug", 0) == 1:
            print(
                f"\nQwen raw response before validation (attempt {attempt + 1}, "
                f"sample IDs: {[entry[0] for entry in batch]}):\n"
                f"{content}\n--- End Qwen raw response ---",
                flush=True,
            )
        # Never accept potentially truncated corrections as a successful batch.
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        if len(generated) >= self.args.qwen_max_new_tokens and int(generated[-1]) not in eos_ids:
            raise QwenBatchError("Qwen output token budget exhausted; increase --qwen_max_new_tokens.")
        return self._parse_response(content, [entry[0] for entry in batch], batch_format)

    def _correct_batch(self, system_prompt, batch):
        results = {}
        remaining = list(batch)
        last_error = None
        for attempt in range(self.args.qwen_retries + 1):
            oom = False
            try:
                accepted = self._generate(system_prompt, remaining, attempt)
            except torch.OutOfMemoryError:
                # Leave the exception scope before splitting so its traceback
                # does not retain the failed batch's GPU tensors.
                oom = True
                accepted = {}
                last_error = QwenBatchError("Qwen ran out of memory.")
            except QwenBatchError as exc:
                accepted = {}
                last_error = QwenBatchError(str(exc))
            self._store([(key, accepted[sample_id]) for sample_id, _, key in remaining if sample_id in accepted])
            results.update(accepted)
            remaining = [entry for entry in remaining if entry[0] not in accepted]
            if not remaining:
                return results
            if accepted or last_error is None:
                last_error = QwenBatchError(f"{len(remaining)} corrections are missing or invalid.")
            if len(remaining) > 1:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                half = len(remaining) // 2
                print(f"Qwen batch incomplete ({last_error}); splitting {len(remaining)} lines.")
                results.update(self._correct_batch(system_prompt, remaining[:half]))
                results.update(self._correct_batch(system_prompt, remaining[half:]))
                return results
            if oom:
                raise RuntimeError(
                    "Qwen ran out of memory for one line. Try --qwen_load_in_4bit, "
                    "a smaller --qwen_model, or fewer --nbest candidates."
                )
        raise RuntimeError(
            f"Qwen correction failed for sample {remaining[0][0]!r} after "
            f"{self.args.qwen_retries + 1} attempts: {last_error}"
        )

    def correct_many(self, system_prompt, items):
        """Correct (sample_id, message) items, caching every accepted line."""
        if len({sample_id for sample_id, _ in items}) != len(items):
            raise ValueError("Qwen input sample IDs must be unique within an experiment.")
        results, pending = {}, []
        for sample_id, message in items:
            key = self._cache_key(system_prompt, message)
            if key in self.cache:
                results[sample_id] = self.cache[key]
            else:
                pending.append((sample_id, message, key))
        print(f"Qwen: {len(results)} cached lines; {len(pending)} to correct; batch size={self.args.qwen_batch_size}.")
        for start in range(0, len(pending), self.args.qwen_batch_size):
            batch = pending[start:start + self.args.qwen_batch_size]
            results.update(self._correct_batch(system_prompt, batch))
            print(f"Post-corrected {len(results)}/{len(items)} samples", end="\r", flush=True)
        print()
        return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_model(args, device, nb_chars):
    cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
    head_cfg = (256, 3)  # (hidden dimension, num_layers blstm)

    model_reco = CRNN(cnn_cfg, head_cfg, nb_chars, args.model_type)

    print("Initializing model weights kaiming")
    for p in model_reco.parameters():
        if p.dim() > 1:
            torch.nn.init.kaiming_normal_(p, nonlinearity="relu")

    if not os.path.isfile(args.path_model):
        raise FileNotFoundError(f"HTR checkpoint not found: {args.path_model}")
    load_pretrained_model(args, model_reco, device)

    print(f"Transferring model to {str(device)}...")
    return model_reco.to(device)


def build_loader(args, split_directory, char_dict, collate_fn):
    width_divisor = 8
    dataset = HTRDataset(
        split_directory,
        (args.height_max, args.width_max),
        width_divisor,
        args.padding_strategy,
        args.pad_left,
        args.pad_right,
        None,
        char_dict,
    )
    print(f"Nb samples {split_directory}: {len(dataset)}")
    return DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )


def summarize_buckets(records):
    buckets = [("0-10%", 0.0, 0.1), ("10-30%", 0.1, 0.3), (">=30%", 0.3, float("inf"))]
    print("Greedy line-CER bucket | lines | greedy CER | Qwen CER | improved | worsened")
    for name, low, high in buckets:
        selected = [
            record for record in records
            if low <= line_cer(record["ground_truth"], record["greedy_text"]) < high
        ]
        if not selected:
            print(f"{name:>22} | 0")
            continue
        truths = [record["ground_truth"] for record in selected]
        greedy_cer = corpus_cer(truths, [record["greedy_text"] for record in selected])
        qwen_cer = corpus_cer(truths, [record["qwen_text"] for record in selected])
        improved = sum(
            editdistance.eval(r["ground_truth"], r["qwen_text"])
            < editdistance.eval(r["ground_truth"], r["greedy_text"])
            for r in selected
        )
        worsened = sum(
            editdistance.eval(r["ground_truth"], r["qwen_text"])
            > editdistance.eval(r["ground_truth"], r["greedy_text"])
            for r in selected
        )
        print(
            f"{name:>22} | {len(selected):5d} | {100 * greedy_cer:9.2f}% | "
            f"{100 * qwen_cer:7.2f}% | {improved:8d} | {worsened:8d}"
        )


def evaluate_experiment(args, device, char_list, char_dict, collate_fn, source_dataset, corrector):
    token_blank = char_dict[BLANK_STR_TOKEN]
    model_reco = build_model(args, device, len(char_list))

    fewshot_block = ""
    if corrector is not None and args.use_fewshot:
        source_split = os.path.join(
            args.dataset_root, source_dataset, args.dataset_subdir, args.fewshot_split
        )
        if os.path.isdir(source_split):
            source_loader = build_loader(args, source_split, char_dict, collate_fn)
            source_records = run_recognizer(
                args, source_loader, model_reco, device, char_list, token_blank
            )
            fewshot_block = build_fewshot_block(
                source_records, args.confusion_top_k, args.fewshot_examples, args.fewshot_seed
            )
            print("Few-shot block:")
            print(fewshot_block)
        else:
            print(f"WARNING: source split not found, few-shot disabled: {source_split}")

    test_loader = build_loader(args, os.path.join(args.dataset_path, "test"), char_dict, collate_fn)
    decoder = build_ctc_decoder(char_list, token_blank)
    records = run_recognizer(args, test_loader, model_reco, device, char_list, token_blank, decoder)
    page_xml_dir = os.path.join(args.dataset_root, args.target_dataset, args.page_xml_subdir)
    add_neighbour_context(records, args.context_lines, page_xml_dir)

    # Release CRNN GPU memory before loading Qwen in this process.
    model_reco.to("cpu")
    del model_reco
    if device.type == "cuda":
        torch.cuda.empty_cache()

    system_prompt = None
    if corrector is not None:
        system_prompt = correction_system_prompt(
            args.language_context, fewshot_block, args.qwen_batch_size, args.qwen_prompt
        )

    items = []
    for record in records:
        record["oracle_text"] = min(
            (candidate["text"] for candidate in record["candidates"]),
            key=lambda text: editdistance.eval(record["ground_truth"], text),
        )
        record["qwen_text"] = record["candidates"][0]["text"]
        if corrector is not None and record["candidates"][0]["text"]:
            record["user_message"] = build_user_message(record, args.use_confidence_marks)
            items.append((str(record["sample_id"]), record["user_message"]))

    if corrector is not None:
        try:
            corrected = corrector.correct_many(system_prompt, items)
            for record in records:
                if "user_message" in record:
                    record["qwen_text"] = corrected[str(record["sample_id"])]
        finally:
            corrector.unload()

    for record in records[:6]:
        print("-----Ground truth:-----")
        print(record["ground_truth"])
        print("-----Greedy prediction:-----")
        print(record["greedy_text"])
        if record.get("user_message"):
            print("-----Qwen input:-----")
            print(record["user_message"])
        print("-----Qwen-corrected prediction:-----")
        print(record["qwen_text"])

    truths = [record["ground_truth"] for record in records]
    metrics = {
        "greedy_cer": corpus_cer(truths, [r["greedy_text"] for r in records]),
        "beam_cer": corpus_cer(truths, [r["candidates"][0]["text"] for r in records]),
        "oracle_cer": corpus_cer(truths, [r["oracle_text"] for r in records]),
        "qwen_cer": corpus_cer(truths, [r["qwen_text"] for r in records]),
        "qwen_in_candidates": sum(
            r["qwen_text"] in {c["text"] for c in r["candidates"]} for r in records
        ) / max(1, len(records)),
    }
    print(f"Greedy CER: {100 * metrics['greedy_cer']:.2f}%")
    candidate_label = "CTC greedy" if args.ctc_decoding == "greedy" else "CTC beam top-1"
    print(f"{candidate_label} CER: {100 * metrics['beam_cer']:.2f}%")
    candidate_count = 1 if args.ctc_decoding == "greedy" else args.nbest
    print(f"Candidate-set ({candidate_count}) oracle CER: {100 * metrics['oracle_cer']:.2f}%")
    print(f"Qwen-corrected CER: {100 * metrics['qwen_cer']:.2f}%")
    print(f"Qwen output equals a candidate: {100 * metrics['qwen_in_candidates']:.1f}% of lines")
    summarize_buckets(records)
    return metrics, records, system_prompt


def dump_records(args, exp_id, experiment, records, system_prompt):
    output_dir = Path(args.predictions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{exp_id}_{experiment['DA_adaptation']}_{args.ctc_decoding}.jsonl"
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        if system_prompt is not None:
            output_file.write(json.dumps({"system_prompt": system_prompt}, ensure_ascii=False) + "\n")
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Per-line predictions written to {output_path}")


def append_result(args, exp_id, experiment, metrics):
    output_path = Path(args.results_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", encoding="utf-8") as output_file:
        if write_header:
            output_file.write(
                "experiment_id;adaptation;greedy_cer;beam_cer;oracle_cer;qwen_cer;"
                "nbest;confidence_marks;confidence_threshold;context_lines;fewshot;qwen_model;qwen_batch_size;qwen_temperature;qwen_max_new_tokens;"
                "ctc_decoding\n"
            )
        output_file.write(
            f"{exp_id};{experiment['DA_adaptation']};"
            f"{100 * metrics['greedy_cer']:.2f};{100 * metrics['beam_cer']:.2f};"
            f"{100 * metrics['oracle_cer']:.2f};{100 * metrics['qwen_cer']:.2f};"
            f"{args.nbest};{args.use_confidence_marks};{args.confidence_threshold};"
            f"{args.context_lines};{args.use_fewshot};"
            f"{args.qwen_model if args.llm_post_correction else 'none'};"
            f"{args.qwen_batch_size};{args.qwen_temperature};{args.qwen_max_new_tokens};"
            f"{args.ctc_decoding}\n"
        )



trained_on = {
    "101": {
        "DA_adaptation": "IAM-Bentham",
        "exp_dir": "DA-101",
        "crnn_model": "net_best_T_IAM_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "102": {
        "DA_adaptation": "IAM-NorHandV3_72702",
        "exp_dir": "DA-102",
        "crnn_model": "net_best_T_IAM_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "103": {
        "DA_adaptation": "IAM-NorHandV3_331861",
        "exp_dir": "DA-103",
        "crnn_model": "net_best_T_IAM_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "104": {
        "DA_adaptation": "IAM-Riksarkivet_Bergskollegium",
        "exp_dir": "DA-104",
        "crnn_model": "net_best_T_IAM_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "105": {
        "DA_adaptation": "IAM-Riksarkivet_Goteborgs2",
        "exp_dir": "DA-105",
        "crnn_model": "net_best_T_IAM_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
    "106": {
        "DA_adaptation": "IAM-READ_2016",
        "exp_dir": "DA-106",
        "crnn_model": "net_best_T_IAM_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "107": {
        "DA_adaptation": "Bentham-IAM",
        "exp_dir": "DA-107",
        "crnn_model": "net_best_T_Bentham_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "108": {
        "DA_adaptation": "Bentham-NorHandV3_72702",
        "exp_dir": "DA-108",
        "crnn_model": "net_best_T_Bentham_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "109": {
        "DA_adaptation": "Bentham-NorHandV3_331861",
        "exp_dir": "DA-109",
        "crnn_model": "net_best_T_Bentham_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "110": {
        "DA_adaptation": "Bentham-Riksarkivet_Bergskollegium",
        "exp_dir": "DA-110",
        "crnn_model": "net_best_T_Bentham_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "111": {
        "DA_adaptation": "Bentham-Riksarkivet_Goteborgs2",
        "exp_dir": "DA-111",
        "crnn_model": "net_best_T_Bentham_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
    "112": {
        "DA_adaptation": "Bentham-READ_2016",
        "exp_dir": "DA-112",
        "crnn_model": "net_best_T_Bentham_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "113": {
        "DA_adaptation": "NorHandV3_72702-IAM",
        "exp_dir": "DA-113",
        "crnn_model": "net_best_T_NorHandV3_72702_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "114": {
        "DA_adaptation": "NorHandV3_72702-Bentham",
        "exp_dir": "DA-114",
        "crnn_model": "net_best_T_NorHandV3_72702_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "115": {
        "DA_adaptation": "NorHandV3_72702-NorHandV3_331861",
        "exp_dir": "DA-115",
        "crnn_model": "net_best_T_NorHandV3_72702_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "116": {
        "DA_adaptation": "NorHandV3_72702-Riksarkivet_Bergskollegium",
        "exp_dir": "DA-116",
        "crnn_model": "net_best_T_NorHandV3_72702_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "117": {
        "DA_adaptation": "NorHandV3_72702-Riksarkivet_Goteborgs2",
        "exp_dir": "DA-117",
        "crnn_model": "net_best_T_NorHandV3_72702_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
    "118": {
        "DA_adaptation": "NorHandV3_72702-READ_2016",
        "exp_dir": "DA-118",
        "crnn_model": "net_best_T_NorHandV3_72702_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "119": {
        "DA_adaptation": "NorHandV3_331861-IAM",
        "exp_dir": "DA-119",
        "crnn_model": "net_best_T_NorHandV3_331861_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "120": {
        "DA_adaptation": "NorHandV3_331861-Bentham",
        "exp_dir": "DA-120",
        "crnn_model": "net_best_T_NorHandV3_331861_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "121": {
        "DA_adaptation": "NorHandV3_331861-NorHandV3_72702",
        "exp_dir": "DA-121",
        "crnn_model": "net_best_T_NorHandV3_331861_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "122": {
        "DA_adaptation": "NorHandV3_331861-Riksarkivet_Bergskollegium",
        "exp_dir": "DA-122",
        "crnn_model": "net_best_T_NorHandV3_331861_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "123": {
        "DA_adaptation": "NorHandV3_331861-Riksarkivet_Goteborgs2",
        "exp_dir": "DA-123",
        "crnn_model": "net_best_T_NorHandV3_331861_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
    "124": {
        "DA_adaptation": "NorHandV3_331861-READ_2016",
        "exp_dir": "DA-124",
        "crnn_model": "net_best_T_NorHandV3_331861_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "125": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-IAM",
        "exp_dir": "DA-125",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "126": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-Bentham",
        "exp_dir": "DA-126",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "127": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-NorHandV3_72702",
        "exp_dir": "DA-127",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "128": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-NorHandV3_331861",
        "exp_dir": "DA-128",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "129": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-Riksarkivet_Goteborgs2",
        "exp_dir": "DA-129",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
    "130": {
        "DA_adaptation": "Riksarkivet_Bergskollegium-READ_2016",
        "exp_dir": "DA-130",
        "crnn_model": "net_best_T_Riksarkivet_Bergskollegium_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "131": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-IAM",
        "exp_dir": "DA-131",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "132": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-Bentham",
        "exp_dir": "DA-132",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "133": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-NorHandV3_72702",
        "exp_dir": "DA-133",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "134": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-NorHandV3_331861",
        "exp_dir": "DA-134",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "135": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-Riksarkivet_Bergskollegium",
        "exp_dir": "DA-135",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "136": {
        "DA_adaptation": "Riksarkivet_Goteborgs2-READ_2016",
        "exp_dir": "DA-136",
        "crnn_model": "net_best_T_Riksarkivet_Goteborgs2_2_READ_2016.pth",
        "target_dataset": "READ_2016"
    },
    "137": {
        "DA_adaptation": "READ_2016-IAM",
        "exp_dir": "DA-137",
        "crnn_model": "net_best_T_READ_2016_2_IAM.pth",
        "target_dataset": "IAM"
    },
    "138": {
        "DA_adaptation": "READ_2016-Bentham",
        "exp_dir": "DA-138",
        "crnn_model": "net_best_T_READ_2016_2_Bentham.pth",
        "target_dataset": "Bentham"
    },
    "139": {
        "DA_adaptation": "READ_2016-NorHandV3_72702",
        "exp_dir": "DA-139",
        "crnn_model": "net_best_T_READ_2016_2_NorHandV3_72702.pth",
        "target_dataset": "NorHandV3_72702"
    },
    "140": {
        "DA_adaptation": "READ_2016-NorHandV3_331861",
        "exp_dir": "DA-140",
        "crnn_model": "net_best_T_READ_2016_2_NorHandV3_331861.pth",
        "target_dataset": "NorHandV3_331861"
    },
    "141": {
        "DA_adaptation": "READ_2016-Riksarkivet_Bergskollegium",
        "exp_dir": "DA-141",
        "crnn_model": "net_best_T_READ_2016_2_Riksarkivet_Bergskollegium.pth",
        "target_dataset": "Riksarkivet_Bergskollegium"
    },
    "142": {
        "DA_adaptation": "READ_2016-Riksarkivet_Goteborgs2",
        "exp_dir": "DA-142",
        "crnn_model": "net_best_T_READ_2016_2_Riksarkivet_Goteborgs2.pth",
        "target_dataset": "Riksarkivet_Goteborgs2"
    },
 }

if __name__ == '__main__':

    time_start = time.time()

    args = parser.parse_args()
    faulthandler.enable()
    # The ⟨...⟩ markers must not crash printing on non-UTF-8 consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if args.nbest <= 0:
        raise ValueError(f"--nbest must be greater than zero, got {args.nbest}.")
    if args.qwen_batch_size <= 0 or args.qwen_max_new_tokens <= 0:
        raise ValueError("--qwen_batch_size and --qwen_max_new_tokens must be positive.")
    if args.qwen_temperature < 0 or args.qwen_retries < 0:
        raise ValueError("--qwen_temperature and --qwen_retries must be non-negative.")
    if args.ctc_decoding == "greedy":
        args.nbest = 1

    corrector = None
    if args.llm_post_correction:
        corrector = LocalQwenCorrector(args)
        print(f"Direct Python Qwen: {args.qwen_model}; batch size={args.qwen_batch_size}")
    else:
        print("Qwen post-correction disabled by command-line parameter.")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device :")
    print(device)
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    print("torch.cuda.device_count(): " + str(torch.cuda.device_count()))
    print(
        f"ctc_decoding={args.ctc_decoding}, nbest={args.nbest}, "
        f"confidence_marks={args.use_confidence_marks} "
        f"(threshold={args.confidence_threshold}), context_lines={args.context_lines}, "
        f"fewshot={args.use_fewshot}"
    )

    # Pad img with black = 0
    c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
    collate_fn = c_collate_fn.collate_fn

    experiments = trained_on
    if args.experiment_ids:
        unknown_ids = sorted(set(args.experiment_ids) - set(trained_on))
        if unknown_ids:
            raise ValueError(f"Unknown experiment IDs: {', '.join(unknown_ids)}")
        experiments = {exp_id: trained_on[exp_id] for exp_id in args.experiment_ids}

    for exp_id, value_train in experiments.items():
        print(f"CRNN model Trained on: {exp_id} ({value_train['DA_adaptation']})")
        args.path_model = f"../../logs/DomainAdaptation/DA_Convnext/{args.run_name}/{value_train['exp_dir']}_{args.run_name}/checkpoint/{value_train['crnn_model']}"
        args.dataset_path = os.path.join(
            args.dataset_root, value_train["target_dataset"], args.dataset_subdir
        )
        source_dataset = value_train["DA_adaptation"].split("-", 1)[0]
        args.target_dataset = value_train["target_dataset"]

        args.language_context = language_context_for_dataset(
            value_train["target_dataset"], args.target_language
        )
        print(f"{value_train['target_dataset']}: Qwen target language/domain: {args.language_context}")

        # Alphabet
        charsets_path = os.path.join(args.dataset_path, args.charset)
        charset = CharsetToken([charsets_path], use_blank=True)
        char_list = charset.get_charset_list()
        char_dict = charset.get_charset_dictionary()

        metrics, records, system_prompt = evaluate_experiment(
            args, device, char_list, char_dict, collate_fn, source_dataset, corrector
        )
        dump_records(args, exp_id, value_train, records, system_prompt)
        append_result(args, exp_id, value_train, metrics)
    end_time = time.time()
    print(f"Total execution time: {end_time - time_start:.2f} seconds")
