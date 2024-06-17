import editdistance
import numpy as np
from utils.utils import load_jsonl
from nltk.tokenize import RegexpTokenizer
from typing import FrozenSet
import keyword
import re

string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''
code_tokenizer = RegexpTokenizer(r'\w+')
IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def compute_EM(target, prediction, language="python"):
    comment_prefix = ""
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"

    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]
    target_lines_str = "".join(target_lines)
    prediction_lines_str = "".join(prediction_lines)
    if target_lines_str == prediction_lines_str:
        return 1
    else:
        return 0


def compute_ES(target, prediction, language="python"):

    comment_prefix = ""
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"

    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]

    target_str = ''.join(target_lines)
    prediction_str = ''.join(prediction_lines)
    ES_score = 1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))

    return ES_score


def hit(search_cases, hits=None):
    if hits is None:
        hits = [1, 5, 10]
    hit_res = [0.0 for _ in range(0, len(hits))]
    for case in search_cases:
        target_lines = [line.strip() for line in case['metadata']['ground_truth'].splitlines() if line.strip()]
        target_lines = [line for line in target_lines if not line.startswith('#')]
        target_line = "".join(target_lines)
        hit_pos = np.inf
        for i in range(1, len(case['top_k_context'])+1):
            prediction_lines = [line.strip() for line in case['top_k_context'][-i][0].splitlines() if line.strip()]
            prediction_lines = [line for line in prediction_lines if not line.startswith('#')]
            prediction_line = "".join(prediction_lines)
            if target_line in prediction_line:
                hit_pos = i
                break
        for i in range(0, len(hits)):
            if hits[i] >= hit_pos:
                hit_res[i] += 1.0

    for i in range(0, len(hit_res)):
        hit_res[i] /= len(search_cases)
    return hit_res


def compute_batch_EM(ground_truth_file_path, generation_res_file_path, language="python"):
    gt_res = load_jsonl(ground_truth_file_path)
    pred_res = load_jsonl(generation_res_file_path)
    em_val = 0
    for i in range(0, len(gt_res)):
        pred_case = pred_res[i]
        pred_str = pred_case['generate_response']
        gt_str = gt_res[i]['metadata']['ground_truth']
        em_val += compute_EM(gt_str, pred_str, language=language)
    return em_val / len(pred_res)


def compute_batch_ES(ground_truth_file_path, generation_res_file_path, language="python"):
    gt_res = load_jsonl(ground_truth_file_path)
    pred_res = load_jsonl(generation_res_file_path)
    es_val = 0

    for i in range(0, len(gt_res)):
        pred_case = pred_res[i]
        pred_str = pred_case['generate_response']
        gt_str = gt_res[i]['metadata']['ground_truth']
        es_val += compute_ES(gt_str, pred_str, language=language)
    return es_val / len(gt_res)


def get_language_keywords() -> FrozenSet[str]:
    return frozenset(k for k in keyword.kwlist if k != 'True' and k != 'False')


def is_identifier(token, language="python"):
    return True if IDENTIFIER_REGEX.match(token) \
                   and (language is None or token not in get_language_keywords()) else False


def extract_identifiers(source_code, language="python"):
    # the main idea is to remove String from a source code
    # then, tokenize the code to get all words and match with identifier regular expression
    # check if it is a language specific keyword, it not, then it is an identifier
    source_code_without_strings = re.sub(string_pattern, '', source_code)
    _ids = [t for t in code_tokenizer.tokenize(source_code_without_strings) if is_identifier(t, language=language)]
    return _ids


def compute_id_match(pred_ids, target_ids):
    pred_ids = list(set(pred_ids))
    target_ids = list(set(target_ids))
    tp = 0
    fp = 0
    fn = 0
    for pid in pred_ids:
        if pid in target_ids:
            tp += 1
        else:
            fp += 1
    for tid in target_ids:
        if tid not in pred_ids:
            fn += 1
    return tp, fp, fn


def compute_identifier_match(prediction, target, language="python"):

    comment_prefix = ""
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"

    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]
    target_lines_str = "".join(target_lines)
    prediction_lines_str = "".join(prediction_lines)

    pred_ids = extract_identifiers(prediction_lines_str, language=language)
    gt_ids = extract_identifiers(target_lines_str, language=language)
    identifier_em = int(pred_ids == gt_ids)
    id_tp, id_fp, id_fn = compute_id_match(pred_ids, gt_ids)
    id_f1 = 2 * id_tp / (2 * id_tp + id_fp + id_fn) if (2 * id_tp + id_fp + id_fn) != 0 else 0
    return identifier_em, id_f1


def compute_bath_identifier_match(ground_truth_file_path, generation_res_file_path, language="python"):
    gt_res = load_jsonl(ground_truth_file_path)
    pred_res = load_jsonl(generation_res_file_path)
    em_val = 0
    f1_val = 0
    for i in range(0, len(gt_res)):
        pred_case = pred_res[i]
        pred_str = pred_case['generate_response']
        gt_str = gt_res[i]['metadata']['ground_truth']
        em, f1 = compute_identifier_match(pred_str, gt_str, language=language)
        em_val += em
        f1_val += f1
    return em_val / len(pred_res), f1_val/len(pred_res)