import editdistance
import numpy as np
from utils.utils import load_jsonl


def compute_EM(target, prediction):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith('#')]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith('#')][:len(target_lines)]
    target_lines_str = "".join(target_lines)
    prediction_lines_str = "".join(prediction_lines)
    if target_lines_str == prediction_lines_str:
        return 1
    else:
        return 0


def compute_ES(target, prediction):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith('#')]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith('#')][:len(target_lines)]

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


def compute_batch_EM(ground_truth_file_path, generation_res_file_path):
    gt_res = load_jsonl(ground_truth_file_path)
    pred_res = load_jsonl(generation_res_file_path)
    em_val = 0
    for i in range(0, len(gt_res)):
        pred_case = pred_res[i]
        pred_str = pred_case['generate_response']
        gt_str = gt_res[i]['metadata']['ground_truth']
        em_val += compute_EM(gt_str, pred_str)
    return em_val / len(pred_res)


def compute_batch_ES(ground_truth_file_path, generation_res_file_path):
    gt_res = load_jsonl(ground_truth_file_path)
    pred_res = load_jsonl(generation_res_file_path)
    es_val = 0

    for i in range(0, len(gt_res)):
        pred_case = pred_res[i]
        pred_str = pred_case['generate_response']
        gt_str = gt_res[i]['metadata']['ground_truth']
        es_val += compute_ES(gt_str, pred_str)
    return es_val / len(gt_res)
