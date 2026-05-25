"""
中文优化的评估指标
使用字符级n-gram，更适合中文评估
"""

import re
import string
from collections import Counter
from hwrag.evaluator.metrics import BaseMetric


def cn_normalize(text):
    """
    中文优化的文本标准化
    1. 转小写
    2. 移除英文标点，保留中文
    """
    text = text.lower()

    # 只移除英文标点，保留中文标点
    en_punc = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    text = "".join(ch for ch in text if ch not in en_punc)

    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def cn_tokenize_2gram(text):
    """
    中文2-gram分词（字符级）
    例如: "计算机科学" -> ["计算", "算计", "算科", "科学"]
    """
    text = cn_normalize(text)

    # 2-gram
    tokens = [text[i:i+2] for i in range(len(text)-1)]

    # 过滤单字符
    tokens = [t for t in tokens if len(t) == 2]

    return tokens


def cn_tokenize_char(text):
    """
    字符级分词
    """
    text = cn_normalize(text)
    return list(text)


def compute_f1_chinese(prediction, ground_truth, use_2gram=True):
    """
    基于字符级n-gram的F1计算
    """
    if use_2gram:
        pred_tokens = cn_tokenize_2gram(prediction)
        gt_tokens = cn_tokenize_2gram(ground_truth)
    else:
        pred_tokens = cn_tokenize_char(prediction)
        gt_tokens = cn_tokenize_char(ground_truth)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return {'f1': 0, 'precision': 0, 'recall': 0}

    # 计算交集
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return {'f1': 0, 'precision': 0, 'recall': 0}

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'f1': f1, 'precision': precision, 'recall': recall}


def compute_f1_best(prediction, ground_truth):
    """
    尝试多种分词方式，取最高分
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    best_metric = {'f1': 0, 'precision': 0, 'recall': 0}

    for gt in ground_truth:
        # 尝试2-gram
        result_2gram = compute_f1_chinese(prediction, gt, use_2gram=True)
        # 尝试字符级
        result_char = compute_f1_chinese(prediction, gt, use_2gram=False)

        # 取每个指标的最高值
        for k in ['f1', 'precision', 'recall']:
            best_metric[k] = max(best_metric[k], result_2gram[k], result_char[k])

    return best_metric


class ChineseF1Metric(BaseMetric):
    """使用中文分词的F1指标"""

    metric_name = "cn_f1"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = data.golden_answers

        metric_score_list = []
        for pred, golden_answers in zip(pred_list, golden_answers_list):
            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]

            best_f1 = 0
            for gt in golden_answers:
                result = compute_f1_best(pred, gt)
                best_f1 = max(best_f1, result['f1'])

            metric_score_list.append(best_f1)

        f1 = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0
        return {"cn_f1": f1}, metric_score_list


class ChineseRecallMetric(BaseMetric):
    """使用中文分词的召回率指标"""

    metric_name = "cn_recall"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = data.golden_answers

        metric_score_list = []
        for pred, golden_answers in zip(pred_list, golden_answers_list):
            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]

            best_recall = 0
            for gt in golden_answers:
                result = compute_f1_best(pred, gt)
                best_recall = max(best_recall, result['recall'])

            metric_score_list.append(best_recall)

        recall = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0
        return {"cn_recall": recall}, metric_score_list


class ChinesePrecisionMetric(BaseMetric):
    """使用中文分词的精确率指标"""

    metric_name = "cn_precision"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = data.golden_answers

        metric_score_list = []
        for pred, golden_answers in zip(pred_list, golden_answers_list):
            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]

            best_precision = 0
            for gt in golden_answers:
                result = compute_f1_best(pred, gt)
                best_precision = max(best_precision, result['precision'])

            metric_score_list.append(best_precision)

        precision = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0
        return {"cn_precision": precision}, metric_score_list


class ChineseSubEMMetric(BaseMetric):
    """中文优化的子串匹配指标"""

    metric_name = "cn_sub_em"

    def __init__(self, config):
        super().__init__(config)

    def calculate_sub_em(self, prediction, golden_answers):
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]

        normalized_pred = cn_normalize(prediction)

        for golden_answer in golden_answers:
            normalized_gt = cn_normalize(golden_answer)
            if normalized_gt in normalized_pred:
                return 1.0

        return 0.0

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = data.golden_answers

        metric_score_list = [
            self.calculate_sub_em(pred, golden_answers)
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]

        sub_em = sum(metric_score_list) / len(metric_score_list)
        return {"cn_sub_em": sub_em}, metric_score_list