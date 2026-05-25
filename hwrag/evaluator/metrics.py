import re
import warnings
from collections import Counter
from hwrag.evaluator.utils import normalize_answer


class BaseMetric:
    """`BaseMetric` serves as the base object of all metrics. Implemented metric should
    inherit this class.
    """

    metric_name = "base"

    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset_name']

    def calculate_metric(self, data):
        """Get the total score of this metric and score for each sample.

        Args:
            data object: it contains basic information and generated information.

        Returns:
            (metric_score: dict, metric_score_list: list)
            metric_score: such as ``{'em': 0.53}``.
            metric_score_list: score for each sample.

        """
        return {}, []


class F1_Score(BaseMetric):
    """
    计算预测和真实答案之间的F1分数，这是精确度和召回率的调和平均数。
    """

    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)

    def token_level_scores(self, prediction: str, ground_truths: str):
        """
        计算单个预测与一个或多个真实答案之间的token级F1分数。

        Args:
            prediction (str): 预测的文本。
            ground_truths (str or list): 真实的答案，可以是单个字符串或字符串列表。

        Returns:
            dict: 包含 'f1', 'precision', 和 'recall' 的字典。
        """
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)

            # 如果预测或真实答案为特定单词（yes, no, noanswer），且不匹配，则跳过
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no',
                                           'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue

            # 将预测的答案和真实的答案分割成单词（tokens）
            #prediction_tokens = normalized_prediction.split()
            #ground_truth_tokens = normalized_ground_truth.split()

            # 修改后 (兼容中英文)
            def get_tokens(text):
                # 如果包含中文，则按字符拆分；如果是纯英文，则按空格拆分
                if re.search(r'[一-龥]', text):
                    # 中文：去掉空格后按字符拆分
                    return list(text.replace(" ", ""))
                else:
                    # 英文：按空格拆分
                    return text.split()

            prediction_tokens = get_tokens(normalized_prediction)
            ground_truth_tokens = get_tokens(normalized_ground_truth)


            # 使用Counter对象计算两组tokens的计数，然后使用 & 操作符得到两者的交集，即预测答案和真实答案中共同出现的单词及其最小出现次数。
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            # 计算公式为共同单词总数除以预测答案中的单词总数。这衡量了预测中的哪些单词实际上是正确的。
            precision = 1.0 * num_same / len(prediction_tokens)

            # 计算公式为共同单词总数除以真实答案中的单词总数。这衡量了所有应该被预测的正确单词中有多少被实际预测出来了。
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def calculate_metric(self, data):
        """
        计算数据集上的F1分数。

        Args:
            data: 包含预测答案和真实答案的数据集。


        Returns:
            tuple: 包含总体F1分数和每个样本的F1分数列表。

        """
        pred_list = data.pred
        golden_answers_list = data.golden_answers
        metric_score_list = [self.token_level_scores(pred, golden_answers)['f1'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        f1 = sum(metric_score_list) / len(metric_score_list)
        return {"f1": f1}, metric_score_list



class Recall_Score(F1_Score):
    """基于分词的召回率得分"""

    # 定义度量名称为召回率
    metric_name = "recall"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        """
        计算数据集上的召回率得分。

        Args:
            data: 包含预测答案和真实答案的数据集。

        Returns:
            tuple: 包含总体召回率得分和每个样本的召回率得分列表。
        """

        # 从数据集中获取所有预测答案
        pred_list = data.pred
        print(f"pred_list: {pred_list}")
        # 从数据集中获取所有真实答案
        golden_answers_list = data.golden_answers
        print(f"golden_answers_list: {golden_answers_list}")
        # 计算每个样本的召回率
        metric_score_list = [self.token_level_scores(pred, golden_answers)['recall'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]

        # 计算平均召回率
        precision = sum(metric_score_list) / len(metric_score_list)

        # 返回召回率结果
        return {"recall": precision}, metric_score_list


class Precision_Score(F1_Score):
    """Token-level Precision score"""

    metric_name = "precision"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = data.golden_answers
        metric_score_list = [self.token_level_scores(pred, golden_answers)['precision'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"precision": precision}, metric_score_list


class ExactMatch(BaseMetric):
    """
    Exact match (EM) 指标用于测量预测答案与标准答案是否完全一致。

    Attributes:
        metric_name (str): 度量指标的名称，这里是"em"，代表精确匹配。


    精确匹配（Exact Match, EM）是一种常用于评估自然语言处理任务，特别是问答系统中的指标。
    它测量的是预测答案是否与真实答案在文本上完全一致。
    这种评估方式非常严格，即只有当预测的答案与参考答案在字面上完全相同，包括所有的单词和标点符号，才会被视为正确。
    """
    metric_name = "em"

    def __init__(self, config):
        """
        初始化ExactMatch实例。

        Args:
            config (dict): 配置字典，包含评估所需的各种参数。
        """
        super().__init__(config)
        self.is_regex = self.dataset_name == 'curatedtrec'

    def calculate_em(self, prediction: str, golden_answers: list) -> float:
        """
        计算单个预测的精确匹配得分。

        Args:
            prediction (str): 模型生成的预测文本。
            golden_answers (list): 可能的正确答案列表。

        Returns:
            float: 预测的精确匹配得分，1.0 表示完全匹配，0.0 表示不匹配。
        """
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            # 如果答案应视为正则表达式，则以正则表达式方式匹配
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.fullmatch(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        em_score = sum(metric_score_list) / len(metric_score_list)

        return {"em": em_score}, metric_score_list


class Sub_ExactMatch(BaseMetric):
    """
    是基于基类 BaseMetric 实现的一个评估指标。
    这个指标用于衡量预测答案是否包含了标准答案，即使不完全相同，也认为是部分正确。
    这种评估方法比完全精确匹配（Exact Match）要宽松，适合于那些允许答案有部分对应即可的场景。

    """
    metric_name = "sub_em"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == 'curatedtrec'

    def calculate_sub_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.search(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                #if golden_answer in normalized_prediction:
                if (golden_answer in normalized_prediction) or (normalized_prediction in golden_answer):
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [self.calculate_sub_em(pred, golden_answers) for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        sub_em_score = sum(metric_score_list) / len(metric_score_list)

        return {"sub_em": sub_em_score}, metric_score_list


class Retrieval_Recall(BaseMetric):
    r"""The recall of the top-k retreived passages, we measure if any of the passage contain the answer string. """
    metric_name = "retrieval_recall"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['metric_setting']['retrieval_recall_topk']

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        retrieve_docs = data.retrieval_result
        recall_score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")
            doc_list = [doc['contents'] for doc in doc_list[:self.topk]]
            hit_list = []
            for doc in doc_list:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(doc):
                        hit_list.append(True)
                        break
                else:
                    hit_list.append(False)
            score = 1 if any(hit_list) else 0
            recall_score_list.append(score)
        recall_score = sum(recall_score_list) / len(recall_score_list)

        return {f"retrieval_recall_top{self.topk}": recall_score}, recall_score_list


class Retrieval_Precision(BaseMetric):
    r"""The precision of the top-k retreived passages, we measure if any of the passage contain the answer string. """
    metric_name = "retrieval_precision"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['metric_setting']['retrieval_recall_topk']

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        retrieve_docs = data.retrieval_result
        precision_score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")
            doc_list = [doc['contents'] for doc in doc_list[:self.topk]]
            hit_list = []
            for doc in doc_list:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(doc):
                        hit_list.append(True)
                        break
                else:
                    hit_list.append(False)
            score = sum(hit_list) / len(hit_list)
            precision_score_list.append(score)
        precision_score = sum(precision_score_list) / len(precision_score_list)

        return {f"retrieval_precision_top{self.topk}": precision_score}, precision_score_list


class Rouge_Score(BaseMetric):
    metric_name = "rouge_score"

    def __init__(self, config):
        super().__init__(config)
        from rouge import Rouge
        self.scorer = Rouge()

    def calculate_rouge(self, pred, golden_answers):
        output = {}
        for answer in golden_answers:
            scores = self.scorer.get_scores(pred, answer)
            for key in ['rouge-1', 'rouge-2', 'rouge-l']:
                if key not in output:
                    output[key] = []
                output[key].append(scores[0][key]['f'])
        for k, v in output.items():
            output[k] = max(v)

        return output


class Rouge_1(Rouge_Score):
    metric_name = "rouge-1"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [self.calculate_rouge(pred, golden_answers)['rouge-1'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-1": score}, metric_score_list


class Rouge_2(Rouge_Score):
    metric_name = "rouge-2"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [self.calculate_rouge(pred, golden_answers)['rouge-2'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-2": score}, metric_score_list


class Rouge_L(Rouge_Score):
    metric_name = "rouge-l"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [self.calculate_rouge(pred, golden_answers)['rouge-l'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-l": score}, metric_score_list


class BLEU(BaseMetric):
    metric_name = "bleu"

    def __init__(self, config):
        super().__init__(config)
        from ._bleu import Tokenizer13a
        self.tokenizer = Tokenizer13a()
        self.max_order = config['metric_setting'].get('bleu_max_order', 4)
        self.smooth = config['metric_setting'].get('bleu_smooth', False)

    def calculate_metric(self, data):
        from ._bleu import compute_bleu
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        pred_list = [self.tokenizer(pred) for pred in pred_list]
        golden_answers_list = [[self.tokenizer(ans) for ans in golden_answers] for golden_answers in
                               golden_answers_list]
        score = compute_bleu(
            reference_corpus=golden_answers_list,
            translation_corpus=pred_list,
            max_order=self.max_order,
            smooth=self.smooth
        )
        (total_bleu, precisions, bp, ratio, translation_length, reference_length) = score

        score_list = []
        for pred, golden_answers in zip(pred_list, golden_answers_list):
            pred = [pred]
            golden_answers = [golden_answers]
            score = compute_bleu(
                reference_corpus=golden_answers_list,
                translation_corpus=pred_list,
                max_order=self.max_order,
                smooth=self.smooth
            )
            (bleu, precisions, bp, ratio, translation_length, reference_length) = score
            score_list.append(bleu)

        return {"bleu": total_bleu}, score_list


class CountToken(BaseMetric):
    metric_name = "input_tokens"

    def __init__(self, config):
        super().__init__(config)
        tokenizer_name = config['metric_setting'].get('tokenizer_name', None)
        is_hf_tokenizer = True
        from flashrag.utils.constants import OPENAI_MODEL_DICT
        if tokenizer_name is None or tokenizer_name in OPENAI_MODEL_DICT:
            # use gpt4 tokenizer
            import tiktoken
            if tokenizer_name is None:
                tokenizer_name = 'gpt-4'
            tokenizer = tiktoken.encoding_for_model(tokenizer_name)
            is_hf_tokenizer = False
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer

    def calculate_metric(self, data):
        input_prompts = data.prompt
        if self.is_hf_tokenizer:
            token_counts = [len(self.tokenizer.tokenize(text)) for text in input_prompts]
        else:
            token_counts = [len(self.tokenizer.encode(text)) for text in input_prompts]
        avg_tokens = sum(token_counts) / len(token_counts)

        return {"avg_input_tokens": avg_tokens}, token_counts


# ============ 以下为新增：中文友好的检索层指标 ============

class HitK(BaseMetric):
    """Hit@K：Top-K 检索文档中是否存在与答案高度相关的文档。

    匹配方式：字符重叠率 >= 阈值 则视为 hit。
    不要求答案整串出现在文档中。
    """
    metric_name = "hit_k"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['metric_setting']['hit_topk']
        # 重叠率阈值，默认 0.6（答案中 60% 的字符出现在文档中即算命中）
        self.threshold = config['metric_setting'].get('hit_threshold', 0.6)

    def calculate_metric(self, data):
        from hwrag.evaluator.utils import char_overlap_ratio
        golden_answers_list = data.golden_answers
        retrieve_docs = data.retrieval_result
        score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            doc_texts = [doc['contents'] for doc in doc_list[:self.topk]]
            hit = False
            for doc_text in doc_texts:
                # 只要任一 golden_answer 与该文档的重叠率超过阈值，就算命中
                for ga in golden_answers:
                    if char_overlap_ratio(ga, doc_text) >= self.threshold:
                        hit = True
                        break
                if hit:
                    break
            score_list.append(1 if hit else 0)
        score = sum(score_list) / len(score_list)
        return {f"hit@{self.topk}": score}, score_list


class MRRK(BaseMetric):
    """MRR@K：Mean Reciprocal Rank，考虑命中文档的排序位置。

    匹配方式同 HitK，使用字符重叠率。
    第一个命中的文档排名越靠前，分数越高（1/rank）。
    """
    metric_name = "mrr_k"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['metric_setting']['mrr_topk']
        self.threshold = config['metric_setting'].get('mrr_threshold', 0.6)

    def calculate_metric(self, data):
        from hwrag.evaluator.utils import char_overlap_ratio
        golden_answers_list = data.golden_answers
        retrieve_docs = data.retrieval_result
        score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            doc_texts = [doc['contents'] for doc in doc_list[:self.topk]]
            rr = 0.0
            for rank, doc_text in enumerate(doc_texts, start=1):
                matched = False
                for ga in golden_answers:
                    if char_overlap_ratio(ga, doc_text) >= self.threshold:
                        matched = True
                        break
                if matched:
                    rr = 1.0 / rank
                    break
            score_list.append(rr)
        score = sum(score_list) / len(score_list)
        return {f"mrr@{self.topk}": score}, score_list


class NDCGK(BaseMetric):
    """NDCG@K：Normalized Discounted Cumulative Gain。

    用字符重叠率作为每篇文档的相关性分数（0~1），
    然后计算 DCG / ideal_DCG。
    比二值的 Hit/MRR 更精细，能区分"部分相关"和"高度相关"。
    """
    metric_name = "ndcg_k"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['metric_setting']['ndcg_topk']

    def calculate_metric(self, data):
        import math
        from hwrag.evaluator.utils import char_overlap_ratio
        golden_answers_list = data.golden_answers
        retrieve_docs = data.retrieval_result
        score_list = []

        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            doc_texts = [doc['contents'] for doc in doc_list[:self.topk]]

            # 计算每篇文档的相关性分数 = 与所有 golden_answer 的最大重叠率
            rel_scores = []
            for doc_text in doc_texts:
                max_overlap = max(
                    char_overlap_ratio(ga, doc_text) for ga in golden_answers
                )
                rel_scores.append(max_overlap)

            # DCG
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rel_scores))

            # Ideal DCG：相关性分数降序排列
            ideal_scores = sorted(rel_scores, reverse=True)
            idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_scores))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            score_list.append(ndcg)

        score = sum(score_list) / len(score_list)
        return {f"ndcg@{self.topk}": score}, score_list


# ============ 以下为新增：中文友好的生成层指标 ============

class CharF1(BaseMetric):
    """字符级 F1：将预测和标准答案拆成字符集合，计算 F1。

    原有 F1 按 whitespace 分词，中文句子不分词就是一整块，
    导致 precision/recall 总是 0 或 1，毫无区分度。
    字符级 F1 对中文更合理。

    注意：与 cn_metrics.py 的 ChineseF1Metric (cn_f1) 不同，
    后者用 2-gram + char 级取 max，本类只用 char 级 Counter 直接计算。
    """
    metric_name = "char_f1"

    def __init__(self, config):
        super().__init__(config)

    def _char_f1_single(self, prediction, golden_answers):
        from hwrag.evaluator.utils import normalize_chinese_answer
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]

        best_f1 = 0.0
        norm_pred = normalize_chinese_answer(prediction)

        for golden_answer in golden_answers:
            norm_gold = normalize_chinese_answer(golden_answer)

            if not norm_pred or not norm_gold:
                continue

            # 字符级 Counter
            from collections import Counter
            pred_counter = Counter(norm_pred)
            gold_counter = Counter(norm_gold)
            common = pred_counter & gold_counter
            num_same = sum(common.values())

            if num_same == 0:
                continue

            precision = num_same / sum(pred_counter.values())
            recall = num_same / sum(gold_counter.values())
            f1 = (2 * precision * recall) / (precision + recall)
            best_f1 = max(best_f1, f1)

        return best_f1

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = data.golden_answers
        score_list = [
            self._char_f1_single(pred, ga)
            for pred, ga in zip(pred_list, golden_answers_list)
        ]
        score = sum(score_list) / len(score_list)
        return {"char_f1": score}, score_list


class SemanticSimilarity(BaseMetric):
    """语义相似度：用嵌入模型计算预测与标准答案的余弦相似度。

    比 token/char 级指标更能捕捉语义等价：
    "机器学习" vs "ML" 字符级 F1=0，但语义相似度很高。
    需要配置 embedding 模型路径。
    """
    metric_name = "semantic_similarity"

    def __init__(self, config):
        super().__init__(config)
        model_path = config['metric_setting']['semantic_model_path']
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path)

    def calculate_metric(self, data):
        import numpy as np
        pred_list = data.pred
        golden_answers_list = data.golden_answers

        # 编码所有预测
        pred_embeddings = self.model.encode(
            pred_list, normalize_embeddings=True,
            show_progress_bar=False
        )

        score_list = []
        for pred_emb, golden_answers in zip(pred_embeddings, golden_answers_list):
            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]
            # 编码所有 golden_answers，取最高相似度
            gold_embeddings = self.model.encode(
                golden_answers, normalize_embeddings=True,
                show_progress_bar=False
            )
            # 余弦相似度（已 normalize，直接点积）
            similarities = np.dot(gold_embeddings, pred_emb)
            score_list.append(float(np.max(similarities)))

        score = sum(score_list) / len(score_list)
        return {"semantic_similarity": score}, score_list


class Faithfulness(BaseMetric):
    """忠实度：生成答案是否基于检索文档，而非幻觉。

    方法：用字符重叠率衡量答案中有多少内容可以被检索文档支撑。
    - 将 pred 逐字符与检索文档比对
    - 被文档"覆盖"的字符比例即为忠实度
    - 比调用 LLM 做 judge 轻量得多，适合批量评估
    """
    metric_name = "faithfulness"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['metric_setting'].get('faithfulness_topk', 3)

    def calculate_metric(self, data):
        from hwrag.evaluator.utils import normalize_chinese_answer
        from collections import Counter

        pred_list = data.pred
        retrieve_docs = data.retrieval_result
        score_list = []

        for pred, doc_list in zip(pred_list, retrieve_docs):
            norm_pred = normalize_chinese_answer(pred)
            if not norm_pred:
                score_list.append(0.0)
                continue

            # 合并 topk 文档的字符集
            doc_texts = [doc['contents'] for doc in doc_list[:self.topk]]
            merged_doc = "".join(doc_texts)
            norm_doc = normalize_chinese_answer(merged_doc)

            # 计算预测中每个字符被文档覆盖的比例
            doc_char_counter = Counter(norm_doc)
            pred_char_counter = Counter(norm_pred)

            covered = 0
            total = 0
            for ch, count in pred_char_counter.items():
                total += count
                covered += min(count, doc_char_counter.get(ch, 0))

            faithfulness_score = covered / total if total > 0 else 0.0
            score_list.append(faithfulness_score)

        score = sum(score_list) / len(score_list)
        return {"faithfulness": score}, score_list


class AnswerRelevancy(BaseMetric):
    """答案相关性：衡量生成答案是否在回答原始问题。

    与 SemanticSimilarity 的本质区别：
    - SemanticSimilarity: pred vs golden_answer → 答得对不对
    - AnswerRelevancy:    pred vs question      → 有没有答到问题上

    例如：问"什么是深度学习"，答"深度学习很费算力"
    - SemanticSimilarity 低（和标准答案不像）
    - AnswerRelevancy 高（确实在聊深度学习这个话题）

    反例：问"什么是深度学习"，答"量子计算利用量子力学..."
    - SemanticSimilarity 低
    - AnswerRelevancy 也低（答非所问）
    """
    metric_name = "answer_relevancy"

    def __init__(self, config):
        super().__init__(config)
        model_path = config['metric_setting']['relevancy_model_path']
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path)

    def calculate_metric(self, data):
        import numpy as np
        pred_list = data.pred
        question_list = data.question  # 比较对象是 question，不是 golden_answers

        pred_embeddings = self.model.encode(
            pred_list, normalize_embeddings=True,
            show_progress_bar=False
        )
        question_embeddings = self.model.encode(
            question_list, normalize_embeddings=True,
            show_progress_bar=False
        )

        score_list = []
        for pred_emb, q_emb in zip(pred_embeddings, question_embeddings):
            similarity = float(np.dot(pred_emb, q_emb))
            score_list.append(similarity)

        score = sum(score_list) / len(score_list)
        return {"answer_relevancy": score}, score_list