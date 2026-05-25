import re
import string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    # def remove_punc(text):
    #     exclude = set(string.punctuation)
    #     return "".join(ch for ch in text if ch not in exclude)

    def remove_punc(text):
        zh_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞 〟〰〾〿–—''‛""„‟…‧"
        exclude = set(string.punctuation + zh_punc)
        return "".join(ch for ch in text if ch not in exclude)


    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ============ 以下为新增内容 ============

# 中文标点集合
CN_PUNCTUATION = set("，。！？；：""''（）、—…《》【】·～「」『』〈〉")


def normalize_chinese_answer(s):
    """中文归一化：去标点、去空格、转小写。保留中文语义核心内容。"""
    s = s.lower()

    # 去除中文标点
    s = "".join(ch for ch in s if ch not in CN_PUNCTUATION)

    # 去除英文标点
    s = "".join(ch for ch in s if ch not in set(string.punctuation))

    # 去除多余空白
    s = "".join(s.split())

    return s


def char_overlap_ratio(golden_answer, doc_text):
    """计算 golden_answer 的字符在 doc_text 中的覆盖率（召回视角）。

    核心思路：不是要求答案整串出现在文档中，
    而是看答案中的关键字符有多少能在文档中找到。
    对中文来说这比子串匹配合理得多。

    例如：golden_answer="人工智能", doc_text="AI技术正推动产业升级"
    -> 匹配了 "人""工""智""能" 中出现在 doc 中的字符数 / 答案总字符数
    """
    norm_answer = normalize_chinese_answer(golden_answer)
    norm_doc = normalize_chinese_answer(doc_text)

    if not norm_answer:
        return 0.0

    # 用集合统计字符重叠
    answer_chars = set(norm_answer)
    doc_chars = set(norm_doc)
    overlap = answer_chars & doc_chars

    return len(overlap) / len(answer_chars)