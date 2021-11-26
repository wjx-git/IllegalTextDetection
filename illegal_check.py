"""
@Project ：illegal_context_recognition
@File ：illegal_check.py
@IDE ：PyCharm
@Author ：wujx
"""

from config import SystemConfig, FastTextConfig
from preprocess import TradToSimple
from ac import AhocorasickNer
from preprocess import word_segment

from models.fasttext_model import FastText
from train_roberta import predict


config = SystemConfig()
f2s = TradToSimple(config.trad2simple_file)
illegal_match = AhocorasickNer()
illegal_match.add_keywords(config.illegal_dicts_file)
suspected_illegal_match = AhocorasickNer()
suspected_illegal_match.add_keywords_by_file(config.suspected_illegal_dicts_file)

ft_config = FastTextConfig()
fasttext_model = FastText(ft_config, train=False)


def check(text):
    """
    检测文本中是否有违规内容
    :param text: str
    :return:bool, True: 存在违规内容，False:不存在违规内容
    """
    # 繁简转换
    text = f2s.transform_sentence(text)

    # 违规关键词匹配
    if illegal_match.match_results(text):
        return True

    # 疑似违规关键词匹配
    suspect_illegal = False
    if suspected_illegal_match.match_results(text):
        suspect_illegal = True

    # fasttext 分类
    fasttext_check = False
    words = word_segment(text)
    pred = fasttext_model.predict(' '.join(words))
    if pred == '__label__1':
        fasttext_check = True

    if suspect_illegal and fasttext_check:
        return True
    elif suspect_illegal or fasttext_check:
        return predict(text) == 1
    return False


if __name__ == '__main__':
    check('你这杂种肯定是个愚蠢的小学生')
