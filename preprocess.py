"""
@Project ：Insult_Recognition
@File ：preprocess.py
@IDE ：PyCharm
@Author ：wujx
"""
import re
import os
import jieba
from pypinyin import pinyin, Style


p_en = re.compile('[a-zA-Z]')


class TradToSimple:
    """
    繁体转简体
    """
    def __init__(self, file):
        self.dicts = self.load_dicts(file)

    @staticmethod
    def load_dicts(file):
        dicts = {}
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                item = line.strip().split()
                if item[0] != item[1]:
                    dicts[item[0]] = item[1]
        return dicts

    def transform_char(self, char):
        """
        单个字转换
        :param char:
        :return:
        """
        return self.dicts.get(char, char)

    def transform_sentence(self, sentence):
        """
        句子转换
        :param sentence:
        :return:
        """
        words = []
        for w in sentence:
            words.append(self.dicts.get(w, w))
        return ''.join(words)


def char_segment(string):
    """
    把句子按字分开，中文按字分，英文按单词
    :param string:
    :return:
    """
    chars = []
    en_char = ''
    for token in string:
        if (u'\u0041' <= token <= u'\u005a') or (u'\u0061' <= token <= u'\u007a'):  # 英文
            if token.isupper():
                chars.append(en_char.lower())
                en_char = token
            else:
                en_char += token
        else:
            if en_char:
                chars.append(en_char.lower())
                en_char = ''
            chars.append(token)
    if en_char:
        chars.append(en_char.lower())
    return [w for w in chars if len(w.strip()) > 0]


def word_segment(string):
    """
    把句子按词分开，中文按词分，英文按单词
    :param string:
    :return:
    """
    words = []
    for token in jieba.cut(string):
        if re.search(p_en, token):
            words.extend(char_segment(token))
        else:
            words.append(token)
    return words


def user_vocab(args):
    # 统计需要添加的词典
    bert_vocab = set()
    with open(os.path.join(args.model_name_or_path, 'vocab.txt'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            bert_vocab.add(line.rstrip())

    # def load_dataset(file):
    #     lines = []
    #     with open(file, 'r', encoding='utf-8') as f:
    #         for line in f.readlines():
    #             string = line.rstrip().split('\t')[1]
    #             lines.append(''.join(string.split()))
    #     return lines
    #
    # train_data = load_dataset(os.path.join(args.data_dir, args.train_file))
    # dev_data = load_dataset(os.path.join(args.data_dir, args.dev_file))
    # test_data = load_dataset(os.path.join(args.data_dir, args.test_file))
    #
    # data = train_data + dev_data + test_data

    chars = set()
    with open('dicts/illegal_char_split.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = line.split()
            pc = pinyin(item[0], style=Style.NORMAL)[0][0]
            chars.add(pc)
            chars.add(item[1])
            chars.add(item[2])

    words = set()
    for string in chars:
        for char in char_segment(string):
            if char.strip() and len(char) <= 10 and char not in bert_vocab:
                words.add(char)
    words = list(words)
    return words

if __name__ == '__main__':
    print(char_segment('请看人儿XinShou123'))
    # print(word_segment('请看人儿XinShou123'))


    # with open('data/new_vocab.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(words))
