"""
@Project ：illegal_context_recognition
@File ：ac.py
@IDE ：PyCharm
@Author ：wujx
"""
import ahocorasick


class AhocorasickNer:
    """
    AC自动机
    """
    def __init__(self):
        self.actree = ahocorasick.Automaton()

    def add_keywords_by_file(self, file):
        words = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                words.append(line.rstrip())

        for flag, word in enumerate(words):
            self.actree.add_word(word, (flag, word))
        self.actree.make_automaton()

    def add_keywords(self, words):
        for flag, word in enumerate(words):
            self.actree.add_word(word, (flag, word))
        self.actree.make_automaton()

    def match_results(self, sentence):
        ner_results = []
        # i的形式为(index1,(index2,word))
        # index1: 提取后的结果在sentence中的末尾索引
        # index2: 提取后的结果在self.actree中的索引
        for i in self.actree.iter(sentence):
            ner_results.append((i[0], i[1][1]))
        return ner_results
