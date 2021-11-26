"""
@Project ：Insult_Recognition
@File ：config.py
@IDE ：PyCharm
@Author ：wujx
"""


class FastTextConfig:
    def __init__(self):
        self.train_file = 'data/train.txt'
        self.test_file = 'data/test.txt'
        self.dev_file = 'data/dev.txt'

        self.model_path = 'outputs/fasttext/fasttext_model.bin'

        self.lr = 0.5
        self.embedding_dim = 200
        self.epoch = 20
        self.ngrams = 2
        self.loss_function = 'softmax'

        self.minCount = 4
        self.bucket = 20000


class SystemConfig:
    def __init__(self):
        self.illegal_dicts_file = 'dicts/illegal.txt'
        self.suspected_illegal_dicts_file = 'dicts/suspected_illegal.txt'
        self.trad2simple_file = 'dicts/trad2simp.txt'
        self.illegal_char_split_file = 'dicts/illegal_char_split.txt'

