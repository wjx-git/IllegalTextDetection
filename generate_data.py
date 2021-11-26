"""
@Project ：Insult_Recognition
@File ：generate_data.py
@IDE ：PyCharm
@Author ：wujx
"""
import re
import csv
import random
from pypinyin import pinyin, Style
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi

from ac import AhocorasickNer
from preprocess import word_segment
from config import SystemConfig

hmmparams = DefaultHmmParams()


def non_insult_sample(file_path):
    """

    :param file_path:
    :return:
    """
    p1 = re.compile(r'\[.*?\]')
    p2 = re.compile('//')
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in csv.reader(f):
            sample = line[1]
            if re.search(p2, sample):
                continue
            sample = re.sub(p1, '', sample).strip()
            if sample:
                results.append(sample)
    return results


def insult_sample(file_path):
    """

    :param file_path:
    :return:
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            samples.append(line)
    return samples


def save_txt_file(data, file_path):
    """

    :param data:
    :param file_path:
    :return:
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(''.join(data))


def add_label(pos, neg):
    """
    添加标签
    :param pos:
    :param neg:
    :return:
    """
    samples = []
    for line in pos:
        words = word_segment(line)
        line = '__label__1' + '\t' + " ".join(words)
        samples.append(line)
    for line in neg:
        words = word_segment(line)
        line = '__label__0' + '\t' + " ".join(words) + '\n'
        samples.append(line)
    return samples


def generate_dataset(samples):
    """

    :param samples:
    :return:
    """
    random.shuffle(samples)
    nums = len(samples)
    train = samples[:int(nums * 0.7)]
    test = samples[int(nums * 0.7): int(nums * 0.9)]
    dev = samples[int(nums * 0.9):]

    save_txt_file(train, 'data/train_original.txt')
    save_txt_file(test, 'data/test_original.txt')
    save_txt_file(dev, 'data/dev_original.txt')


class ParameterError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class DataAugment:
    def __init__(self, config):
        self.dict = self._load_split_dict(config.illegal_char_split_file)
        self.need_split = self.dict.keys()
        self.irrelevant_char = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', ',', '.', '-', '?',
                                '!', '=', '+', '&', '$', '#', '@', '，', '。', '？', '！', '*']
        self.length = len(self.irrelevant_char)

        self.words_ac = AhocorasickNer()
        words = self.keywords(config.illegal_dicts_file) + self.keywords(config.suspected_illegal_dicts_file)
        self.words_ac.add_keywords(words)

        self.function = {'insert': self.insert_char,
                         'split': self.split_char,
                         'pinyin': self.pinyin_char,
                         'homophone': self.homophone_char}

    @staticmethod
    def keywords(file):
        """

        :param file:
        :return:
        """
        words = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                words.append(line.rstrip())
        return words

    @staticmethod
    def _load_need_split_char(file):
        """
        需要拆解的字
        :param file:
        :return:
        """
        needs = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                needs.append(line.rstrip())
        return set(needs)

    @ staticmethod
    def _load_split_dict(file):
        """
        加载拆字词典
        :param file:
        :return:
        """
        dicts = {}
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                item = line.rstrip().split('\t')
                if len(item) == 2:
                    dicts[item[0]] = item[1]
        return dicts

    @staticmethod
    def merge_index(indexes):
        """

        :param indexes:
        :return:
        """
        result = []
        indexes.append([float('inf'), float('inf')])
        beg, end = indexes[0][0], indexes[0][1]
        for ind in indexes[1:]:
            if ind[0] <= end:
                end = ind[1]
            else:
                result.append((beg, end))
                beg = ind[0]
                end = ind[1]
        return result

    def match_illegal_char(self, string):
        """

        :return:
        """
        match_res = self.words_ac.match_results(string)
        indexes = []
        for res in match_res:
            length = len(res[1])
            end = res[0]
            begin = end - length + 1
            indexes.append((begin, end))
        indexes = self.merge_index(indexes)
        return indexes

    def split_char(self, string):
        """
        单个句子拆字法增强
        :param string:
        :return:
        """
        res = []
        for word in string.split():
            if word in self.need_split:
                word = self.dict.get(word, word)
            res.append(word)
        return ' '.join(res)

    def insert_char(self, string):
        """
        单个句子插入无关字符
        :param string: 不含空格
        :return:
        """
        def random_string():
            ind = random.randint(0, self.length-1)
            char = self.irrelevant_char[ind]
            num = random.randint(0, 3)
            return char * num

        indexes = self.match_illegal_char(string)
        inserted_string = string
        for ind in indexes:
            res = ''
            s = string[ind[0]: ind[1]+1]
            for i in range(ind[0], ind[1]):
                rand_str = random_string()
                res += string[i] + rand_str
            res += string[ind[1]]
            inserted_string = inserted_string.replace(s, res)
        if len(inserted_string) != len(string):
            return inserted_string

        # 针对没匹配到违规词的样本
        times = random.randint(1, 3)
        inserted_string = string
        for i in range(times):
            ind = random.randint(0, len(inserted_string) - 1)
            rand_str = random_string()
            inserted_string = inserted_string[:ind] + rand_str + inserted_string[ind:]
        if len(inserted_string) != len(string):
            return inserted_string
        return False

    def pinyin_char(self, string):
        """

        :param string:
        :return:
        """
        indexes = self.match_illegal_char(string)
        inserted_string = string
        for ind in indexes:
            for i in range(ind[0], ind[1]):
                flag = random.randint(0, 1)
                if flag:
                    r = pinyin(inserted_string[i], style=Style.NORMAL)[0][0]
                    inserted_string = inserted_string[:i] + r + inserted_string[i+1:]
        if inserted_string != string:
            return inserted_string

        # 针对没有匹配到违规词的样本
        times = random.randint(0, 3)
        lg = len(inserted_string)
        for i in range(times):
            k = random.randint(0, lg-1)
            r = pinyin(inserted_string[k], style=Style.NORMAL)[0][0]
            inserted_string = inserted_string[:k] + r + inserted_string[k+1:]
        if inserted_string != string:
            return inserted_string
        return False

    def homophone_char(self, string):
        """
        同音字替换
        :param string:
        :return:
        """

        def is_all_chinese(strs):
            for _char in strs:
                if not '\u4e00' <= _char <= '\u9fa5':
                    return False
            return True

        indexes = self.match_illegal_char(string)
        inserted_string = string
        for ind in indexes:
            if is_all_chinese(string[ind[0]: ind[1]+1]):
                r = pinyin(string[ind[0]: ind[1]+1], style=Style.NORMAL)
                py = [p[0] for p in r]
                try:
                    result = viterbi(hmm_params=hmmparams, observations=py, path_num=2)
                    for item in result:
                        words = ''.join(item.path)
                        if words != string[ind[0]: ind[1]+1]:
                            inserted_string = inserted_string[:ind[0]] + words + inserted_string[ind[1]+1:]
                            break
                except:
                    pass
        if inserted_string != string:
            return inserted_string

        # 处理没有匹配到违规词的样本
        words = word_segment(string)
        times = random.randint(0, 2)
        for _ in range(times):
            ind = random.randint(0, len(words)-1)
            if is_all_chinese(words[ind]):
                r = pinyin(words[ind], style=Style.NORMAL)
                py = [p[0] for p in r]
                try:
                    result = viterbi(hmm_params=hmmparams, observations=py, path_num=2)
                    for item in result:
                        homo = ''.join(item.path)
                        if homo != words[ind]:
                            words[ind] = ''.join(item.path)
                            break
                except:
                    pass
        inserted_string = ''.join(words)
        if inserted_string != string:
            return inserted_string
        return False

    # def split_augment_samples(self, infile, outfile):
    #     """
    #     拆字法增强样本
    #     :param infile:
    #     :param outfile:
    #     :return:
    #     """
    #     samples = []
    #     with open(infile, 'r', encoding='utf-8') as f:
    #         for line in f.readlines():
    #             item = line.split('\t')
    #             sample = self.split_char(item[1])
    #             samples.append(item[0] + '\t' + sample)
    #     random.shuffle(samples)
    #     with open(outfile, 'w', encoding='utf-8') as f:
    #         f.write('\n'.join(samples))
    #
    # def insert_augment_samples(self, infile, outfile):
    #     """
    #     插入无关字符增强样本
    #     :param infile:
    #     :param outfile:
    #     :return:
    #     """
    #     samples = []
    #     with open(infile, 'r', encoding='utf-8') as f:
    #         for line in f.readlines():
    #             item = line.split('\t')
    #             sample = self.insert_char(''.join(item[1].split()))
    #             if sample:
    #                 sample = word_segment(sample)
    #                 samples.append(item[0] + '\t' + ' '.join(sample))
    #     random.shuffle(samples)
    #     with open(outfile, 'w', encoding='utf-8') as f:
    #         f.write('\n'.join(samples))
    #
    # def pinyin_augment_samples(self, infile, outfile):
    #     """
    #     拼音替代文字增强样本
    #     :param infile:
    #     :param outfile:
    #     :return:
    #     """
    #     samples = []
    #     with open(infile, 'r', encoding='utf-8') as f:
    #         for line in f.readlines():
    #             item = line.split('\t')
    #             sample = self.pinyin_char(''.join(item[1].split()))
    #             if sample:
    #                 sample = word_segment(sample)
    #                 samples.append(item[0] + '\t' + ' '.join(sample))
    #     random.shuffle(samples)
    #     with open(outfile, 'w', encoding='utf-8') as f:
    #         f.write('\n'.join(samples))
    #
    # def homophone_augment_samples(self, infile, outfile):
    #     """
    #     同音字替换样本增强
    #     :param infile:
    #     :param outfile:
    #     :return:
    #     """
    #     samples = []
    #     with open(infile, 'r', encoding='utf-8') as f:
    #         for line in f.readlines():
    #             item = line.split('\t')
    #             sample = self.homophone_char(''.join(item[1].split()))
    #             if sample:
    #                 sample = word_segment(sample)
    #                 samples.append(item[0] + '\t' + ' '.join(sample))
    #     random.shuffle(samples)
    #     with open(outfile, 'w', encoding='utf-8') as f:
    #         f.write('\n'.join(samples))

    def augment_samples(self, infile, outfile, method):
        """

        :param method: ('insert', 'split', 'pinyin', 'homophone')
        :param infile:
        :param outfile:
        :return:
        """
        if not method:
            raise ParameterError("请指定 method 参数值，"
                                 "从('insert', 'split', 'pinyin', 'homophone')中选择一个或多个，"
                                 "列表形式，如 method=['split', 'pinyin']")

        original_samples = []
        with open(infile, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                original_samples.append(line)

        samples = []
        for m in method:
            func = self.function.get(m)
            for line in original_samples:
                item = line.split('\t')
                sample = func(''.join(item[1].split()))
                if sample:
                    sample = word_segment(sample)
                    samples.append(item[0] + '\t' + ' '.join(sample))

        samples += original_samples
        random.shuffle(samples)
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write('\n'.join(samples))


if __name__ == '__main__':
    # ====================================================
    #                  样本增强
    # ====================================================
    train_file = 'data/train_original.txt'
    test_file = 'data/test_original.txt'
    dev_file = 'data/dev_original.txt'

    aug_train_file = 'data/train_augment.txt'
    aug_test_file = 'data/test_augment.txt'
    aug_dev_file = 'data/dev_augment.txt'

    # config = SystemConfig()
    # da = DataAugment(config)
    # da.augment_samples('insert', train_file, aug_train_file)
    # da.augment_samples('insert', test_file, aug_test_file)
    # da.augment_samples('insert', dev_file, aug_dev_file)

    # ====================================================
    #           原始训练数据生成
    # ====================================================
    # non_insult = non_insult_sample('D:\ProgramData\微博评论数据集/weibo_senti_100k.csv')
    # insult = insult_sample('D:\ProgramData\中文词库\Dirty/Insult.txt')
    # random.shuffle(non_insult)
    # data = add_label(insult, non_insult[:int(1.5 * len(insult))])
    # generate_dataset(data)

    # ====================================================
    #                  敏感词词典清理
    # ====================================================
    # file = 'dicts/辱骂类.txt'
    # words = []
    # with open(file, 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         item = line.split('=')
    #         if len(item) == 2:
    #             words.append(item[0])
    #             continue
    #         item = line.split('|')
    #         if len(item) == 2:
    #             words.append(item[0])
    #             continue
    #         words.append(line.strip())
    # words = list(set(words))
    # with open(file, 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(words))

    # file = 'dicts/porn.txt'
    # words = []
    # with open(file, 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         words.append(line.strip())
    # words = list(set(words))
    # with open(file, 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(words))

    # 判断人名
    # import jieba.posseg as pseg
    #
    #
    # def isname(string):
    #     pair_word_list = pseg.lcut(string)
    #     for eve_word, cixing in pair_word_list:
    #         if cixing == "nr":
    #             return True
    #     return False
    #
    # names = []
    # others = []
    # with open('dicts/political.txt', 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         if isname(line.rstrip()):
    #             names.append(line)
    #         else:
    #             others.append(line)
    #
    # names.append('==========')
    # with open('dicts/political.txt', 'w', encoding='utf-8') as f:
    #     f.write(''.join(names + others))

    # 拆字词典
    # dicts = {}
    # with open('dicts/chaizi-jt.txt', 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         item = line.rstrip().split('\t')
    #         char = item[0]
    #         if len(item) >= 3:
    #             dicts[char] = item[2]
    #         else:
    #             dicts[char] = item[1]
    # with open('dicts/chaizi-ft.txt', 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         item = line.rstrip().split('\t')
    #         char = item[0]
    #         if char not in dicts:
    #             if len(item) >= 3:
    #                 dicts[char] = item[2]
    #             else:
    #                 dicts[char] = item[1]
    #
    # split_char = []
    # for k, v in dicts.items():
    #     split_char.append(k + '\t' + v)
    # with open('dicts/split_char.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(split_char))

    # split_char = []
    # chars = set()
    # with open('dicts/illegal_char.txt', 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         chars.add(line.rstrip())
    #
    # with open('dicts/split_char.txt', 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         if line and line[0] in chars:
    #             split_char.append(line)
    #
    # with open('dicts/illegal_char_split.txt', 'w', encoding='utf-8') as f:
    #     f.write(''.join(split_char))

    # def is_all_chinese(strs):
    #     for _char in strs:
    #         if not '\u4e00' <= _char <= '\u9fa5':
    #             return False
    #     return True
    #
    # print(is_all_chinese('自治州sb'))

    # train_files = ['data/train_original.txt']
    # test_files = ['data/test_original.txt']
    # dev_files = ['data/dev_original.txt']
    # name = ['homophone', 'insert', 'pinyin', 'split']
    # for n in name:
    #     train_files.append('data/train_' + n + '_augment.txt')
    #     test_files.append('data/test_' + n + '_augment.txt')
    #     dev_files.append('data/dev_' + n + '_augment.txt')
    #
    # def merge_file(files, outfile):
    #     samples = []
    #     for file in files:
    #         with open(file, 'r', encoding='utf-8') as f:
    #             for line in f.readlines():
    #                 samples.append(line)
    #     random.shuffle(samples)
    #     with open(outfile, 'w', encoding='utf-8') as f:
    #         f.write(''.join(samples))
    #
    # merge_file(train_files, 'data/train.txt')
    # merge_file(test_files, 'data/test.txt')
    # merge_file(dev_files, 'data/dev.txt')

    def load(file):
        samples = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                samples.append(line)
        return samples

    def save(d, file):
        with open(file, 'w', encoding='utf-8') as f:
            f.write(''.join(d))

    train = load('data/train.txt')
    test = load('data/test.txt')
    dev = load('data/dev.txt')

    data = train + test + dev

    random.shuffle(data)

    train_d = data[:int(len(data) * 0.7)]
    dev_d = data[int(len(data)*0.7): int(len(data) * 0.8)]
    test_d = data[int(len(data) * 0.8):]

    save(train_d, 'data/train.txt')
    save(dev_d, 'data/dev.txt')
    save(test_d, 'data/test.txt')
