"""
@Project ：Insult_Recognition
@File ：dataset.py
@IDE ：PyCharm
@Author ：wujx
"""
import copy
import json
import os
import logging
import torch

from torch.utils.data import TensorDataset

from preprocess import word_segment

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single train/dev/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be specified
                for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary"""
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:

    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary"""
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class IllegalRecProcessor(object):
    """Process for the illegal text recognition data set"""

    def __init__(self, args):
        self.args = args
        self.labels = ['__label__0', '__label__1']

    @classmethod
    def _read_txt(cls, infile):
        lines = []
        with open(infile, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                lines.append(line.rstrip().split('\t'))
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets"""
        examples = []
        for i, line in enumerate(lines):
            guid = f"{set_type}-{str(i)}"
            text_a = ''.join(line[1].split())
            label = self.labels.index(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, mode):
        """

        :param mode: train, dev, test
        :return:
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        return self._create_examples(self._read_txt(os.path.join(self.args.data_dir, file_to_read)), mode)


def convert_examples_to_features(
        examples,
        max_seq_len,
        tokenizer,
        pad_token_segment_id=1,
        sequence_a_segment_id=1,
        add_sep_token=True,
):
    features = []
    for ex_ind, example in enumerate(examples):
        token = tokenizer.tokenize(example.text_a)
        token = ['[CLS]'] + token
        if add_sep_token:
            token += ['[SEP]']
        seq_len = len(token)
        token_type_ids = [sequence_a_segment_id] * seq_len
        token_ids = tokenizer.convert_tokens_to_ids(token)
        if seq_len < max_seq_len:
            mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
            token_ids += [0] * (max_seq_len - seq_len)
            token_type_ids += [pad_token_segment_id] * (max_seq_len - seq_len)
        else:
            mask = [1] * max_seq_len
            token_ids = token_ids[:max_seq_len]
            token_type_ids = token_type_ids[:max_seq_len]

        label_id = int(example.label)
        features.append(InputFeatures(input_ids=token_ids, attention_mask=mask,
                                      token_type_ids=token_type_ids, label_id=label_id))
    return features


def load_and_cache_examples(args, tokenizer, mode):
    """

    :param args:
    :param mode:
    :param tokenizer:
    :return:
    """
    processor = IllegalRecProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file {}".format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at {}".format(args.data_dir))
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, only train, dev, test is available")

        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer)
        logger.info("Saving features into cached file {}".format(cached_features_file))
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, label_ids)

    return dataset


def load_data(file_path):
    """

    :param file_path:
    :return:
    """
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = line.rstrip().split('\t')
            labels.append('__label__' + item[0])
            words = word_segment(item[1])
            texts.append(words)
    return texts, labels
