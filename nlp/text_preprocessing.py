from d2l import torch as d2l
import collections
import re


def read_time_machine():
    """Load the time machine dataset into a list of text lines."""
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                    '090b5e7e70c295757f55df93cb0a180b9691891a')
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符标记。"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知令牌类型：' + token)


def tokens_frequency(tokens) -> dict:
    """
    计算tokens列表中的token频率

    :param tokens:
    :return: token频率的哈希表
    """
    ret = dict()
    if not tokens:
        return ret
    if isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab(object):
    def __init__(self, tokens, reserved_tokens = None):
        if tokens is None:
            self.tokens = []
        else:
            self.tokens = tokens
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
