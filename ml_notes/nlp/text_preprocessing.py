from typing import Type
from d2l import torch as d2l
import collections
import re
import torch
import math
import random


class TextSet(object):
    def __init__(self, file_name=None, file_url=None) -> None:
        if not file_name and not file_url:
            raise ValueError("You should give a file name or file url")
        self.lines = []
        self.token_lines = []
        if file_name:
            with open(file_name, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
            self.lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower()
                          for line in lines]
            
    def tokenize(self, type="word"):
        if type == "word":
            self.token_lines = [line.split() for line in self.lines]
        elif type == "char":
            self.token_lines = [list(line) for line in self.lines]
        else:
            raise TypeError(f"Invalid type: {type}.")
            


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
        raise TypeError(f"Invalid token type: {token}")


def trans_corpus(tokens: list) -> list:
    """
    Transfer the given list of tokens to a 1d list which only contains
    simple tokens

    :param tokens: list of any dim which contains tokens
    :return: token_list: 1d token list
    """
    token_list = []
    for token in tokens:
        if isinstance(token, list):
            # If the element of tokens is a list, we deploy the function
            # with recursion
            token_list += trans_corpus(token)
        else:
            token_list.append(token)
    return token_list


class Vocab(object):
    def __init__(self, tokens, reserved_tokens: list = None, min_freq: int = 0) -> None:
        """

        :param tokens:
        :param reserved_tokens:
        :param min_freq:
        """
        if tokens is None:
            raise ValueError("The given token is empty!")
        else:
            self.tokens = tokens
        token_list = trans_corpus(tokens)
        # Store the tokens by the frequency
        token_freq_count = collections.Counter(token_list)
        self.token_freq = sorted(
            token_freq_count.items(), key=lambda x: - x[1])
        self.idx2token = ['<unk>'] + reserved_tokens
        self.token2idx = collections.defaultdict(int)
        for idx in range(len(self.idx2token)):
            self.token2idx[self.idx2token[idx]] = idx
        for token, freq in self.token_freq:
            if freq < min_freq:
                continue
            self.idx2token.append(token)
            self.token2idx[token] = idx + 1
            idx += 1

    def __getitem__(self, item):
        """
        Get the index of token(s)

        :param item: Given tokens. The type is either list/tuple or string
        :return: list or int
        """
        if isinstance(item, str):
            return self.token2idx.get(item, self.unknown)
        elif isinstance(item, (list, tuple)):
            return [self.token2idx.get(x, self.unknown) for x in item]
        else:
            raise TypeError(f"The given token(s) should be either a list(tuple) or a string, "
                            f"the given type is {type(item)}")

    def __len__(self):
        # Length of tokens
        return len(self.idx2token)

    def subsample(self, words, t: torch.float32 = 1e-4):
        def prob(_freq: torch.float32):
            return max(.0, 1 - math.sqrt(t / _freq))
        if isinstance(words, str):
            words = [words]
        # Transfer the words to a flatten list
        words = trans_corpus(words)
        total_freq = sum([x[1] for x in self.token_freq])
        return [word for word in words
                if self.token2idx[word] != self.unknown and random.uniform(0, 1) < prob(
                    self.token_freq[word] / total_freq)]

    @property
    def token_frequency(self) -> dict:
        # Catch the frequency of each token.
        # If the token is not given, return 0.
        _token_dict = collections.defaultdict(int)
        for token, freq in self.token_freq:
            _token_dict[token] = freq
        return _token_dict

    @property
    def unknown(self) -> int:
        # For the tokens which do not appear in the know list, we set
        # the index as 0
        return 0
