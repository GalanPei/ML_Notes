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
        self.token_freq = sorted(token_freq_count.items(), key=lambda x: - x[1])
        self.idx2token = ['<unk>'] + reserved_tokens
        self.token2idx = dict()
        for idx in range(len(self.idx2token)):
            self.token2idx[self.idx2token[idx]] = idx
        for token, freq in self.token_freq:
            if freq < min_freq:
                continue
            self.idx2token.append(token)
            self.token2idx[token] = idx + 1
            idx += 1


    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
