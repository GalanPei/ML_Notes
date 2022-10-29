import torch
from torch import nn


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size,
                               num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * hidden_size, 2)

    def forward(self, inputs):
        """

        :param inputs: (batch_size, time_steps)
        :return:
        """
        embedding = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embedding)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=-1)
        outs = self.decoder(encoding)
        return outs


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_size, channel_size, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        if isinstance(channel_size, int):
            channel_size = [kernel_size]
        for channel, kernel in zip(channel_size, kernel_size):
            self.convs.append(nn.Conv1d(2 * embed_size, channel, kernel))
        self.dropout = nn.Dropout(.5)
        self.decoder = nn.Linear(sum(channel_size), 2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        embedding = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embedding = embedding.permute(0, 2, 1)
        encoding = torch.cat(
            [torch.squeeze(self.relu(self.pool(conv(embedding))), dim=-1) for conv in self.convs],
            dim=1
        )
        output = self.decoder(self.dropout(encoding))
        return output
