from transformers import BertModel
from .BERT import BertModel
import torch
import torch.nn as nn

class Bert_Layer(torch.nn.Module):
    def __init__(self, config):
        super(Bert_Layer, self).__init__()
        self.device = config.device
        # BERT
        self.bert_layer = BertModel.from_pretrained(config.model_name)
        self.dim = config.vocab_dim

    def forward(self, **kwargs):
        bert_output = self.bert_layer(input_ids=kwargs['text_idx'].to(self.device),
                                 token_type_ids=kwargs['text_ids'].to(self.device),
                                 attention_mask=kwargs['text_mask'].to(self.device),
                                 toxic_ids=kwargs["toxic_ids"].to(self.device))
        return bert_output[0], bert_output[1]


class TwoLayerFFNNLayer(torch.nn.Module):
    def __init__(self, config):
        super(TwoLayerFFNNLayer, self).__init__()
        self.output = config.dropout
        # self.input_dim = config.vocab_dim+2*config.gru_hidden_dim+11
        self.input_dim = config.att_input_size
        self.hidden_dim = config.fc_hidden_dim
        self.out_dim = config.num_classes
        self.dropout = nn.Dropout(config.dropout)
        self.model = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                   nn.Tanh(),
                                   nn.Linear(self.hidden_dim, self.out_dim))

    def forward(self, pooled_emb):
        pooled_emb = self.dropout(pooled_emb)
        return self.model(pooled_emb)

class AttentionModel(nn.Module):
    def __init__(self, config):
        super(AttentionModel, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=config.att_input_size, num_heads=config.num_attention_heads)

    def forward(self, input_vector):
        attention_output, _ = self.attention(input_vector, input_vector, input_vector)
        attended_vector = input_vector + attention_output

        return attended_vector

class MultiscaleCNNLayer(nn.Module):
    def __init__(self, config):
        super(MultiscaleCNNLayer, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(config.input_channels, config.output_channels, kernel_size=kernel_size) for kernel_size in config.kernel_sizes
        ])

    def forward(self, x):

        outputs = [conv_layer(x) for conv_layer in self.conv_layers]
        concatenated_output = torch.cat(outputs, dim=1)

        return concatenated_output

class LSTMLayer(nn.Module):
    def __init__(self, config):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=config.vocab_dim,
                            hidden_size=config.lstm_hidden_dim,
                            num_layers=2,
                            batch_first=True,
                            dropout=config.dropout,
                            bidirectional=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, pooled_emb):
        pooled_emb = self.dropout(pooled_emb)
        lstm_out, _ = self.lstm(pooled_emb)
        out = torch.mean(lstm_out, dim=1)
        return lstm_out

class BiGRULayer(nn.Module):
    def __init__(self, config):
        super(BiGRULayer, self).__init__()
        self.gru = nn.GRU(input_size=config.vocab_dim,
                          hidden_size=config.gru_hidden_dim,
                          num_layers=2,
                          batch_first=True,
                          dropout=config.dropout,
                          bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, pooled_emb):
        pooled_emb = self.dropout(pooled_emb)
        gru_out, _ = self.gru(pooled_emb)
        out = torch.mean(gru_out, dim=1)

        return gru_out