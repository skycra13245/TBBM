# coding: UTF-8
import torch
from os import path

class Config_base(object):
    """config"""

    def __init__(self, model_name, dataset):
        # path
        self.model_name = model_name
        self.train_path = path.dirname(path.dirname(__file__)) + '/' + dataset + '/data/train.json'
        self.dev_path = path.dirname(path.dirname(__file__)) + '/' + dataset + '/data/test.json'
        self.test_path = path.dirname(path.dirname(__file__)) + '/' + dataset + '/data/test.json'
        self.vocab_path = path.dirname(path.dirname(__file__)) + '/' + dataset + '/data/vocab.pkl'
        self.lexicon_path = path.dirname(
            path.dirname(__file__)) + '/' + dataset + '/lexicon/'
        self.result_path = path.dirname(path.dirname(__file__)) + '/' + dataset + '/result/'
        self.checkpoint_path = path.dirname(path.dirname(__file__)) + '/' + dataset + '/saved_dict/'
        self.data_path = self.checkpoint_path + '/data_updated.tar'

        # dataset
        self.seed = 1
        self.num_classes = 5
        self.pad_size = 80

        # model
        self.dropout = 0.3
        self.vocab_dim = 768
        self.fc_hidden_dim = 16

        # cnn
        self.input_channels = 768
        self.output_channels = 64 - 11
        self.kernel_sizes = [2, 3, 4]

        # lstm
        self.lstm_hidden_dim = 64
        # self.lstm_out_dim = 16

        # BiGRU
        self.gru_hidden_dim = 32
        self.bilstm_hidden_dim = 64

        # attention
        self.att_input_size = 2*self.gru_hidden_dim+12
        self.num_attention_heads = 4

        # train
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = 1e-5
        self.scheduler = False  # decay
        self.adversarial = False
        self.num_warm = 8
        self.num_epochs = 10  # epoch
        self.batch_size = 32  # batch_size

        # loss
        self.alpha1 = 0.5
        self.gamma1 = 4

        # evaluate
        self.threshold = 0.5  # Binary classification threshold
        self.score_key = "F1"  # Evaluation metrics


if __name__ == '__main__':
    config = Config_base("BERT", "SWSR")
    print(config.vocab_path)
    print(path.dirname(__file__))
    print(path.dirname(path.dirname(__file__)))