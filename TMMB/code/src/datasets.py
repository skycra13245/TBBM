import json, time, random, torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer
from tqdm import tqdm
import pickle as pkl
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'

# 统一token长度并转化为索引
def token_to_index(token, config): 
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    # print(f"Vocab size: {len(vocab)}")

    words_line = []
    if len(token) < config.pad_size:
        token.extend([PAD] * (config.pad_size - len(token)))
    else:
        token = token[:config.pad_size]

    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    return words_line

# 获取dirty_word的索引（以下四个函数）
def get_dirty_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)    
    dirty_words = list(data.values())
    return dirty_words

def get_all_dirty_words(base_path):
    all_dirty_words = []
    paths = ["general.json", "LGBT.json", "region.json", "sexism.json", "racism.json"]
    for i in paths:
        all_dirty_words.append(get_dirty_words(base_path + i))
    return all_dirty_words

def get_toxic_id(text_idx, toxic_ids, dirty_words, toxic_index):
    for dirty_word in dirty_words:
        for index, token in enumerate(text_idx):
            if token == 0:
                break
            if token != dirty_word[0]:
                continue
            else:
                flag = 1
                start_index = index
                end_index = index+1
                for i in range(len(dirty_word[1:])):
                    if end_index >= len(text_idx):  # 越界
                        flag = 0
                        break
                    if text_idx[end_index] != dirty_word[1+i]:
                        flag = 0
                        break
                    end_index += 1
            if flag:
                for dirty_index in range(start_index, end_index):
                    toxic_ids[dirty_index] = toxic_index
    # print(toxic_ids)
    return toxic_ids
       
def get_all_toxic_id(pad_size, text_idx, all_dirty_words):
    toxic_ids = [0 for i in range(pad_size)]
    for toxic_index, dirty_words in enumerate(all_dirty_words):
        toxic_ids = get_toxic_id(text_idx, toxic_ids, dirty_words, toxic_index+1)
    return toxic_ids

class Datasets(Dataset):
    '''
    The dataset based on Bert.
    '''
    def __init__(self, kwargs, data_name, add_special_tokens=True, not_test=True):
        self.kwargs = kwargs
        self.not_test = not_test
        self.data_name = data_name
        self.lexicon_base_path = kwargs.lexicon_path
        self.max_tok_len = kwargs.pad_size
        self.add_special_tokens = add_special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs.model_name)
        with open(data_name, 'r', encoding='utf-8') as f:
            self.data_file = json.load(f)
        self.preprocess_data()

    def preprocess_data(self):
        print('Preprocessing Data {} ...'.format(self.data_name))
        data_time_start=time.time()
        all_dirty_words = get_all_dirty_words(self.lexicon_base_path)
        for row in tqdm(self.data_file):
            ori_text = row['content']
            text = self.tokenizer(ori_text, add_special_tokens=self.add_special_tokens,
                                  max_length=int(self.max_tok_len), padding='max_length', truncation=True)
            # For BERT
            row['text_idx'] = text['input_ids']
            row['text_ids'] = text['token_type_ids']
            row['text_mask'] = text['attention_mask']
            row['toxic_ids'] = get_all_toxic_id(self.max_tok_len, row['text_idx'], all_dirty_words)

        data_time_end = time.time()
        print("... finished preprocessing cost {} ".format(data_time_end-data_time_start))

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        row = self.data_file[idx]
        sample = {
                    # For BERT
                    'text_idx': row['text_idx'], 'text_ids': row['text_ids'], 'text_mask': row['text_mask'], 'toxic_ids': row['toxic_ids'],
                    # For GloVe
                    'toxic': row["toxic_one_hot"], 'toxic_type': row["toxic_type_one_hot"], 'expression': row["expression_one_hot"], 'target': row["target"],
                    # Topic model
                    'pro_vec': row['pro_vec']
                }
        return sample


class Dataloader(DataLoader):
    '''
    A batch sampler of a dataset. 
    '''
    def __init__(self, data, batch_size, shuffle=True, SEED=0):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.SEED = SEED
        random.seed(self.SEED)

        self.indices = list(range(len(data))) 
        if shuffle:
            random.shuffle(self.indices) 
        self.batch_num = 0 

    def __len__(self):
        return int(len(self.data) / float(self.batch_size))

    def num_batches(self):
        return len(self.data) / float(self.batch_size)

    def __iter__(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.indices != []:
            idxs = self.indices[:self.batch_size]
            batch = [self.data.__getitem__(i) for i in idxs]
            self.indices = self.indices[self.batch_size:]
            return batch
        else:
            raise StopIteration

    def get(self):
        self.reset() 
        return self.__next__()

    def reset(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)

def to_tensor(batch):
    '''
    Convert a batch data into tensor
    '''
    args = {}

    # For BERT
    args['text_idx'] = torch.tensor([b['text_idx'] for b in batch])
    args['text_ids'] = torch.tensor([b['text_ids'] for b in batch])
    args['text_mask'] = torch.tensor([b['text_mask'] for b in batch])
    args['toxic_ids'] = torch.tensor([b['toxic_ids'] for b in batch])
    args['pro_vec'] = torch.tensor([b['pro_vec'] for b in batch])

    args['toxic'] = torch.tensor([b['toxic'] for b in batch])
    args['toxic_type'] = torch.tensor([b['toxic_type'] for b in batch])
    args['expression'] = torch.tensor([b['expression'] for b in batch])
    args['target'] = torch.tensor([b['target'] for b in batch])

    return args

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def convert_onehot(config, label):
    onehot_label = [0 for i in range(config.num_classes)]
    onehot_label[int(label)] = 1
    return onehot_label


