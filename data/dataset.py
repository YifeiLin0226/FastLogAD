from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm
import re
import copy
import numpy as np
import random
import os
import string
import pickle
import json
import sys
from math import ceil
from collections import defaultdict, Counter, Iterable
from transformers import BertTokenizer, BertModel


from .vocab import WordVocab, LSTMVocab


class LogDataset(Dataset):
    def __init__(self, dataset, data_dir, features = ['EventTemplate'], mode = 'train'):
        super(LogDataset, self).__init__()
        if dataset not in ['HDFS', 'BGL', 'OpenStack', 'Thunderbird']:
            raise NotImplementedError
        self.dataset = dataset
        self.data_dir = data_dir
        self.features = features
        if mode not in ['train', 'valid', 'test']:
            raise NotImplementedError
        self.mode = mode
        self.normal_len = 0
        self.anomaly_len = 0
        if mode != 'test':
            self.data_path = os.path.join(self.data_dir, mode + '.json')
            self.data = json.load(open(self.data_path, 'r'))
        else:
            self.data_path = [os.path.join(self.data_dir, 'test_normal.json'), os.path.join(self.data_dir, 'test_abnormal.json')]
            normal_data = json.load(open(self.data_path[0], 'r')) 
            anomaly_data = json.load(open(self.data_path[1], 'r'))
            self.normal_len = len(normal_data)
            self.anomaly_len = len(anomaly_data)
            self.data = normal_data + anomaly_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        result_data = {}
        for feature in self.features:
            result_data[feature] = data[feature]
        return result_data
    

class LogBertDataset(LogDataset):
    def __init__(self, dataset, data_dir, features = ['EventTemplate'], mode = 'train', mask_ratio = 0.15, seq_len = 512, load_vocab = True):
        super(LogBertDataset, self).__init__(dataset, data_dir, features, mode)
        self.mask_ratio = mask_ratio
        self.seq_len = seq_len
        self.vocab_path = os.path.join(self.data_dir, 'logbert', 'vocab.pkl')
        if load_vocab and os.path.exists(self.vocab_path):
            print("Loading Vocab...")
            self.vocab = WordVocab.load_vocab(self.vocab_path)
            print("VOCAB SIZE:", len(self.vocab))
            print("Loading Vocab Done!")
            
        else:
            os.makedirs(os.path.join(self.data_dir, 'logbert'), exist_ok=True)
            texts = []
            for data in self.data:
                for feature in self.features:
                    texts.append(data[feature])
            self.vocab = WordVocab(texts)
            print("VOCAB SIZE:", len(self.vocab))
            print("save vocab in", self.vocab_path)
            self.vocab.save_vocab(self.vocab_path)
            print("Building Vocab Done!")
        
        
    
    def random_item(self, k, t):
        tokens = list(k)
        output_label = []

        time_intervals = list(t)
        time_label = []

        for i, token in enumerate(tokens):
            time_int = time_intervals[i]
            prob = random.random()
            # replace 15% of tokens in a sequence to a masked token
            if prob < self.mask_ratio:
                # raise AttributeError("no mask in visualization")

                if self.mode == 'test':
                    tokens[i] = self.vocab.mask_index
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                    time_label.append(time_int)
                    time_intervals[i] = 0
                    continue

                prob /= self.mask_ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                time_intervals[i] = 0  # time mask value = 0
                time_label.append(time_int)

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
                time_label.append(0)

        return tokens, output_label, time_intervals, time_label
    

    
    def __getitem__(self, index):
        data = super().__getitem__(index)
        assert len(data.keys()) == 1, 'only support one feature'
        feature = list(data.keys())[0]
        k = data[feature]
        t = np.zeros(len(k))
        
        k, k_label, t, t_label = self.random_item(k, t)

        k = [self.vocab.sos_index] + k
        k_label = [self.vocab.pad_index] + k_label

        t = [0] + t
        t_label = [self.vocab.pad_index] + t_label

        return k, k_label, t, t_label
    
    def collate_fn(self, batch):
        lens = [len(seq[0]) for seq in batch]
        max_len = max(lens)
        output = defaultdict(list)
        if self.seq_len:
            max_len = min(max_len, self.seq_len)
    
        for seq in batch:
            bert_input = seq[0][: max_len]
            bert_label = seq[1][: max_len]
            time_input = seq[2][: max_len]
            time_label = seq[3][: max_len]

            padding = [self.vocab.pad_index for _ in range(max_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), time_input.extend(padding), time_label.extend(padding)

            time_input = np.array(time_input)[:, np.newaxis]
            output["bert_input"].append(bert_input)
            output["bert_label"].append(bert_label)
            output["time_input"].append(time_input)
            output["time_label"].append(time_label)
        
        output["bert_input"] = np.array(output["bert_input"])
        output["bert_label"] = np.array(output["bert_label"])
        output["time_input"] = np.array(output["time_input"])
        output["time_label"] = np.array(output["time_label"])
        output["bert_input"] = torch.tensor(output["bert_input"], dtype=torch.long)
        output["bert_label"] = torch.tensor(output["bert_label"], dtype=torch.long)
        output["time_input"] = torch.tensor(output["time_input"], dtype=torch.float)
        output["time_label"] = torch.tensor(output["time_label"], dtype=torch.float)


        return output
            

class HitAnomalyDataset(LogDataset):
    def __init__(self, dataset, data_dir, mode = 'train', load = True):
        super(HitAnomalyDataset, self).__init__(dataset, data_dir, ['EventTemplate', 'ParameterList', 'Label'], mode)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model = self.model.to(device)
        self.model.eval()
        self.len = len(self.data)
        self.file_dir = os.path.join(self.data_dir, 'hitAnomaly', mode)
        num_files = 0
        if os.path.exists(self.file_dir) and load:
            num_files = len(os.listdir(self.file_dir))
        if os.path.exists(self.file_dir) and load and num_files == self.len:
            print("Loading tokenized and embedded data...")
            
        else:
            print("Tokenization and Embedding...")
            os.makedirs(self.file_dir, exist_ok=True)
            for i, sample in enumerate(tqdm(self.data)):
                k = sample['EventTemplate']
                p = sample['ParameterList']
                k = self.tokenizer(k, return_tensors='pt', padding=True, truncation=True)
                p = self.tokenizer(p, return_tensors='pt', padding=True, truncation=True)
                with torch.no_grad():
                    k = self.model(k.input_ids.to(device), k.attention_mask.to(device))[0]
                    p = self.model(p.input_ids.to(device), p.attention_mask.to(device))[0]
                label = torch.tensor(sample['Label'], dtype=torch.float)
                torch.save({'EventTemplate': k.cpu(), 'ParameterList': p.cpu(), 'Label': label.cpu()}, os.path.join(self.file_dir, f'sample_{i}.pt'))
        print("Tokenization and Embedding Done!")
    


    
    def __getitem__(self, index):
        if index < 0 or index >= self.len:
            raise IndexError("Index out of range")
        
        sample = torch.load(os.path.join(self.file_dir, f'sample_{index}.pt'))

        return sample
       
    

    def __len__(self):
        return self.len




class ElectraDataset(LogDataset):
    def __init__(self, dataset, data_dir, features = ['EventTemplate'], mode = 'train', seq_len = 128, mask_num = 50, mask_ratio = 0.5, load_vocab = True, load_mask = True, rmd_loss_weight = 0.5):
        super(ElectraDataset, self).__init__(dataset, data_dir, features, mode)
        self.seq_len = seq_len
        self.rmd_loss_weight = rmd_loss_weight
        self.mask_num = mask_num
        # if rmd_loss_weight > 0:
        #     self.mask_dir = mask_dir
        #     self.mask_path = os.path.join(self.mask_dir, 'electra', f'mask_{seq_len}_{mask_num}_{mask_ratio}.pkl')
        self.mask_ratio = mask_ratio
        self.vocab_path = os.path.join(self.data_dir, 'electra', 'vocab.pkl')
        
        if load_vocab and os.path.exists(self.vocab_path):
            print("Loading Vocab...")
            self.vocab = WordVocab.load_vocab(self.vocab_path)
            print("VOCAB SIZE:", len(self.vocab))
            print("Loading Vocab Done!")
            
        else:
            if mode != 'train':
                raise NotImplementedError("Only support build vocab in train mode")
            os.makedirs(os.path.join(self.data_dir, 'electra'), exist_ok=True)
            texts = []
            for data in self.data:
                for feature in self.features:
                    texts.append(data[feature])
            self.vocab = WordVocab(texts)
            print("VOCAB SIZE:", len(self.vocab))
            print("save vocab in", self.vocab_path)
            self.vocab.save_vocab(self.vocab_path)
            print("Building Vocab Done!")
        
        # if mode != 'test' and self.rmd_loss_weight > 0:
        #     if load_mask and os.path.exists(self.mask_path):
        #         print("Loading Mask...")
        #         self.mask = pickle.load(open(self.mask_path, 'rb'))
        #         print("Loading Mask Done!")
        #     else:
        #         os.makedirs(os.path.join(self.mask_dir, 'electra'), exist_ok=True)
        #         self.mask = self.build_mask()
        #         print("save mask in", self.mask_path)
        #         pickle.dump(self.mask, open(self.mask_path, 'wb'))
        #         print("Building Mask Done!")
        # else:
        #     print("Using random mask")
    
    def build_mask(self):
        masks = []
        sample_num_mask = int(self.seq_len * self.mask_ratio)
        while len(masks) < self.mask_num:
            mask = [False for _ in range(self.seq_len)]
            while sum(mask) < sample_num_mask:
                mask_idx = random.randint(0, self.seq_len - 1)
                if mask[mask_idx]:
                    continue
                else:
                    mask[mask_idx] = True
            
            if mask not in masks:
                masks.append(mask)
        
        return masks
    
    def random_mask(self, k):
        # if self.rmd_loss_weight > 0:
        #     random_mask_cls = random.randint(0, self.mask_num - 1)
        #     random_mask = self.mask[random_mask_cls]
        #     labels = np.ones(len(k)) * -100
        #     for i in range(len(k)):
        #         id = self.vocab.stoi.get(k[i], self.vocab.unk_index)
        #         if random_mask[i] and id != self.vocab.unk_index:
        #             labels[i] = id
        #             k[i] = self.vocab.mask_index
        #         else:
        #             k[i] = id
        # else:
        random_mask_cls = 0
        labels = np.ones(len(k)) * -100
        mask_count = 0
        for i in range(len(k)):
            id = self.vocab.stoi.get(k[i], self.vocab.unk_index)
            if mask_count == int(self.mask_ratio * len(k)):
                k[i] = id
            else:
                prob = random.random()
                if prob < self.mask_ratio and id != self.vocab.unk_index:
                    labels[i] = id
                    k[i] = self.vocab.mask_index
                    mask_count += 1
                else:
                    k[i] = id
        

        return k, labels.tolist(), random_mask_cls
        
    
    def __getitem__(self, index):
        gt = self.data[index]['Label']
        data = super().__getitem__(index)
        assert len(data.keys()) == 1, 'only support one feature'
        feature = list(data.keys())[0]
        k = data[feature]
        attention_mask = np.ones(len(k)).tolist()
        if self.mode != 'train':
            k = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in k]
            if isinstance(gt, Iterable) and self.mode != 'test':
                return k, attention_mask, max(gt)
            else:
                return k, attention_mask, gt
        
        else:
            attention_mask = np.ones(len(k)).tolist()
            labels_normal = (np.ones(len(k)) * -100).tolist()
            k_normal = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in k]
            k, labels, _ = self.random_mask(k)

            return [[k_normal, attention_mask, labels_normal, 0], [k, attention_mask, labels, 1]]


    def collate_fn(self, batch):
        if self.mode == 'train':
            batch = [item for sublist in batch for item in sublist]

        lens = [len(seq[0]) for seq in batch]
        max_len = max(lens)
        output = defaultdict(list)
        if self.seq_len:
            max_len = min(max_len, self.seq_len)

        num_seq = 0
        for seq in batch:
            
            for i in range(0, ceil(len(seq[0]) / max_len), max_len):
                k = seq[0][i : i + max_len]
                attention_mask = seq[1][i : i + max_len]
                padding = [self.vocab.pad_index for _ in range(max_len - len(k))]
                k.extend(padding)
                attention_mask.extend([0 for _ in range(max_len - len(attention_mask))])


                if self.mode == 'train':
                    labels = seq[2][i : i + max_len]
                    labels_padding = [-100 for _ in range(max_len - len(labels))]
                    labels.extend(labels_padding)
                    output["labels"].append(labels)
                    gt = seq[3]
                    if np.all(labels == -100):
                        gt = 0
                    output["gt"].append(gt)
                else:
                    if self.mode == 'test':
                        gt = seq[2][i : i + max_len]
                        gt_padding = [0 for _ in range(max_len - len(gt))]
                        gt.extend(gt_padding)
                        output["gt"].append(gt)
                
                    else:
                        gt = seq[2]
                        output["gt"].append(gt)

                num_seq += 1
                
                output["k"].append(k)
                output["attention_mask"].append(attention_mask)
            
            output["num_seq"].append(num_seq)

                
            
        
        output["k"] = np.array(output["k"])
        output["k"] = torch.tensor(output["k"], dtype=torch.long)
        output["attention_mask"] = np.array(output["attention_mask"])
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        output["gt"] = np.array(output["gt"])
        output["gt"] = torch.tensor(output["gt"], dtype=torch.int)
        output["num_seq"] = np.array(output["num_seq"])
        output["num_seq"] = torch.tensor(output["num_seq"], dtype=torch.int)
        if self.mode == 'train':
            output["labels"] = np.array(output["labels"])
            output["labels"] = torch.tensor(output["labels"], dtype=torch.long)
        
        else:
            cls_token = torch.ones(output["k"].shape[0], 1, dtype=torch.long) * self.vocab.sos_index
            output["k"] = torch.cat([cls_token, output["k"]], dim = 1)
            output["attention_mask"] = torch.cat([torch.ones(output["attention_mask"].shape[0], 1, dtype=torch.long), output["attention_mask"]], dim = 1)
            if self.mode == 'test':
                output['gt'] = torch.cat([torch.zeros(output["gt"].shape[0], 1, dtype=torch.long), output["gt"]], dim = 1)
        
        return output
        
        
    
    
# class ElectraRawDataset:

#     def __init__(self, dataset, data_dir, mode = 'train', mask_ratio = 0.5):
#         self.dataset = dataset
#         self.data_dir = data_dir
#         self.mode = mode
#         self.mask_ratio = mask_ratio
#         if self.mode != 'test':
#             self.data_path = os.path.join(self.data_dir, mode + '.json')
#             self.data = json.load(open(self.data_path, 'r'))
#         else:
#             self.data_path = [os.path.join(self.data_dir, 'test_normal.json'), os.path.join(self.data_dir, 'test_abnormal.json')]
#             normal_data = json.load(open(os.path.join(self.data_dir, 'test_normal.json'), 'r'))
#             anomaly_data = json.load(open(os.path.join(self.data_dir, 'test_abnormal.json'), 'r'))
#             self.data = normal_data + anomaly_data

#         self.vocab_path = os.path.join(self.data_dir, 'vocab.txt')

#         if os.path.exists(self.vocab_path):
#             print("Loading Vocab...")
#             self.tokenizer = BertTokenizer.from_pretrained(self.vocab_path)
#             print("VOCAB SIZE:", len(self.tokenizer.vocab))
#             print("Loading Vocab Done!")
        
#         else:
#             if self.mode != 'train':
#                 raise NotImplementedError("Only support build vocab in train mode")
#             os.makedirs(os.path.join(self.data_dir, 'raw'), exist_ok=True)
#             counter = Counter()
#             for data in self.data:
#                 for log in data['Content']:
#                     counter.update(log.split())
            
#             vocab = [token for token, count in counter.items() if count >= 3]
#             vocab = ['<pad>', '<unk>', '<eos>', '<sos>', '<mask>'] + vocab
#             with open(self.vocab_path, 'w') as f:
#                 f.write('\n'.join(vocab))
            
#             self.tokenier = BertTokenizer.from_pretrained(self.vocab_path)
#             print("VOCAB SIZE:", len(vocab))
#             print("save vocab in", self.vocab_path)
#             print("Building Vocab Done!")
        
#         self.vocab = self.tokenizer.vocab

#     def __len__(self):
#         return len(self.data)
    

#     def random_mask(self, data):
#         labels = np.ones(len(data)) * -100
#         mask_count = 0
#         for i in range(len(data)):
#             id = self.vocab.stoi.get(data[i], self.vocab.unk_index)
#             if mask_count == int(self.mask_ratio * len(data)) or id in [self.vocab.sos_index, self.vocab.eos_index, self.vocab.pad_index]:
#                 data[i] = id
#             else:
#                 prob = random.random()
#                 if prob < self.mask_ratio:
#                     labels[i] = id
#                     data[i] = self.vocab.mask_index
#                     mask_count += 1
#                 else:
#                     data[i] = id

#         return data, labels.tolist()


#     def __getitem__(self, index):
#         data, gt = self.data[index]['Content'], self.data[index]['Label']
#         input_ids = []
#         data = ' [SEP] '.join(data)
#         data = self.clean(data)
#         print(data)
        
#         data = data['input_ids'][0].tolist()
#         attention_mask = np.ones(len(data)).tolist()
#         if self.mode == 'train':
#             data_normal = copy.deepcopy(data)
#             data_abnormal, labels = self.random_mask(data_normal)
#             data_normal = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in data_normal]

#             assert gt == 0

#             labels_normal = (np.ones(len(data_normal)) * -100).tolist()
#             return [[data_normal, attention_mask, labels_normal, 0], [data_abnormal, attention_mask, labels, 1]]
        
#         else:
#             data = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in data]
#             return data, attention_mask, gt
        
    
#     def collate_fn(self, batch):
#         if self.mode != 'test':
#             batch = [item for sublist in batch for item in sublist]
        
#         lens = [len(seq[0]) for seq in batch]
#         max_len = max(lens)
#         output = defaultdict(list)
#         for seq in batch:
#             k = seq[0][: max_len]
#             attention_mask = seq[1][: max_len]
#             padding = [self.vocab.pad_index for _ in range(max_len - len(k))]
#             k.extend(padding)
#             attention_mask.extend([0 for _ in range(max_len - len(attention_mask))])

#             if self.mode == 'train':
#                 labels = seq[2][: max_len]
#                 labels_padding = [-100 for _ in range(max_len - len(labels))]
#                 labels.extend(labels_padding)
#                 output["labels"].append(labels)
#                 gt = seq[3]
#                 output["gt"].append(gt)
#             else:
#                 gt = seq[2]
#                 output["gt"].append(gt)
            
#             output["k"].append(k)
#             output["attention_mask"].append(attention_mask)
        
#         output["k"] = np.array(output["k"])
#         output["k"] = torch.tensor(output["k"], dtype=torch.long)
#         output["attention_mask"] = np.array(output["attention_mask"])
#         output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
#         output["gt"] = np.array(output["gt"])
#         output["gt"] = torch.tensor(output["gt"], dtype=torch.int)
#         if self.mode == 'train':
#             output["labels"] = np.array(output["labels"])
#             output["labels"] = torch.tensor(output["labels"], dtype=torch.long)
            
        
#         else:
#             cls_token = torch.ones(output["k"].shape[0], 1, dtype=torch.long) * self.vocab.sos_index
#             output["k"] = torch.cat([cls_token, output["k"]], dim = 1)
#             output["attention_mask"] = torch.cat([torch.ones(output["attention_mask"].shape[0], 1, dtype=torch.long), output["attention_mask"]], dim = 1)
        
#         return output
            
            

        



class DeepLogDataset(LogDataset):
    def __init__(self, dataset, data_dir, features = ['EventTemplate'], mode = 'train',load_vocab = True, window_size = 10):
        super(DeepLogDataset, self).__init__(dataset, data_dir, features, mode)
        self.vocab_path = os.path.join(self.data_dir, 'deeplog', 'vocab.pkl')

        if load_vocab and os.path.exists(self.vocab_path):
            print("Loading Vocab...")
            self.vocab = LSTMVocab.load_vocab(self.vocab_path)
            print("VOCAB SIZE:", len(self.vocab))
            print("Loading Vocab Done!")
        
        else:
            if self.mode != 'train':
                raise NotImplementedError("Only support build vocab in train mode")
            os.makedirs(os.path.join(self.data_dir, 'deeplog'), exist_ok=True)
            texts = []
            for data in self.data:
                for feature in self.features:
                    texts.append(data[feature])
            self.vocab = LSTMVocab(texts)
            print("VOCAB SIZE:", len(self.vocab))
            print("save vocab in", self.vocab_path)
            self.vocab.save_vocab(self.vocab_path)
            print("Building Vocab Done!")
        
        self.window_size = window_size

        if mode != 'test':
            self.data = self.build_window(self.data)
    

    def build_window(self, data):
        result_data = []
        for sample in data:
            for i in range(len(sample['EventTemplate']) - self.window_size):
                if len(sample['EventTemplate'][i: i + self.window_size + 1]) >= 2:
                    result_data.append({'EventTemplate': sample['EventTemplate'][i: i + self.window_size + 1], 'Label': sample['Label'][i : i + self.window_size + 1]})
        
        return result_data
    
    def __getitem__(self, index):
        gt = self.data[index]['Label']
        data = super().__getitem__(index)
        assert len(data.keys()) == 1, 'only support one feature'
        feature = list(data.keys())[0]
        k = data[feature]
        k = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in k]
        k = torch.tensor(k, dtype=torch.long)
        if self.mode != 'test':
            return k[:-1], k[-1]

        else:
            if isinstance(gt, Iterable):
                return k, max(gt)
            else:
                return k, gt
            
    
    def collate_fn(self, batch):
        if self.mode != 'test':
            x = [seq[0] for seq in batch]
            x = pad_sequence(x, batch_first=True, padding_value=self.vocab.pad_index)
            next_log = [seq[1] for seq in batch]
            next_log = torch.tensor(next_log, dtype=torch.long)
            return x, next_log
        
        else:
            x = [seq[0] for seq in batch]
            x_seq_batch = []
            next_log_batch = []
            num_windows = []
            for sample in x:
                subwindows = 0
                for i in range(max(len(sample) - self.window_size, 1)):
                    if len(sample) <= self.window_size:
                        x_seq = sample[:-1]
                        next_log = sample[-1].item()
                    else:
                        x_seq = sample[i : i + self.window_size]
                        next_log = sample[i + self.window_size]
                    subwindows += 1
                    x_seq_batch.append(x_seq)
                    next_log_batch.append(next_log)

                num_windows.append(subwindows)
            
            x_seq_batch = pad_sequence(x_seq_batch, batch_first = True, padding_value = self.vocab.pad_index)
            next_log_batch = torch.tensor(next_log_batch, dtype = torch.long)
            # num_windows = torch.tensor(num_windows, dtype = torch.long)

            gt = [seq[1] for seq in batch]
            gt = torch.tensor(gt, dtype=torch.long)
            return x_seq_batch, next_log_batch, num_windows, gt
    


    

        
class LogAnomalyDataset(LogDataset):
    def __init__(self, dataset, data_dir, features = ['EventTemplate'], mode = 'train',load_vocab = True, embedding_strategy = 'tfidf', window_size = 10):
        super(LogAnomalyDataset, self).__init__(dataset, data_dir, features, mode)
        self.vocab_path = os.path.join(self.data_dir, 'loganomaly', 'vocab.pkl')
        self.window_size = window_size

        if load_vocab and os.path.exists(self.vocab_path):
            print("Loading Vocab...")
            self.vocab = LSTMVocab.load_vocab(self.vocab_path)
            print("VOCAB SIZE:", len(self.vocab))
            print("Loading Vocab Done!")
        
        else:
            if self.mode != 'train':
                raise NotImplementedError("Only support build vocab in train mode")
            os.makedirs(os.path.join(self.data_dir, 'loganomaly'), exist_ok=True)
            texts = []
            for data in self.data:
                for feature in self.features:
                    texts.append(data[feature])
            self.vocab = LSTMVocab(texts, semantics = True, emb_file = os.path.join(self.data_dir, f'{dataset}.log_embeddings_{embedding_strategy}.json'), embedding_dim = 300)
            print("VOCAB SIZE:", len(self.vocab))
            print("save vocab in", self.vocab_path)
            self.vocab.save_vocab(self.vocab_path)
            print("Building Vocab Done!")

        if mode != 'test':
            self.data = self.build_window(self.data)
        
        

        
        

    def build_window(self, data):
        result_data = []
        for sample in data:
            for i in range(0, len(sample['EventTemplate']) - self.window_size, 1):
                if len(sample['EventTemplate'][i: i + self.window_size + 1]) >= 2:
                    sequential_pattern = [self.vocab.get_event(template, True) for template in sample['EventTemplate'][i: i + self.window_size + 1]]
                    semantic_vector = [self.vocab.get_embedding(template) for template in sample['EventTemplate'][i: i + self.window_size + 1]]
                    quantitative_vector = [0] * len(self.vocab)
                    counter = Counter(sequential_pattern)
                    for key, value in counter.items():
                        quantitative_vector[key] = value
                    quantitative_vector[sequential_pattern[-1]] -= 1
                    quantitative_vector = torch.tensor(quantitative_vector, dtype=torch.float)
                    semantic_vector = torch.tensor(semantic_vector, dtype=torch.float)
                    sequential_pattern = torch.tensor(sequential_pattern, dtype=torch.long)

                    result_data.append({'sequential_pattern': sequential_pattern, 'semantic_vector': semantic_vector, 'quantitative_vector': quantitative_vector})
                    # result_data.append({'sequential_pattern': sequential_pattern, 'quantitative_vector': quantitative_vector})
        return result_data

    def build_test_window(self, data):
        result_data = []
        for sample in data:
            gt = sample['Label']
            if isinstance(gt, Iterable):
                gt = max(gt)
            
                
            sequential_pattern = [self.vocab.get_event(template, True) for template in sample['EventTemplate']]
            semantic_vector = [self.vocab.get_embedding(template) for template in sample['EventTemplate']]
            semantic_vector = torch.tensor(semantic_vector, dtype=torch.float)
            sequential_pattern = torch.tensor(sequential_pattern, dtype=torch.long)
            
            result_data.append({'sequential_pattern': sequential_pattern, 'semantic_vector': semantic_vector, 'gt': gt})
        
        return result_data





    
    def __getitem__(self, index):
        data = self.data[index]
        if self.mode != 'test':
            return data['semantic_vector'][:-1, :], data['quantitative_vector'], data['sequential_pattern'][-1]
            # return data['sequential_pattern'][:-1], data['quantitative_vector'], data['sequential_pattern'][-1]
        
        else:
            sequential_pattern = [self.vocab.get_event(template, True) for template in data['EventTemplate']]
            semantic_vector = [self.vocab.get_embedding(template) for template in data['EventTemplate']]
            sequential_pattern = torch.tensor(sequential_pattern, dtype=torch.long)
            semantic_vector = torch.tensor(semantic_vector, dtype=torch.float)
            gt = data['Label']
            if isinstance(gt, Iterable):
                gt = max(gt)
            return semantic_vector, sequential_pattern, gt
            # return sequential_pattern, gt
    

    def collate_fn(self, batch):
        if self.mode != 'test':
            # semantic_vector = [seq[0] for seq in batch]
            # semantic_vector = pad_sequence(semantic_vector, batch_first=True, padding_value= self.vocab.pad_index)
            sequential_pattern = [seq[0] for seq in batch]
            sequential_pattern = pad_sequence(sequential_pattern, batch_first=True, padding_value= self.vocab.pad_index)
            quantitative_vector = [seq[1] for seq in batch]
            quantitative_vector = torch.stack(quantitative_vector, dim=0)
            next_log = [seq[2] for seq in batch]
            next_log = torch.tensor(next_log, dtype=torch.long)
            # return semantic_vector, quantitative_vector, next_log
            return sequential_pattern, quantitative_vector, next_log
        
        else:
            semantic = [seq[0] for seq in batch]
            sequential = [seq[1] for seq in batch]
            x_sem_batch = []
            next_log_batch = []
            x_quan_batch = []
            num_windows = []
            for i in range(len(semantic)):
                subwindows = 0
                sem = semantic[i]
                seq = sequential[i]
                for j in range(0, max(len(sem) - self.window_size, 1), int(self.window_size / 2)):
                    if len(sem) <= self.window_size:
                        x_sem = sem[:-1]
                        x_seq = seq[:-1]
                        next_log = seq[-1].item()
                    else:
                        x_sem = sem[j : j + self.window_size]
                        x_seq = seq[j : j + self.window_size]
                        next_log = seq[j + self.window_size]
                    subwindows += 1
                    x_sem_batch.append(x_sem)
                    next_log_batch.append(next_log)

                    x_quan = torch.zeros(len(self.vocab))
                    counter = Counter(x_seq)
                    for k in range(len(x_quan)):
                        x_quan[k] = counter[k]
                    
                    x_quan_batch.append(x_quan)

                num_windows.append(subwindows)
            
            x_sem_batch = pad_sequence(x_sem_batch, batch_first = True, padding_value = self.vocab.pad_index)
            next_log_batch = torch.tensor(next_log_batch, dtype = torch.int8)
            x_quan_batch = torch.stack(x_quan_batch)


            gt = [seq[2] for seq in batch]
            gt = torch.tensor(gt, dtype=torch.int8)

            return x_sem_batch, x_quan_batch, next_log_batch, num_windows, gt

            # sequential = [seq[0] for seq in batch]
            # x_seq_batch = []
            # next_log_batch = []
            # x_quan_batch = []
            # num_windows = []
            # for i in range(len(sequential)):
            #     subwindows = 0
            #     seq = sequential[i]
            #     for i in range(0, max(len(seq) - self.window_size, 1), int(self.window_size)):
            #         if len(seq) <= self.window_size:
            #             x_seq = seq[:-1]
            #             next_log = seq[-1].item()
            #         else:
            #             x_seq = seq[i : i + self.window_size]
            #             next_log = seq[i + self.window_size]
            #         subwindows += 1
            #         x_seq_batch.append(x_seq)
            #         next_log_batch.append(next_log)

            #         x_quan = torch.zeros(len(self.vocab))
            #         counter = Counter(x_seq)
            #         for i in range(len(x_quan)):
            #             x_quan[i] = counter[i]
                    
            #         x_quan_batch.append(x_quan)

            #     num_windows.append(subwindows)

            # x_seq_batch = pad_sequence(x_seq_batch, batch_first = True, padding_value = self.vocab.pad_index)
            # next_log_batch = torch.tensor(next_log_batch, dtype = torch.int8)
            # x_quan_batch = torch.stack(x_quan_batch)

            # gt = [seq[1] for seq in batch]
            # gt = torch.tensor(gt, dtype=torch.int8)

            # return x_seq_batch, x_quan_batch, next_log_batch, num_windows, gt
            




class PLELogDataset(LogDataset):
    def __init__(self, dataset, data_dir, features = ['EventTemplate'], mode = 'train',load_vocab = True, embedding_strategy = 'tfidf', use_pseudo = True, tok_anomaly_ratio = 0.5, seq_len = 120):
        super(PLELogDataset, self).__init__(dataset, data_dir, features, mode)
        self.use_pseudo = use_pseudo
        self.tok_anomaly_ratio = tok_anomaly_ratio
        self.vocab_path = os.path.join(self.data_dir, 'plelog', 'vocab.pkl')
        
        if load_vocab and os.path.exists(self.vocab_path):
            print("Loading Vocab...")
            self.vocab = WordVocab.load_vocab(self.vocab_path)
            print("VOCAB SIZE:", len(self.vocab))
            print("Loading Vocab Done!")
        
        else:
            if self.mode != 'train':
                raise NotImplementedError("Only support build vocab in train mode")
            os.makedirs(os.path.join(self.data_dir, 'plelog'), exist_ok=True)
            texts = []
            for data in self.data:
                for feature in self.features:
                    texts.append(data[feature])
            self.vocab = WordVocab(texts, semantics = True, emb_file = os.path.join(self.data_dir, f'{dataset}.log_embeddings_{embedding_strategy}.json'), embedding_dim = 300)
            print("VOCAB SIZE:", len(self.vocab))
            print("save vocab in", self.vocab_path)
            self.vocab.save_vocab(self.vocab_path)
            print("Building Vocab Done!")
        

        if use_pseudo and mode == 'train':
            print("Building Pseudo Anomaly...")
            self.pseudo_anomaly_dict = self.vocab.build_pseudo_anomaly()
        
        # for i, data in enumerate(self.data):
        #     gt = data['Label']
        #     gt = gt[:seq_len]
        #     semantic_vector = [self.vocab.get_embedding(template) for template in data['EventTemplate']]
        #     semantic_vector = semantic_vector[:seq_len]
        #     semantic_vector = torch.tensor(semantic_vector, dtype=torch.float)

        #     if self.mode == 'train' and self.use_pseudo:
        #         template_vector = data['EventTemplate']
        #         template_vector = template_vector[:seq_len]
        #         anomaly_vector, anomaly_count = self.random_anomaly(template_vector)
        #         anomaly_vector = torch.tensor(anomaly_vector, dtype=torch.float)

        #         self.data[i] = {'semantic_vector': semantic_vector, 'anomaly_vector': anomaly_vector, 'Label': gt} if anomaly_count > 0 else {'semantic_vector': semantic_vector, 'Label': gt}
        #     else:
        #         self.data[i] = {'semantic_vector': semantic_vector, 'Label': gt}
        seq_data = []
        for i, data in enumerate(self.data):
            length = len(data['EventTemplate'])
            for j in range(0, max(1, length - seq_len + 1), seq_len):
                gt = data['Label']
                gt = gt[j:j+seq_len]
                semantic_vector = [self.vocab.get_embedding(template) for template in data['EventTemplate'][j:j+seq_len]]
                semantic_vector = torch.tensor(semantic_vector, dtype=torch.float)

                if self.mode == 'train' and self.use_pseudo:
                    template_vector = data['EventTemplate'][j:j+seq_len]
                    anomaly_vector, anomaly_count = self.random_anomaly(template_vector)
                    anomaly_vector = torch.tensor(anomaly_vector, dtype=torch.float)

                    seq_data.append({'semantic_vector': semantic_vector, 'anomaly_vector': anomaly_vector, 'Label': gt} if anomaly_count > 0 else {'semantic_vector': semantic_vector, 'Label': gt})
                else:
                    seq_data.append({'semantic_vector': semantic_vector, 'Label': gt})
            
        
        self.data = seq_data
        

        
    
    def random_anomaly(self, sequential_pattern):
        anomaly_vector = []
        anomaly_count = 0
        anomaly_total = int(len(sequential_pattern) * self.tok_anomaly_ratio)
        for i in range(len(sequential_pattern)):
            prob = random.random()
            if prob < self.tok_anomaly_ratio and anomaly_count < anomaly_total:
                index = self.vocab.stoi[sequential_pattern[i]]
                candidates = self.pseudo_anomaly_dict[index]
                if len(candidates) == 0:
                    anomaly_vector.append(self.vocab.get_embedding(sequential_pattern[i]))
                else:
                    anomaly_vector.append(self.vocab.get_embedding(self.vocab.itos[random.choice(candidates)]))
                    anomaly_count += 1
            else:
                anomaly_vector.append(self.vocab.get_embedding(sequential_pattern[i]))
        
        return anomaly_vector, anomaly_count
    
    def __getitem__(self, index):
        data = self.data[index]
        gt = data['Label']
        if isinstance(gt, Iterable):
            gt = max(gt)

        semantic_vector = data['semantic_vector']
        mask = torch.ones(semantic_vector.shape[0], dtype=torch.long)
        
        if self.mode == 'train' and self.use_pseudo:
            anomaly_vector = data.get('anomaly_vector', None)
            if anomaly_vector is None:
                return [[semantic_vector, mask, 0]]
            else:
                return [[semantic_vector, mask, 0], [anomaly_vector, mask, 1]]
            
        return semantic_vector, mask, gt





    def collate_fn(self, batch):
        if self.mode == 'train' and self.use_pseudo:
            batch = [item for sublist in batch for item in sublist]

        semantic_vector = [seq[0] for seq in batch]
        semantic_vector = pad_sequence(semantic_vector, batch_first=True, padding_value=0)
        mask = [seq[1] for seq in batch]
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        
        gt = [seq[2] for seq in batch]
        gt = torch.tensor(gt, dtype=torch.float)
        return semantic_vector, mask, gt
            
        
    

# class ElectraSemDataset(LogDataset):
#     def __init__(self, dataset, data_dir, features = ['EventTemplate'], mode = 'train', load_vocab = True, embedding_strategy = 'tfidf', use_pseudo = True, tok_anomaly_ratio = 0.5, seq_len = 120):
#         super(ElectraSemDataset, self).__init__(dataset, data_dir, features, mode)
#         self.seq_len = seq_len
#         self.vocab_path = os.path.join(self.data_dir, 'electra_sem', 'vocab.pkl')
        
#         if load_vocab and os.path.exists(self.vocab_path):
#             print("Loading Vocab...")
#             self.vocab = WordVocab.load_vocab(self.vocab_path)
#             print("VOCAB SIZE:", len(self.vocab))
#             print("Loading Vocab Done!")
        
#         else:
#             if self.mode != 'train':
#                 raise NotImplementedError("Only support build vocab in train mode")
#             os.makedirs(os.path.join(self.data_dir, 'electra_sem'), exist_ok=True)
#             texts = []
#             for data in self.data:
#                 for feature in self.features:
#                     texts.append(data[feature])
#             self.vocab = WordVocab(texts, semantics = True, emb_file = os.path.join(self.data_dir, f'{dataset}.log_embeddings_{embedding_strategy}.json'), embedding_dim = 300)
#             print("VOCAB SIZE:", len(self.vocab))
#             print("save vocab in", self.vocab_path)
#             self.vocab.save_vocab(self.vocab_path)
#             print("Building Vocab Done!")
        

#         if use_pseudo and mode == 'train':
#             print("Building Pseudo Anomaly...")
#             self.pseudo_anomaly_dict = self.vocab.build_pseudo_anomaly()
        

#         seq_data = []
#         for i, data in enumerate(self.data):
#             length = len(data['EventTemplate'])
#             for j in range(0, max(1, length - seq_len + 1), seq_len):
#                 gt = data['Label']
#                 gt = gt[j:j+seq_len]
#                 semantic_vector = [self.vocab.get_embedding(template) for template in data['EventTemplate'][j:j+seq_len]]
#                 semantic_vector = torch.tensor(semantic_vector, dtype=torch.float)

#                 if self.mode == 'train' and self.use_pseudo:
#                     template_vector = data['EventTemplate'][j:j+seq_len]
#                     anomaly_vector, token_label = self.random_anomaly(template_vector)
#                     anomaly_vector = torch.tensor(anomaly_vector, dtype=torch.float)

#                     seq_data.append({'semantic_vector': semantic_vector, 'anomaly_vector': anomaly_vector, 'Label': gt} if max(token_label) == 1 else {'semantic_vector': semantic_vector, 'Label': gt})
#                 else:
#                     seq_data.append({'semantic_vector': semantic_vector, 'Label': gt})
            
        
#         self.data = seq_data
    




#     def random_anomaly(self, sequential_pattern):
#         anomaly_vector = []
#         anomaly_count = 0
#         anomaly_total = int(len(sequential_pattern) * self.tok_anomaly_ratio)
#         token_label = [0] * len(sequential_pattern)
#         for i in range(len(sequential_pattern)):
#             prob = random.random()
#             if prob < self.tok_anomaly_ratio and anomaly_count < anomaly_total:
#                 index = self.vocab.stoi[sequential_pattern[i]]
#                 candidates = self.pseudo_anomaly_dict[index]
#                 if len(candidates) == 0:
#                     anomaly_vector.append(self.vocab.get_embedding(sequential_pattern[i]))
#                 else:
#                     anomaly_vector.append(self.vocab.get_embedding(self.vocab.itos[random.choice(candidates)]))
#                     token_label[i] = 1
#                     anomaly_count += 1
#             else:
#                 anomaly_vector.append(self.vocab.get_embedding(sequential_pattern[i]))
        
#         return anomaly_vector, token_label