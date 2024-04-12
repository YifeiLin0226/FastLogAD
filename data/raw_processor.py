import re
import string
import os
import numpy as np
import json
from collections import Counter

class RawLogProcessor:

    def __init__(self, log_dir, dataset, window_type, size_unit, window_size, step_size, train_ratio = 0.4, valid_ratio = 0.1, shuffle = False):
        self.log_dir = log_dir
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.shuffle = shuffle

        self.log_file = None
        if dataset == "BGL":
            self.log_file = "BGL.log"
        elif dataset == "HDFS":
            self.log_file = "HDFS.log"
        elif dataset == "OpenStack":
            self.log_file = ["openstack_abnormal.log", "openstack_normal1.log", "openstack_normal2.log"]
        elif dataset == "Thunderbird":
            self.log_file = "Thunderbird_20M.log"
        else:
            raise NotImplementedError

        if window_type not in ['fixed', 'sliding', 'session']:
            raise ValueError(f'window_type must be one of fixed, sliding, or session, but got {window_type}')
        self.window_type = window_type

        if window_type == 'session' and dataset != 'HDFS':
            raise ValueError('session window is only supported for HDFS dataset')
        
        if dataset == 'HDFS' and window_type != 'session':
            raise ValueError('HDFS dataset only supports session window')

        if size_unit is None:
            if window_type != 'session':
                raise ValueError('size_unit must be specified for fixed or sliding window')
        elif size_unit not in ['minutes', 'logs']:
            raise ValueError(f'size_unit must be one of minutes or logs, but got {size_unit}')
        
        self.size_unit = size_unit
        self.window_size = window_size

        if self.window_type == 'sliding':
            if step_size is None:
                raise ValueError('step_size must be specified for sliding window')
            self.step_size = step_size

        self.log_path = os.path.join(self.log_dir, self.log_file)
        self.shuffle = shuffle
        
    
    def _sliding_window(self):
        step_size = self.step_size
        window_size = self.window_size
        sliding_func = self._sliding_window_in_minutes if self.size_unit == 'minutes' else self._sliding_window_in_logs
        if self.size_unit == 'minutes':
            step_size = step_size * 60
            window_size = window_size * 60
        
        if self.dataset == 'OpenStack':
            new_data = [sliding_func(data, window_size, step_size)]
    
    def clean(s):
        """ Preprocess log message
        Parameters
        ----------
        s: str, raw log message

        Returns
        -------
        str, preprocessed log message without number tokens and special characters
        """
        s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
        s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
        s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
        s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
        s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
        s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
        trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
        content = s.translate(trantab)
        s = " ".join([word.lower().strip() for word in content.strip().split()])
        return s
    
    def generate_train_valid(self, train):
        valid_len = int(len(train) * self.valid_ratio)
        np.random.seed(12)
        np.random.shuffle(train)
        valid = train[:valid_len]
        train = train[valid_len:]

        return train, valid
    

    def generate_train_valid_test(self, data, n_train = None):
        if self.shuffle:
            np.random.seed(12)
            np.random.shuffle(data)
        
        normal_data = [d for d in data if d["Label"] == 0]
        abnormal_data = [d for d in data if d["Label"] == 1]
        normal_len, abnormal_len = len(normal_data), len(abnormal_data)
        if n_train:
            train_len = n_train
        else:
            train_len = int(normal_len * self.train_ratio)
        
        train = normal_data[:train_len]
        test_normal = normal_data[train_len:]
        test_abnormal = abnormal_data

        train, valid = self.generate_train_valid(train)

        return train, valid, test_normal, test_abnormal


    def process(self, n_train = None):
        print(f"processing {self.dataset} raw dataset with {self.train_ratio} train ratio and {self.valid_ratio} valid ratio")
        if self.dataset == "BGL" or self.dataset == "Thunderbird":
            data = []
            with open(self.log_path, "r") as f:
                for line in f.readlines():
                    if line.strip() != "":
                        label = int(line.strip().split()[0] != "-")
                        content = " ".join(line.strip().split()[1:])
                        data.append({"Content": content, "Label": label})
       


        
    


    def write_to_file(self, data, name):
        with open(os.path.join(self.log_dir, 'raw', name + '.json'), "w") as f:
            json.dump(data, f)

    
    
        
        
        
        

        

            

