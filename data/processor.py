import os
from typing import Optional
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import sys
from statistics import median

class LogDataProcessor:
    """
    class for grouping parsed data into windows and splitting into train and test sets
    """
    def __init__(self, parsed_data_dir : str, output_dir : str, dataset : str, window_type : str, size_unit : Optional[str] = 'logs', window_size : int = 15, step_size : Optional[int] = None, train_ratio = 0.4, valid_ratio = 0.1, shuffle = False, supervised = False):
        """
        Parameters
        ----------
        parsed_data_dir : str
            directory containing parsed data
        dataset : str
            one of BGL, HDFS, or OpenStack
        window_type : str
            one of fixed, sliding, or session
        size_unit : str
            one of minutes or logs
        window_size : int
            size of window
        step_size : Optional[str], optional
            size of step when sliding window
        train_ratio : float, optional
            ratio of training data to total normal data
        shuffle : bool, optional
            whether to maintain chronological order of data
        supervised : bool, optional
            whether to use abnormal data for training
        """
        if not os.path.isdir(parsed_data_dir):
            raise ValueError(f'parsed_data_dir {parsed_data_dir} does not exist')
        self.parsed_data_dir = parsed_data_dir

        if not os.path.isdir(output_dir):
            raise ValueError(f'output_dir {output_dir} does not exist')
        self.output_dir = output_dir

        if dataset not in ['BGL', 'HDFS', 'OpenStack', 'Thunderbird']:
            raise ValueError(f'dataset must be one of BGL, HDFS, or OpenStack, but got {dataset}')
        self.dataset = dataset

        if window_type not in ['fixed', 'sliding', 'session']:
            raise ValueError(f'window_type must be one of fixed, sliding, or session, but got {window_type}')
        self.window_type = window_type

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
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

        self.shuffle = shuffle
        self.supervised = supervised
        
        # if not shuffle and self.window_type == 'session':
        #     raise ValueError('session window does not support shuffle')
        # if shuffle and self.dataset == 'OpenStack':
        #     raise ValueError('OpenStack dataset does not support shuffle')



    
    def _sliding_window(self):
        """
        splits data into windows of size window_size with step size step_size
        """
        df = None
        if self.dataset == 'BGL':
            df = pd.read_csv(os.path.join(self.parsed_data_dir, 'BGL.log_structured.csv'))
        elif self.dataset == 'HDFS':
            df = pd.read_csv(os.path.join(self.parsed_data_dir, 'HDFS.log_structured.csv'))
        elif self.dataset == 'OpenStack':
            data_files = ['openstack_abnormal.log_structured.csv', 'openstack_normal1.log_structured.csv', 'openstack_normal2.log_structured.csv']
            df = [pd.read_csv(os.path.join(self.parsed_data_dir, data_file)) for data_file in data_files]
            # if self.supervised:
            #     df = pd.concat(df, axis = 0)
            #     df = df.sort_values(by = ['Timestamp'])
        elif self.dataset == 'Thunderbird':
            df = pd.read_csv(os.path.join(self.parsed_data_dir, 'Thunderbird_20M.log_structured.csv'))
            
                

        step_size = self.step_size
        window_size = self.window_size
        sliding_func = self._sliding_window_in_minutes if self.size_unit == 'minutes' else self._sliding_window_in_logs
        if self.size_unit == 'minutes':
            step_size = step_size * 60
            window_size = window_size * 60

        if self.dataset == 'OpenStack':
            new_data = [sliding_func(data, window_size, step_size) for data in df]
        
        else:
            new_data = sliding_func(df, window_size, step_size)
        
        return new_data
        



            
    def _sliding_window_in_minutes(self, df, window_size, step_size):
        """
        splits data into windows of size window_size with step size step_size in minutes
        
        """
        log_size = len(df)
        label_data, param_data = df['Label'], df['ParameterList']
        logkey_data, logtemplate_data = df['EventId'], df['EventTemplate']
        time_data, deltaT_data = df['Timestamp'], df['DeltaT']
        # content = df['Content']
        new_data = []
        start_end_index_pair = set()

        start_time = time_data[0]
        end_time = start_time + window_size
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in time_data:
            if cur_time < end_time:
                end_index += 1
            else:
                break
    
        start_end_index_pair.add(tuple([start_index, end_index]))

        num_session = 1
        while end_index < log_size:
            
            start_time = start_time + step_size
            end_time = start_time + window_size
            if start_index == end_index:
                start_time = time_data[start_index]
                end_time = start_time + window_size
            i = start_index
            j = end_index
            while i < log_size:
                if time_data[i] < start_time:
                    i += 1
                else:
                    break
            
            while j < log_size:
                if time_data[j] < end_time:
                    j += 1
                else:
                    break
            
            start_index = i
            end_index = j
            # print(i, j)
            # print(time_data[start_index], time_data[end_index])
            # if start_index == end_index:
            #     sys.exit()
            
            if start_index != end_index:
                start_end_index_pair.add(tuple([start_index, end_index]))
            
            num_session += 1
            if num_session % 1000 == 0:
                print(f'process {num_session} time window', end='\r')

        
        for (start_index, end_index) in start_end_index_pair:
            dt = deltaT_data[start_index: end_index].values.tolist()
            dt[0] = 0
            new_data.append({
                'DeltaT': dt,
                'EventId': logkey_data[start_index: end_index].values.tolist(),
                'EventTemplate': logtemplate_data[start_index: end_index].values.tolist(),
                # 'Content': content[start_index: end_index].values.tolist(),
                'ParameterList': param_data[start_index: end_index].values.tolist(),
                'Label': label_data[start_index:end_index].values.tolist()
            })
        
        return new_data
    

    def _sliding_window_in_logs(self, df, window_size, step_size):
        """
        splits data into windows of size window_size with step size step_size in logs
        
        """
        log_size = len(df)
        label_data, param_data = df['Label'], df['ParameterList']
        logkey_data, logtemplate_data = df['EventId'], df['EventTemplate']
        time_data, deltaT_data = df['Timestamp'], df['DeltaT']
        content = df['Content']
        new_data = []
        start_end_index_pair = set()

        start_index = 0
        end_index = window_size

        while end_index < log_size:
            start_end_index_pair.add(tuple([start_index, end_index]))
            start_index += step_size
            end_index += step_size
        
        for (start_index, end_index) in start_end_index_pair:
            dt = deltaT_data[start_index: end_index].values.tolist()
            dt[0] = 0
            new_data.append({
                'DeltaT': dt,
                'EventId': logkey_data[start_index: end_index].values.tolist(),
                'EventTemplate': logtemplate_data[start_index: end_index].values.tolist(),
                # 'Content': content[start_index: end_index].values.tolist(),
                'ParameterList': param_data[start_index: end_index].values.tolist(),
                'Label': label_data[start_index:end_index].values.tolist()
            })
        
        return new_data


    def _fixed_window(self):
        """
        splits data into windows of size window_size
        """
        self.step_size = self.window_size
        return self._sliding_window()
    

    def _session_window(self):
        if self.dataset == "HDFS":
            df = pd.read_csv(os.path.join(self.parsed_data_dir, 'HDFS.log_structured.csv'))
        
        else:
            raise NotImplementedError("session windows only support HDFS dataset")
        # else: 
        #     data_files = ['openstack_abnormal.log_structured.csv', 'openstack_normal1.log_structured.csv', 'openstack_normal2.log_structured.csv']
        #     df = [pd.read_csv(os.path.join(self.parsed_data_dir, data_file)) for data_file in data_files]
        #     df = pd.concat(df, axis = 0)
        #     df = df.sort_values(by = ['Timestamp'])

        
        
        # blkId_data = df['BlockId']

        # new_data = []
        # for blkId in tqdm(blkId_data.unique()):
        #     blkId_df = df[df['BlockId'] == blkId]
        #     dt = blkId_df['DeltaT'].values.tolist()
        #     dt[0] = 0
        #     new_data.append({
        #         'BlockId': blkId,
        #         'DeltaT': dt,
        #         'EventId': blkId_df['EventId'].values.tolist(),
        #         'EventTemplate': blkId_df['EventTemplate'].values.tolist(),
        #         'ParameterList': blkId_df['ParameterList'].values.tolist(),
        #         'Label': max(blkId_df['Label'])
        #     })

        grouped = df.groupby('BlockId')
        new_data = []

        for blkId, group in tqdm(grouped):
            dt = group['DeltaT'].values
            dt[0] = 0
            new_data.append({
                'BlockId': blkId,
                'DeltaT': dt.tolist(),
                'EventId': group['EventId'].values.tolist(),
                'EventTemplate': group['EventTemplate'].values.tolist(),
                # 'Content': group['Content'].values.tolist(),
                'ParameterList': group['ParameterList'].values.tolist(),
                'Label': group['Label'].values.tolist()
            })
        
        return new_data


    def _generate_train_valid_test(self, data, n_train = None):
        """
        splits data into train, validation and test sets
        """
        np.random.seed(12)
        if self.shuffle:
            np.random.shuffle(data)
        
        normal_data = [d for d in data if max(d['Label']) == 0]
        abnormal_data = [d for d in data if max(d['Label']) == 1]
        normal_len, abnormal_len = len(normal_data), len(abnormal_data)
        print(f'normal data size: {normal_len}, abnormal data size: {abnormal_len}')
        if n_train:
            train_len = n_train
        else:
            train_len = int(normal_len * self.train_ratio)

        train = normal_data[:train_len]
        test_normal = normal_data[train_len:]
        test_abnormal = abnormal_data

        if self.supervised:
            train, test_abnormal = self._supervised_split(train, test_abnormal)

        train, valid = self._generate_train_valid(train)

        print(f'train size: {len(train)}, valid size: {len(valid)}, test_normal size: {len(test_normal)}, test_abnormal size: {len(test_abnormal)}')
        num_train_logs = sum([len(d['EventId']) for d in train])
        num_valid_logs = sum([len(d['EventId']) for d in valid])
        num_test_normal_logs = sum([len(d['EventId']) for d in test_normal])
        num_test_abnormal_logs = sum([len(d['EventId']) for d in test_abnormal])
        print(f'train logs: {num_train_logs}, valid logs: {num_valid_logs}, test_normal logs: {num_test_normal_logs}, test_abnormal logs: {num_test_abnormal_logs}')
        
        return train, valid, test_normal, test_abnormal
    
    def _generate_train_valid(self, train):
        """
        splits train data into train and validation sets
        """
        valid_len = int(len(train) * self.valid_ratio)
        np.random.seed(12)
        np.random.shuffle(train)
        valid = train[:valid_len]
        train = train[valid_len:]

        return train, valid
    
    def _supervised_split(self, train, test_abnormal):
        """
        add abnormal data to train set
        """
        train, test_abnormal = train + test_abnormal[:int(len(test_abnormal) * 0.5)], test_abnormal[int(len(test_abnormal) * 0.5):]
        return train, test_abnormal

    def _write_to_file(self, data, file_name):
        """
        writes data to file
        """
        with open(os.path.join(self.output_dir, file_name + '.json'), 'w') as f:
            json.dump(data, f)
    
    def _sequence_window(self, data, window_size, min_seq = None, seq_len = None, adaptive_window = False):
        """
        splits data into windows of size window_size
        """
        new_data = []
        for d in tqdm(data):
            if min_seq and len(d['EventId']) < min_seq:
                continue
            if seq_len:
                d = {'DeltaT': d['DeltaT'][:seq_len], 'EventId': d['EventId'][:seq_len], 'EventTemplate': d['EventTemplate'][:seq_len], 'ParameterList': d['ParameterList'][:seq_len], 'Label': d['Label'][:seq_len]}
            if adaptive_window:
                window_size = len(d['EventId'])

            for i in range(0, len(d['EventId']), window_size):
                if min_seq and len(d['EventId'][i: i + window_size]) < min_seq:
                    continue
                new_data.append({
                    'DeltaT': d['DeltaT'][i: i + window_size],
                    'EventId': d['EventId'][i: i + window_size],
                    'EventTemplate': d['EventTemplate'][i: i + window_size],
                    'ParameterList': d['ParameterList'][i: i + window_size],
                    'Label': d['Label'][i: i + window_size]
                })
        return new_data
            

    def process(self, n_train = None, window_size = None, min_seq = None, seq_len = None, adaptive_window = False):
        """
        processes data into train and test sets
        """
        print(f'processing {self.dataset} dataset with {self.window_type} window')
        new_data = None
        if self.window_type == 'fixed':
            new_data = self._fixed_window()
        elif self.window_type == 'sliding':
            new_data = self._sliding_window()
        elif self.window_type == 'session':
            new_data = self._session_window()
        
        print(f'splitting data into train and test sets with train ratio {self.train_ratio}')
        if self.dataset != "OpenStack":
            train, valid, test_normal, test_abnormal = self._generate_train_valid_test(new_data, n_train = n_train)
        else:
            assert type(new_data) == list
            assert len(new_data) == 3
            train, test_normal, test_abnormal = new_data[1], new_data[2], new_data[0]
            if self.supervised:
                train, test_abnormal = self._supervised_split(train, test_abnormal)
            train, valid = self._generate_train_valid(train)
        
        if window_size:
            train = self._sequence_window(train, window_size, min_seq = min_seq, seq_len = seq_len, adaptive_window = adaptive_window)
            valid = self._sequence_window(valid, window_size, min_seq = min_seq, seq_len = seq_len, adaptive_window = adaptive_window)
            test_normal = self._sequence_window(test_normal, window_size, min_seq = min_seq, seq_len = seq_len, adaptive_window = adaptive_window)
            test_abnormal = self._sequence_window(test_abnormal, window_size, min_seq = min_seq, seq_len = seq_len, adaptive_window = adaptive_window)

        print(f'writing train, valid, test_normal, and test_abnormal to file')
        print(f'After sequence window, train size: {len(train)}, valid size: {len(valid)}, test_normal size: {len(test_normal)}, test_abnormal size: {len(test_abnormal)}')
        self._write_to_file(train, 'train')
        self._write_to_file(valid, 'valid')
        self._write_to_file(test_normal, 'test_normal')
        self._write_to_file(test_abnormal, 'test_abnormal')


    

        