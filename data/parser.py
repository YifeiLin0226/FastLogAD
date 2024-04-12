import os
import re
from logparser.Drain import LogParser as Drain
import pandas as pd
import numpy as np

_LOG_FORMAT = {'HDFS': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
               'BGL': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
               'OpenStack': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
               'Thunderbird': '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
               }


class Parser:
    def __init__(self, input_dir : str, output_dir : str, dataset : str, parser_name : str, config):
        self.input_dir = input_dir
        assert os.path.isdir(self.input_dir)
        self.dataset = dataset

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
        
        self.output_dir = output_dir
        assert os.path.isdir(self.output_dir)
        self.parser_name = parser_name
        if parser_name.lower() == "drain":
            self.parser = Drain(log_format = _LOG_FORMAT[dataset], indir = self.input_dir, outdir = self.output_dir, **config)
        else:
            raise NotImplementedError
        

    def parse(self):
        if self.dataset == "OpenStack":
            length_index = [0]
            concat_log_file = os.path.join(self.input_dir, "OpenStack.log")
            with open(concat_log_file, "w") as concat_log:
                for log_file in self.log_file:
                    with open(os.path.join(self.input_dir, log_file)) as f:
                        lines = f.readlines()
                        length_index.append(len(lines))
                        concat_log.writelines(lines)
            
            self.parser.parse("OpenStack.log")
            os.remove(concat_log_file)

            # get Timestamp and DeltaT
            df = pd.read_csv(os.path.join(self.output_dir, "OpenStack.log_structured.csv"))
            df['Timestamp'] = pd.to_datetime(df['Date'] + '-' + df['Time']).values.astype(np.int64) // 10 ** 9
            df['Datetime'] = pd.to_datetime(df['Date'] + '-' + df['Time'])
            df['DeltaT'] = df['Datetime'].diff() / np.timedelta64(1, 's')
            df['DeltaT'] = df['DeltaT'].fillna(0)
            df = df[['Timestamp', 'Datetime', 'DeltaT', 'EventId', 'EventTemplate', 'ParameterList']]
            df.to_csv(os.path.join(self.output_dir, "OpenStack.log_structured.csv"), index = False)

            
            structure_files = ["openstack_abnormal.log_structured.csv", "openstack_normal1.log_structured.csv", "openstack_normal2.log_structured.csv"]
            labels = [1, 0, 0]
            df = pd.read_csv(os.path.join(self.output_dir, "OpenStack.log_structured.csv"))
            for i in range(len(self.log_file)):
                df_log = df.iloc[length_index[i]:length_index[i+1]]
                df_log['Label'] = labels[i]
                df_log.to_csv(os.path.join(self.output_dir, structure_files[i]), index = False)

        
           
            
        else:
            self.parser.parse(self.log_file)

            if self.dataset == 'HDFS':
                df = pd.read_csv(os.path.join(self.output_dir, "HDFS.log_structured.csv"), dtype = {'Date': str, 'Time': str})
                # df['Date'] = df['Date'].apply(lambda x: '0' + x if len(x) == 5 else x)
                df['Datetime'] = pd.to_datetime(df['Date'] + '-' + df['Time'], format='%y%m%d-%H%M%S')
                df['Timestamp'] = df['Datetime'].values.astype(np.int64) // 10 ** 9
                df['DeltaT'] = df['Datetime'].diff() / np.timedelta64(1, 's')
                df['DeltaT'] = df['DeltaT'].fillna(0)
                
                label_df = pd.read_csv(os.path.join(self.input_dir, "preprocessed", "anomaly_label.csv"))
                label_df['Label'] = label_df['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)
                label_dict = dict(zip(label_df['BlockId'], label_df['Label']))
                df['BlockId'] = df['Content'].apply(lambda x: re.findall(r'(blk_-?\d+)', x)[0])
                df['Label'] = df['BlockId'].apply(lambda x: label_dict.get(x))
                
                df = df[['BlockId', 'Timestamp', 'Datetime', 'DeltaT', 'Label', 'EventId', 'EventTemplate', 'ParameterList', 'Content']]
                df.to_csv(os.path.join(self.output_dir, "HDFS.log_structured.csv"), index = False)
            
            elif self.dataset == 'BGL':
                df = pd.read_csv(os.path.join(self.output_dir, "BGL.log_structured.csv"))
                df['Datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
                df['Timestamp'] = df['Datetime'].values.astype(np.int64) // 10 ** 9
                df['DeltaT'] = df['Datetime'].diff() / np.timedelta64(1, 's')
                df['DeltaT'] = df['DeltaT'].fillna(0)
                df['Label'] = df['Label'].apply(lambda x: int(x != '-'))
                df = df[['Timestamp', 'Datetime', 'DeltaT', 'Label', 'EventId', 'EventTemplate', 'ParameterList', 'Content']]

                df.to_csv(os.path.join(self.output_dir, "BGL.log_structured.csv"), index = False)


            elif self.dataset == "Thunderbird":
                df = pd.read_csv(os.path.join(self.output_dir, "Thunderbird_20M.log_structured.csv"))
                df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
                df['Timestamp'] = df['Datetime'].values.astype(np.int64) // 10 ** 9
                df['DeltaT'] = df['Datetime'].diff() / np.timedelta64(1, 's')
                df['DeltaT'] = df['DeltaT'].fillna(0)
                df['Label'] = df['Label'].apply(lambda x: int(x != '-'))
                df = df[['Timestamp', 'Datetime', 'DeltaT', 'Label', 'EventId', 'EventTemplate', 'ParameterList', 'Content']]
                df.to_csv(os.path.join(self.output_dir, "Thunderbird_20M.log_structured.csv"), index = False)
                   

