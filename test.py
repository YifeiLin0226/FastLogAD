from data.parser import Parser
from data.processor import LogDataProcessor
from data.dataset import HitAnomalyDataset
from torch.utils.data import DataLoader
from multiprocessing import set_start_method

from data.dataset import LogBertDataset
from train_test.trainer import LogBertTrainer
from train_test.tester import LogBertTester

indir = '/home/datasets/log_data/OpenStack/'
outdir = '/home/datasets/log_data/OpenStack/output/'


# parser = Parser(indir, outdir, 'OpenStack', 'drain', {'depth':4, 'st':0.5, 'maxChild':100})
# parser.parse()
indir  = '/home/datasets/log_data/HDFS' # The input directory of log file
outdir = '/home/datasets/log_data/HDFS/output'  # The output directory of parsing results
# log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format

indir = '/home/datasets/log_data/BGL'
outdir = '/home/datasets/log_data/BGL/output/'
log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>' # BGL log format
# Regular expression list for optional preprocessing (default: [])
# regex      = [
#     r'blk_(|-)[0-9]+' , # block id
#     r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
#     r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
# ]
st         = 0.3  # Similarity threshold
depth      = 3  # Depth of all leaf nodes
# parser = Parser(indir, outdir, 'BGL', 'drain', {'depth':depth, 'st':st})
# parser.parse()

# processor = LogDataProcessor(outdir, outdir, 'BGL', 'sliding', size_unit = "minutes", window_size = 5, step_size = 1, shuffle = True)
# processor.process(window_size = 128, min_seq = 10, seq_len = 512, adaptive_window= True)
options = {'dataset': 'BGL', 'data_dir': outdir, 'batch_size': 32, 'epochs': 200, 'save_dir': './saved_models', 'device': 'cuda'}
options['hidden'] = 256
options['layers'] = 4
options['attn_heads'] = 4

options["n_epochs_stop"] = 10
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00

options['mask_ratio'] = 0.5
options['max_len'] = 512
options['seq_len'] = 512

options['hypersphere_loss'] = True
options["hypersphere_loss_test"] = False
options["is_logkey"] = True
options["is_time"] = False
options['model_dir'] = options['save_dir'] + '/bert/'
options['model_path'] = options['model_dir'] + 'best_bert.pth'

options["gaussian_mean"] = 0
options["gaussian_std"] = 1
options["num_candidates"] = 15

trainer = LogBertTrainer(options)
trainer.train()
tester = LogBertTester(options)
tester.predict()


# dataset = HitAnomalyDataset('OpenStack', outdir)
# dataLoader = DataLoader(dataset, batch_size=1)
# iterator = iter(dataLoader)
# model = HitAnomaly(768, 128, 128, 4)
# x, y = next(iterator)
# print(x.shape, y.shape)
# x = model(x[0], y[0])
# print(x.shape)
# num_anomalies = 0
# for data in dataset:
#     if data[2] == 1:
#         num_anomalies += 1

# print(num_anomalies / len(dataset))
# options = {"embed_dim": 768, "hidden_dim": 128, "dff": 128, "heads": 4, "dataset": "OpenStack", "data_dir": outdir, 'device':'cuda', 'epochs': 100, 'save_dir': './saved_models'}
# trainer = HitAnomalyTrainer(options)
# trainer.train()

# tester = HitAnomalyTester(options)
# tester.test()