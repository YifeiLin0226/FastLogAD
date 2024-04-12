from train_test.trainer import DeepLogTrainer
from train_test.tester import DeepLogTester
from data.parser import Parser
from data.processor import LogDataProcessor
from utils import seed_all
from data.dataset import DeepLogDataset

from torch.utils.data import DataLoader

indir  = '/home/datasets/log_data/HDFS' # The input directory of log file
outdir = '/home/datasets/log_data/HDFS/output'  # The output directory of parsing results

st         = 0.5  # Similarity threshold
depth      = 5  # Depth of all leaf nodes

# parser = Parser(indir, outdir, 'HDFS', 'drain', {'depth':depth, 'st':st})
# parser.parse()

# processor = LogDataProcessor(outdir, outdir, 'HDFS', 'session', shuffle = False)
# processor.process(n_train = 4855, min_seq = 2, adaptive_window= True)
# processor.process(window_size = None, min_seq = 2, seq_len = None, adaptive_window= True)

options = {'dataset': 'HDFS', 'data_dir': outdir, 'batch_size': 2048, 'epochs': 100, 'save_dir': indir + '/saved_models', 'device': 'cuda'}

options['hidden_size'] = 64
options['num_layers'] = 2
options["embedding_dim"] = 64
options["dropout"] = 0.1
options['k'] = 6

options["n_epochs_stop"] = 100
options["lr"] = 0.001
options['lr_step'] = (50, 80)
options['lr_decay_ratio'] = 0.1

options['model_dir'] = options['save_dir'] + '/deeplog/'
options['model_path'] = options['model_dir'] + 'best_deeplog.pth'

# seed_all()
# trainer = DeepLogTrainer(options)
# trainer.train()

# tester = DeepLogTester(options)
# tester.test()

for seed in range(42, 45):
    seed_all(seed)
    trainer = DeepLogTrainer(options)
    trainer.train()
    del trainer
    tester = DeepLogTester(options)
    tester.test()
    del tester

