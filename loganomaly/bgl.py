from train_test.trainer import LogAnomalyTrainer
from train_test.tester import LogAnomalyTester
from data.dataset import LogAnomalyDataset
from data.processor import LogDataProcessor
from utils import seed_all
from torch.utils.data import DataLoader

import gc
import torch

indir  = '/home/datasets/log_data/BGL' # The input directory of log file
outdir = '/home/datasets/log_data/BGL/output'  # The output directory of parsing results


options = {'dataset': 'BGL', 'data_dir': outdir, 'batch_size': 2048, 'epochs': 200, 'save_dir': indir + '/saved_models', 'device': 'cuda'}


options['hidden_size'] = 64
options['num_layers'] = 2
options["embedding_dim"] = 300
options['embedding_strategy'] = 'tfidf' 
options["dropout"] = 0.1
options['k'] = 120

options["n_epochs_stop"] = 30
options["lr"] = 0.001
options['lr_step'] = (50, 80, 150, 250)
options['lr_decay_ratio'] = 0.1

options['model_dir'] = options['save_dir'] + '/loganomaly/'
options['model_path'] = options['model_dir'] + 'best_loganomaly.pth'

# seed_all()
# trainer = LogAnomalyTrainer(options)
# trainer.train()
# # del trainer
# options['device'] = 'cpu'
# tester = LogAnomalyTester(options)
# tester.test()


for seed in range(42, 45):
    seed_all(seed)
    trainer = LogAnomalyTrainer(options)
    trainer.train()
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    tester = LogAnomalyTester(options)
    tester.test()
    del tester