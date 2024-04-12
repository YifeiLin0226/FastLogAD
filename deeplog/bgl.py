from train_test.trainer import DeepLogTrainer
from train_test.tester import DeepLogTester
from data.processor import LogDataProcessor
from utils import seed_all

import torch

import gc

indir  = '/home/datasets/log_data/BGL' # The input directory of log file
outdir = '/home/datasets/log_data/BGL/output'  # The output directory of parsing results

# processor = LogDataProcessor(outdir, outdir, 'BGL', 'sliding', size_unit = "minutes", window_size = 5, step_size = 1, shuffle = False)
# processor.process(window_size = None, min_seq = 2, seq_len = None, adaptive_window= True)
options = {'dataset': 'BGL', 'data_dir': outdir, 'batch_size': 1024, 'epochs': 200, 'save_dir': indir + '/saved_models', 'device': 'cuda'}

options['hidden_size'] = 64
options['num_layers'] = 2
options["embedding_dim"] = 64
options["dropout"] = 0.1
options['k'] = 150

options["n_epochs_stop"] = 100
options["lr"] = 0.001
options['lr_step'] = (50, 80)
options['lr_decay_ratio'] = 0.1

options['model_dir'] = options['save_dir'] + '/deeplog/'
options['model_path'] = options['model_dir'] + 'best_deeplog.pth'

# seed_all()
# # trainer = DeepLogTrainer(options)
# # trainer.train()
# options['device'] = 'cpu'
# tester = DeepLogTester(options)
# tester.test()


for seed in range(42, 45):
    seed_all(seed)
    options['device'] = 'cuda'
    trainer = DeepLogTrainer(options)
    trainer.train()
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    options['device'] = 'cpu'
    tester = DeepLogTester(options)
    tester.test()
    del tester