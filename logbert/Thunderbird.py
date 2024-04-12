from data.parser import Parser
from data.processor import LogDataProcessor
from data.dataset import HitAnomalyDataset
from torch.utils.data import DataLoader
from multiprocessing import set_start_method

from data.dataset import LogBertDataset
from train_test.trainer import LogBertTrainer
from train_test.tester import LogBertTester
from utils import seed_all

seed_all()
indir  = '/home/datasets/log_data/Thunderbird' # The input directory of log file
outdir = '/home/datasets/log_data/Thunderbird/output'  # The output directory of parsing results

st         = 0.3  # Similarity threshold
depth      = 3  # Depth of all leaf nodes

# parser = Parser(indir, outdir, 'Thunderbird', 'drain', {'depth':depth, 'st':st})
# parser.parse()

# processor = LogDataProcessor(outdir, outdir, 'Thunderbird', 'sliding', size_unit = 'minutes', window_size= 1, step_size= 0.5, shuffle = True)
# processor.process(n_train = 6000, window_size = 128, min_seq = 10, seq_len = 512, adaptive_window= True)

# processor = LogDataProcessor(outdir, outdir, 'Thunderbird', 'sliding', size_unit = 'minutes', window_size= 1, step_size= 0.5, shuffle = False)
# processor.process(window_size = 511, min_seq = 2, adaptive_window= False)

options = {'dataset': 'Thunderbird', 'data_dir': outdir, 'batch_size': 32, 'epochs': 200, 'save_dir': indir + '/saved_models', 'device': 'cuda'}
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

# trainer = LogBertTrainer(options)
# trainer.train()
# tester = LogBertTester(options)
# tester.predict()
for seed in range(42, 45):
    seed_all(seed)
    trainer = LogBertTrainer(options)
    trainer.train()
    del trainer
    tester = LogBertTester(options)
    tester.predict()
    del tester