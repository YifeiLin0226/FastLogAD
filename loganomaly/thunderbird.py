from train_test.trainer import LogAnomalyTrainer
from train_test.tester import LogAnomalyTester
from data.dataset import LogAnomalyDataset
from data.processor import LogDataProcessor
from utils import seed_all
from torch.utils.data import DataLoader

indir  = '/home/datasets/log_data/Thunderbird' # The input directory of log file
outdir = '/home/datasets/log_data/Thunderbird/output'  # The output directory of parsing results

# processor = LogDataProcessor(outdir, outdir, 'Thunderbird', 'sliding', size_unit = 'minutes', window_size= 1, step_size= 0.5, shuffle = True)
# processor.process(n_train = 6000, window_size = 20, min_seq = 10, seq_len = None, adaptive_window= False)
options = {'dataset': 'Thunderbird', 'data_dir': outdir, 'batch_size': 1024, 'epochs': 100, 'save_dir': indir + '/saved_models', 'device': 'cuda'}

options['hidden_size'] = 128
options['num_layers'] = 2
options["embedding_dim"] = 300
options['embedding_strategy'] = 'tfidf'
options["dropout"] = 0.1
options['k'] = 968

options["n_epochs_stop"] = 30
options["lr"] = 0.001
options['lr_step'] = (50, 80)
options['lr_decay_ratio'] = 0.1

options['model_dir'] = options['save_dir'] + '/loganomaly/'
options['model_path'] = options['model_dir'] + 'best_loganomaly.pth'

# seed_all()
# # trainer = LogAnomalyTrainer(options)
# # trainer.train()
# # del trainer
# options['device'] = 'cuda'
# tester = LogAnomalyTester(options)
# tester.test()

# for seed in range(43, 45):
#     seed_all(seed)
#     trainer = LogAnomalyTrainer(options)
#     trainer.train()
#     del trainer
#     tester = LogAnomalyTester(options)
#     tester.test()
#     del tester

# seed_all(44)
# trainer = LogAnomalyTrainer(options)
# trainer.train()
# del trainer
# tester = LogAnomalyTester(options)
# tester.test()
# del tester

# seed_all(45)
# trainer = LogAnomalyTrainer(options)
# trainer.train()
# del trainer
tester = LogAnomalyTester(options)
tester.test()
del tester