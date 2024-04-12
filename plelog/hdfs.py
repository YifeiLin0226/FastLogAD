from train_test.trainer import PLELogTrainer
from train_test.tester import PLELogTester
from data.dataset import PLELogDataset
from data.processor import LogDataProcessor
from utils import seed_all


indir  = '/home/datasets/log_data/HDFS' # The input directory of log file
outdir = '/home/datasets/log_data/HDFS/output'  # The output directory of parsing results


options = {'dataset': 'HDFS', 'data_dir': outdir, 'batch_size': 32, 'epochs': 30, 'save_dir': indir + '/saved_models', 'device': 'cuda'}

# processor = LogDataProcessor(outdir, outdir, 'BGL', 'sliding', size_unit = "minutes", window_size = 5, step_size = 1, shuffle = True)
# processor.process(window_size = 20, min_seq = 10, seq_len = None, adaptive_window= False)
processor = LogDataProcessor(outdir, outdir, 'HDFS', 'session', train_ratio = 0.8, valid_ratio = 0.2, shuffle = False)
processor.process(window_size = 120, min_seq = 10, adaptive_window= False)
options['use_pseudo'] = True
options['tok_anomaly_ratio'] = 0.5

options['hidden_size'] = 100
options['num_layers'] = 2
options["embedding_dim"] = 300
options['embedding_strategy'] = 'tfidf' 
options["dropout"] = 0.1


options["n_epochs_stop"] = 10
options["lr"] = 0.00005

options['model_dir'] = options['save_dir'] + '/plelog/'
options['model_path'] = options['model_dir'] + 'best_plelog.pth'

seed_all()
trainer = PLELogTrainer(options)
trainer.train()
del trainer
tester = PLELogTester(options)
tester.test()