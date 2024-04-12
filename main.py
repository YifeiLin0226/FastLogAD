import yaml
import argparse
import os
import numpy as np

from data.parser import Parser
from data.processor import LogDataProcessor
from data.dataset import ElectraDataset
from utils import seed_all
from train_test.trainer import ElectraTrainer
from train_test.tester import ElectraTester 

seed_all(42)
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='bgl.yaml', help='config file name.')
args = parser.parse_args()

configs = yaml.load(open(os.path.join('./configs', args.config), 'r'), Loader=yaml.FullLoader)
model_configs = configs['model_configs']
del configs['model_configs']


trainer = ElectraTrainer(model_configs, configs)
trainer.train()
# # del trainer
tester = ElectraTester(model_configs, configs)
tester.test()