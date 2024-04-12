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

indir  = '/home/datasets/log_data/Thunderbird' # The input directory of log file
outdir = '/home/datasets/log_data/Thunderbird/output'  # The output directory of parsing results

st         = 0.3  # Similarity threshold
depth      = 3  # Depth of all leaf nodes

parser = Parser(indir, outdir, 'Thunderbird', 'drain', {'depth':depth, 'st':st})
parser.parse()

processor = LogDataProcessor(outdir, outdir, 'Thunderbird', 'sliding', size_unit = 'minutes', window_size= 1, step_size= 0.5, shuffle = False)
processor.process(n_train = 6000, min_seq = 2, adaptive_window= True)
