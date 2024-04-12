import yaml
import argparse
import os
import numpy as np

from data.parser import Parser
from data.processor import LogDataProcessor


indir  = '/home/datasets/log_data/HDFS' # The input directory of log file
outdir = '/home/datasets/log_data/HDFS/output'  # The output directory of parsing results

st         = 0.5  # Similarity threshold
depth      = 5  # Depth of all leaf nodes

parser = Parser(indir, outdir, 'HDFS', 'drain', {'depth':depth, 'st':st})
parser.parse()

processor = LogDataProcessor(outdir, outdir, 'HDFS', 'session', shuffle = False)
processor.process(n_train = 4855, min_seq = 2, adaptive_window= True)
