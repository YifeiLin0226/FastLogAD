from data.parser import Parser
from data.processor import LogDataProcessor


indir = '/home/datasets/log_data/BGL'
outdir = '/home/datasets/log_data/BGL/output/'
log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>' # BGL log format

st         = 0.3  # Similarity threshold
depth      = 3  # Depth of all leaf nodes


parser = Parser(indir, outdir, 'BGL', 'drain', {'depth':depth, 'st':st})
parser.parse()



processor = LogDataProcessor(outdir, outdir, 'BGL', 'sliding', size_unit = "minutes", window_size = 5, step_size = 1, shuffle = False)
processor.process(n_train = 5000, min_seq = 2, adaptive_window= True)



