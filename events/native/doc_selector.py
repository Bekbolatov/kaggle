from __future__ import division
import numpy as np
import pandas as pd
import math
from doc_processor import DocProcessor
from doc_parser import parse

class ParserRunner:

    def run(self, input_data):
        """ input_data = '01:1:40' """
        run_id, my_id, total_ids = input_data.split(':')
        filenames = self.get_filenames(my_id, total_ids)
        processor = DocProcessor(filenames, run_id, parse, my_id) 
        processor.process()
        
    def get_filenames(self, my_id, total_ids):
        my_id = int(my_id)
        total_ids = int(total_ids)
        filenames = pd.read_csv('/home/ec2-user/data/native/all_filenames.csv', header=None, names=['filename', 'size'])
        filenames = np.asarray(filenames['filename'])
        batch_size = int(math.ceil(len(filenames) / total_ids))
        start_idx = my_id * batch_size
        stop_idx = start_id + batch_size
        return filenames[start_id:stop_idx]


