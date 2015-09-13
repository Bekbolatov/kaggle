from __future__ import division
import numpy as np
import pandas as pd
import math
import doc_processor
import doc_parser

class ParserRunner:

    def run(self, run_id, my_id, total_ids):
        reload(doc_processor)
        reload(doc_parser)
        filenames = self.get_filenames(my_id, total_ids)
        processor = doc_processor.DocProcessor(filenames, run_id, doc_parser.parse, my_id) 
        processor.process()
       

    def get_filenames(self, my_id, total_ids):
        my_id = int(my_id)
        total_ids = int(total_ids)
        filenames = pd.read_csv('/home/ec2-user/data/native/all_filenames.csv', header=None, names=['filename', 'size'])
        filenames = np.asarray(filenames['filename'])
        batch_size = int(math.ceil(len(filenames) / total_ids))
        start_idx = my_id * batch_size
        stop_idx = start_idx + batch_size
        return filenames[start_idx:stop_idx]


