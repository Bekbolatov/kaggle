import boto3
import os.path
from bs4 import BeautifulSoup as bs

class FileCache:

    def __init__(self, local_cache_loc='/home/ec2-user/data/cache/', s3_key_prefix='kaggle/native/orig/'):
        self.local_cache_loc = local_cache_loc
        self.s3_key_prefix = s3_key_prefix
        self.client = boto3.client('s3')
        self.bucket = 'sparkydotsdata'

    def get_file(self, filename):
        if not os.path.isfile(self.local_cache_loc + filename):
            self.download_file(filename)
        return open(self.local_cache_loc + filename)

    def download_file(self, filename):
       self.client.download_file(self.bucket, self.s3_key_prefix + filename , self.local_cache_loc + filename) 

class SoupReader:

    def __init__(self, local_cache_loc='/home/ec2-user/data/cache/', s3_key_prefix='kaggle/native/orig/'):
        self.file_cache = FileCache(local_cache_loc=local_cache_loc, s3_key_prefix=s3_key_prefix)

    def get_soup(self, filename):
        file_handle = self.file_cache.get_file(filename)
        file_content = file_handle.read()
        file_handle.close()
        return bs(file_content)

class SoupIterator:
    def __init__(self, filenames):
        self.filenames = filenames
        self.index = 0
        self.soup_reader = SoupReader()

    def __iter__(self):
        self.index = 0
        return self

    def next(self):
        if self.index >= len(self.filenames):
            raise StopIteration
        else:
            value = self.filenames[self.index]
            self.index += 1
            return self.soup_reader.get_soup(value)


