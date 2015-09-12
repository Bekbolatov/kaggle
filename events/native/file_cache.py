import boto3
import os, os.path
from bs4 import BeautifulSoup as bs

class FileCache:

    def __init__(self, local_cache_loc='/home/ec2-user/data/cache/', s3_key_prefix='kaggle/native/orig/', max_kb=3145728):
        """ max_kb 3145728 = 3 GB *1024 * 1024"""
        self.local_cache_loc = local_cache_loc
        self.s3_key_prefix = s3_key_prefix
        self.client = boto3.client('s3')
        self.bucket = 'sparkydotsdata'
        self.max_kb = max_kb

    def get_file(self, filename):
        if not os.path.isfile(self.local_cache_loc + filename):
            self.download_file(filename)
        return open(self.local_cache_loc + filename)

    def download_file(self, filename):
       self.clean_cache()
       self.client.download_file(self.bucket, self.s3_key_prefix + filename , self.local_cache_loc + filename) 

    def clean_cache(self):
        used_kb = self.current_cache_size()
        if used_kb > self.max_kb:
            print("Cache reached max_kb size - clearing cache")
            files = self.sorted_ls()
            earlier_files = files[:(len(files)//2)]
            print("first trying to remove earlier half")
            for filename in earlier_files:
                os.remove(self.local_cache_loc + filename)
            if self.current_cache_size() > self.max_kb:
                later_files = files[(len(files)//2):]
                print("... not enough, removing all")
                for filename in later_files:
                    os.remove(self.local_cache_loc + filename)
        print("Used space: %d KB." % (used_kb))

    def sorted_ls(self):
        path = self.local_cache_loc
        mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
        return list(sorted(os.listdir(path), key=mtime))

    def current_cache_size(self):
        used_kb = sum([os.path.getsize(self.local_cache_loc + filename) for filename in os.listdir(self.local_cache_loc)]) // 1024
        return used_kb

class SoupReader:

    def __init__(self, local_cache_loc='/home/ec2-user/data/cache/', s3_key_prefix='kaggle/native/orig/',max_kb=3145728):
        self.file_cache = FileCache(local_cache_loc=local_cache_loc, s3_key_prefix=s3_key_prefix,max_kb=max_kb)

    def get_soup(self, filename):
        file_handle = self.file_cache.get_file(filename)
        file_content = file_handle.read()
        file_handle.close()
        return bs(file_content, 'lxml')

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


