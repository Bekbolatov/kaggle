import boto3
import os, os.path
from bs4 import BeautifulSoup as bs
import json

class FileCache:

    def __init__(self,
            local_cache_loc='/home/ec2-user/data/cache/',
            s3_key_prefix='kaggle/native/',
            run_name='docs01',
            max_kb=3145728):
        """ max_kb 3145728 = 3 GB *1024 * 1024"""
        self.s3_key_prefix = s3_key_prefix + 'orig/'
        self.s3_key_prefix_upload = s3_key_prefix + 'docs/' + run_name + '/'
        self.client = boto3.client('s3')
        self.bucket = 'sparkydotsdata'
        self.download_cache = os.path.join(local_cache_loc, 'download/')
        self.upload_cache = os.path.join(local_cache_loc, 'upload/')
        self.max_kb_download = int(max_kb*0.8)
        self.max_kb_upload = max_kb - self.max_kb_download

    def open_file(self, filename):
        if not os.path.isfile(os.path.join(self.download_cache, filename)):
            self.download_file(filename)
        return open(os.path.join(self.download_cache, filename))

    def download_file(self, filename):
       self.clean_download_cache()
       s3_location = self.s3_key_prefix + filename
       local_location = os.path.join(self.download_cache, filename)
       print("Downloading from %s to %s" % ( s3_location, local_location))
       self.client.download_file(self.bucket, s3_location, local_location)

    def upload_file(self, filename, contents):
       self.clean_upload_cache()
       self.save_file(os.path.join(self.upload_cache, filename), contents)
       self.client.upload_file(os.path.join(self.upload_cache, filename), self.bucket, self.s3_key_prefix_upload + filename) 

    def save_file(self, location, json_array):
        with open(location, mode='w') as feedsjson:
            for entry in json_array:
                json.dump(entry, feedsjson)
                feedsjson.write('\n')
        feedsjson.close()

    def clean_download_cache(self):
        self.clean_cache(self.download_cache, self.max_kb_download)

    def clean_upload_cache(self):
        self.clean_cache(self.upload_cache, self.max_kb_upload)

    def clean_cache(self, path, limit_kb):
        used_kb = self.current_cache_size(path)
        print("Used space: %d KB in path: %s" % (used_kb, path))
        if used_kb > limit_kb:
            print("Cache reached max_kb size - clearing cache")
            files = self.sorted_ls(path)
            earlier_files = files[:(len(files)//2)]
            later_files = files[(len(files)//2):]
            print("First trying to remove only earlier half")
            for filename in earlier_files:
                os.remove(os.path.join(path, filename))
            if self.current_cache_size(path) > limit_kb:
                print("... not enough, removing all")
                for filename in later_files:
                    os.remove(os.path.join(path, filename))

    def sorted_ls(self, path):
        mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
        return list(sorted(os.listdir(path), key=mtime))

    def current_cache_size(self, path):
        used_kb = sum([os.path.getsize(os.path.join(path, filename)) for filename in os.listdir(path)]) // 1024
        return used_kb

class SoupIO:

    def __init__(self, run_name='01', local_cache_loc='/home/ec2-user/data/cache/', s3_key_prefix='kaggle/native/',max_kb=3145728):
        self.file_cache = FileCache(local_cache_loc=local_cache_loc, s3_key_prefix=s3_key_prefix, run_name=run_name, max_kb=max_kb)

    def get_soup(self, filename):
        file_handle = self.file_cache.open_file(filename)
        file_content = file_handle.read()
        file_handle.close()
        if file_content:
            soup = bs(file_content, 'lxml')
        else:
            soup = None
        return soup

    def put_docs(self, filename, docs_array):
        self.file_cache.upload_file(filename, docs_array)

class DocProcessor:
    def __init__(self, filenames, run_name, parse, part_id, log_dir='/home/ec2-user/logs/'):
        self.filenames = filenames
        self.run_name = run_name
        if type(part_id) is str:
            self.part_id = part_id
        else:
            self.part_id = str(part_id).zfill(3)
        self.parse = parse
        self.log_dir = log_dir
        self.soup_io = SoupIO(run_name=run_name)

    def process(self):
        ferr = open(os.path.join(self.log_dir, "errors_in_scraping.log"),"w")
        ferr.write("Starting processing part_id: %s\n" % part_id)
        json_array = []  
        for i, filename in enumerate(self.filenames):
            soup = self.soup_io.get_soup(filename)
            if soup:
                try:
                    doc = self.parse(soup, filename)
                    json_array.append(doc)
                except Exception as e:
                    ferr.write("parse error with reason : %s on file: %s\n" %(str(e), filename))
            if i % 100 == 0:
                print("processed %d docs" % (i))

        self.soup_io.put_docs(part_id, json_array)
        ferr.close()



def test_parse(soup, filename):
    return {"id": filename, "hg": "df"}


