import boto3
import os.path


class FileCache:
    def __init__(self, loc = '/home/ec2-user/data/cache/'):
        self.loc = loc
        self.s3_key = 'kaggle/native/orig/'
        self.client = boto3.client('s3')

    def get_file(self, filename):
        if not os.path.isfile(self.loc + filename):
            self.download_file(filename)
        return open(self.loc + filename)

    def download_file(self, filename):
       self.client.download_file('sparkydotsdata', self.s3_key + filename , self.loc + filename) 

