#from urllib import request
import requests
import sys
import tarfile
import ntpath, re, os, shutil
import urllib
import urllib.request
import urllib.error

import torch
from torch.utils.model_zoo import tqdm
USER_AGENT = "pytorch/vision"
chunk_size = 1024
url = "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz"
filename = ntpath.basename(url)
data_path = "notMNIST_small"
with open(filename, "wb") as fh:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
        with tqdm(total=response.length) as pbar:
            for chunk in iter(lambda: response.read(chunk_size), ""):
                if not chunk:
                    break
                pbar.update(chunk_size)
                fh.write(chunk)