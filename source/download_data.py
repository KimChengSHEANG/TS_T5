
from __future__ import print_function
from source.resources import EXP_DIR
from pathlib import Path
import requests
import zipfile
import warnings
import os
from os import makedirs
from os.path import dirname
from os.path import exists
import shutil
from tqdm import tqdm
import gdown 

def download_5tokens_trained_model():
    # https://drive.google.com/file/d/19s68NxHybYta0t-smhWKrGZxUYyRlgHA/view?usp=sharing
    pretrained_model = EXP_DIR / 'model_5tokens/pytorch_model.bin'
    
    if not pretrained_model.exists():
        print("Downloading 5 tokens pretrained models...")
        dst_file = str(EXP_DIR / "model_5tokens.zip")
        
        gdown.download(id='19s68NxHybYta0t-smhWKrGZxUYyRlgHA', output=dst_file, quiet=False)
        if Path(dst_file).exists():
            gdown.extractall(dst_file, str(EXP_DIR))
            os.remove(dst_file)

def download_4tokens_trained_model():
    #https://drive.google.com/file/d/19sDdfrnRDAeL6bKraKxb8mrI9t7JCdVE/view?usp=sharing
    pretrained_model = EXP_DIR / 'model_4tokens/pytorch_model.bin'
    
    if not pretrained_model.exists():
        print("Downloading 4 tokens pretrained models...")
        dst_file = str(EXP_DIR / "model_4tokens.zip")
        
        gdown.download(id='19sDdfrnRDAeL6bKraKxb8mrI9t7JCdVE', output=dst_file, quiet=False)
        if Path(dst_file).exists():
            gdown.extractall(dst_file, str(EXP_DIR))
            os.remove(dst_file)
