#source: https://github.com/ndrplz/google-drive-downloader

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

class GoogleDriveDownloader:

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def download_file_from_google_drive(file_id, dest_path, overwrite=False, unzip=False):

        destination_directory = dirname(dest_path)
        if not exists(destination_directory):
            makedirs(destination_directory)

        if not exists(dest_path) or overwrite:

            session = requests.Session()

            print('Downloading {} into {}... '.format(file_id, dest_path), end='', flush=True)
            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)
            print(response.headers['Content-Length'])
            GoogleDriveDownloader._save_response_content(response, dest_path)
            print('Done.')

            if unzip:
                try:
                    print('Unzipping...', end='', flush=True)
                    with zipfile.ZipFile(dest_path, 'r') as z:
                        z.extractall(destination_directory)
                    print('Done.')
                except zipfile.BadZipfile:
                    warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_id))

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination):
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(GoogleDriveDownloader.CHUNK_SIZE), total=int(int(response.headers['Content-Length'])/GoogleDriveDownloader.CHUNK_SIZE)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)



def download_5tokens_trained_model():
    # https://drive.google.com/file/d/19s68NxHybYta0t-smhWKrGZxUYyRlgHA/view?usp=sharing
    pretrained_model = EXP_DIR / 'model_5tokens/pytorch_model.bin'
    
    if not pretrained_model.exists():
        print("Downloading 5 tokens pretrained models...")
        dst_file = EXP_DIR / "model_5tokens.zip"
        GoogleDriveDownloader.download_file_from_google_drive(file_id='19s68NxHybYta0t-smhWKrGZxUYyRlgHA', dest_path=dst_file, unzip=True)
        os.remove(dst_file)

def download_4tokens_trained_model():
    #https://drive.google.com/file/d/19sDdfrnRDAeL6bKraKxb8mrI9t7JCdVE/view?usp=sharing
    pretrained_model = EXP_DIR / 'model_4tokens/pytorch_model.bin'
    
    if not pretrained_model.exists():
        print("Downloading 4 tokens pretrained models...")
        dst_file = EXP_DIR / "model_4tokens.zip"
        GoogleDriveDownloader.download_file_from_google_drive(file_id='19sDdfrnRDAeL6bKraKxb8mrI9t7JCdVE', dest_path=dst_file, unzip=True)
        os.remove(dst_file)

