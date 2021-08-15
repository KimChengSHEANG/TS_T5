import os
import tarfile
import urllib
import zipfile
from pathlib import Path
from tqdm import tqdm


def download_report_hook(t):
  last_b = [0]
  def inner(b=1, bsize=1, tsize=None):
    if tsize is not None:
        t.total = tsize
    t.update((b - last_b[0]) * bsize)
    last_b[0] = b
  return inner

def download_url(url, output_path):
    name = url.split('/')[-1]
    file_path = f'{output_path}/{name}'
    if not Path(file_path).exists():
        # with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        #     urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        with tqdm(unit='B', unit_scale=True, leave=True, miniters=1,
                  desc=name) as t:  # all optional kwargs
            urllib.request.urlretrieve(url, filename=file_path, reporthook=download_report_hook(t), data=None)
    return file_path

def unzip(file_path, dest_dir=None):
    if dest_dir is None:
        dest_dir = os.path.dirname(file_path)
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    elif file_path.endswith("tar.gz") or file_path.endswith("tgz"):
        tar = tarfile.open(file_path, "r:gz")
        tar.extractall(dest_dir)
        tar.close()
    elif file_path.endswith("tar"):
        tar = tarfile.open(file_path, "r:")
        tar.extractall(dest_dir)
        tar.close()

