import hashlib
import urllib.request

import requests
import os
import pathlib
import shutil
import zipfile
import gzip
import json
import urllib.request

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download(root="./"):
    if _check_integrity(root):
        return True
    _clean_directory(root)
    _download(root)
    if not _check_integrity(root):
        print("Data is not ready")
        raise Exception()


def _check_integrity(root):
    data_path = os.path.join(root, "ETHEC_dataset")
    try:
        if os.path.isdir(data_path):
            for section in ["train", "test", "val"]:
                section_json_address = os.path.join(data_path, "splits", f"{section}.json")
                if not os.path.isfile(section_json_address):
                    print("Incomplete Dataset")
                    return False
                section_data = json.loads(open(section_json_address).read())
                for index, (key, values) in enumerate(section_data.items()):
                    file_path = os.path.join(data_path, "IMAGO_build_test_resized", values['image_path'],
                                             values['image_name'])
                    if not os.path.isfile(file_path):
                        print("Incomplete Dataset")
                        return False


        else:
            print("Incomplete Dataset")
            return False
    except:
        return False
    print("Data is ready")
    return True


def _download_zip(file_path):
    if os.path.isfile(file_path):
        if str(_get_md5(file_path)) == "d846bc07b0a356d5701bf19942c77b2d":
            return
        else:
            # os.remove(file_path)
            print("Downloaded Zip File Was Corrupted")
    _download_url(
        "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/365379/ETHEC_dataset_v0.2.zip?sequence=6&isAllowed=y",
        file_path
    )


def _download(root):
    file_path = os.path.join(root, "ETHEC_dataset_v0.2.zip")
    _download_zip(file_path)
    data_path = os.path.join(root, ".")
    _unzip(file_path, data_path)


def _unzip(file_path, data_path):
    print("Unzipping the dataset")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    print("Dataset Unzipped")


def _clean_directory(root):
    data_path = os.path.join(root, "ETHEC_dataset")
    if os.path.isdir(data_path):
        shutil.rmtree(data_path)
        print("REMOVED", data_path)


def _get_md5(file_address):
    return hashlib.md5(open(file_address, 'rb').read()).hexdigest()
