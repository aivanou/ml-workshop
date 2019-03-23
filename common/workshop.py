#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import requests
import common.tqdm_utils as tqdm_utils


REPOSITORY_PATH = "https://github.com/tierex/ml-workshop"


def download_file(url, file_path):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length'))
    try:
        with open(file_path, 'wb', buffering=16*1024*1024) as f:
            bar = tqdm_utils.tqdm_notebook_failsafe(total=total_size, unit='B', unit_scale=True)
            bar.set_description(os.path.split(file_path)[-1])
            for chunk in r.iter_content(32 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
            bar.close()
    except Exception as e:
        print("Download failed")
        print(str(e))
    finally:
        if os.path.getsize(file_path) != total_size:
            os.remove(file_path)
            print("Removed incomplete download")


def download_from_github(version, fn, target_dir, force=False):
    url = REPOSITORY_PATH + "/releases/download/{0}/{1}".format(version, fn)
    file_path = os.path.join(target_dir, fn)
    if os.path.exists(file_path) and not force:
        print("File {} is already downloaded.".format(file_path))
        return
    download_file(url, file_path)


def sequential_downloader(version, fns, target_dir, force=False):
    os.makedirs(target_dir, exist_ok=True)
    for fn in fns:
        download_from_github(version, fn, target_dir, force=force)


def download_tag_prediction(force=False):
    sequential_downloader(
        "tag-prediction",
        [
            "train.tsv",
            "validation.tsv",
            "test.tsv",
            "text_prepare_tests.tsv",
        ],
        "tag-prediction",
        force=force
    )


def download_name_generation(force=False):
    sequential_downloader(
        "names",
        [
            "names",
        ],
        "names",
        force=force
    )
    
def download_text_generation(force=False):
    sequential_downloader(
        "text-generation",
        [
            "nitz_texts.txt",
            "ny_articles.tar.gz",
        ],
        "text-generation",
        force=force
    )
