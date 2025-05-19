# data/speech_commands.py

import os
import sys
import tarfile
import urllib.request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from training.utils.config import DATASET_PATH

def download_dataset():
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    archive_path = os.path.join(DATASET_PATH, "speech_commands_v0.02.tar.gz")

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    if not os.path.exists(archive_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, archive_path)
    else:
        print("Archive already downloaded.")

    print("Extracting dataset...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(DATASET_PATH)

    print("Cleaning up...")
    os.remove(archive_path)

    print("Dataset ready at", DATASET_PATH)

if __name__ == "__main__":
    download_dataset()
