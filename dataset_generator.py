import gzip
import json
import logging
import os
import pathlib
import re
import shutil
import zipfile
from typing import Optional

import hydra
import requests
from omegaconf import DictConfig
from tqdm import tqdm

from main import set_seed

LANGUAGES_CSN = (
    'python',
    'java',
    'ruby',
    'javascript',
    'go',
    'php'
)
CSN_DATASET_SPLIT_PATH = 'https://github.com/guoday/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip'
CSN_DATASET_BASE_PATH = 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/'


def download_url(url: str, save_path: str, chunk_size: int = 128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
            f.write(chunk)


def unzip_file(file_path: str, output_dir: str, output_path: Optional[str] = None):
    if 'gz' in file_path:
        with gzip.open(file_path, 'rb') as f1, open(output_path, 'w') as f2:
            f2.write(f1.read().decode('utf-8'))
    else:
        with zipfile.ZipFile(file_path, 'r') as f:
            f.extractall(output_dir)


def download_codesearchnet_dataset(dataset_dir):
    """Download CodeSearchNet dataset and clean it using GraphCodeBERT cleaning splits (Guo et al's)
    Return:
        dataset_dir (str): the path containing the downloaded dataset.
    """
    zip_file_path = 'dataset.zip'
    logger = logging.getLogger()

    if not os.path.exists(zip_file_path):
        logger.info('Downloading CodeSearchNet dataset...')
        download_url(CSN_DATASET_SPLIT_PATH, zip_file_path)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    unzip_file(zip_file_path, './')

    os.chdir(dataset_dir)
    for lang in LANGUAGES_CSN:
        logger.info(f'Creating {lang} dataset.')
        try:
            os.remove(os.path.join(lang, 'codebase.txt'))
            os.remove(os.path.join(lang, 'test.txt'))
            os.remove(os.path.join(lang, 'valid.txt'))
        except:
            pass
        if not os.path.exists(os.path.join(lang, 'final')):
            logger.info(f'Downloading CodeSearchNet {lang} dataset.')
            download_url(os.path.join(CSN_DATASET_BASE_PATH, f'{lang}.zip'), f'{lang}.zip')
            unzip_file(f'{lang}.zip', './')
        # we care about the training set that we can further split into train/val/test
        if os.path.exists(os.path.join(lang, 'final/jsonl/test')):
            shutil.rmtree(os.path.join(lang, 'final/jsonl/test'))
        if os.path.exists(os.path.join(lang, 'final/jsonl/valid')):
            shutil.rmtree(os.path.join(lang, 'final/jsonl/valid'))

    for lang in LANGUAGES_CSN:
        logger.info(f'Cleaning {lang} dataset.')
        data = {}
        # gzip all .gz files and add them to `data` with their url as key
        for file in tqdm(pathlib.Path(f'./{lang}').rglob('*.gz')):
            unzip_file(str(file), '', str(file)[:-3])
            os.remove(file)
            with open(str(file)[:-3]) as f:
                for line in f:
                    js = json.loads(line)
                    data[js['url']] = js
        with open(f'./{lang}/dataset.jsonl', 'w') as f1, open(f'./{lang}/train.txt', encoding='utf-8') as f2:
            for line in f2:
                line = line.strip()
                # we only keep code snippets that are clean (based on GraphCodeBERT cleaning)
                #   by matching the url with a key in `data`.
                if line in data:
                    # we only extract the original code and the code tokens to filter
                    js = {'original_string': data[line]['original_string'],
                          'code_tokens': data[line]['code_tokens']}
                    f1.write(json.dumps(js) + '\n')
        os.remove(os.path.join(lang, 'train.txt'))
        shutil.rmtree(os.path.join(lang, 'final'))
    # clean folders
    for file in os.listdir('.'):
        if re.match('.*.(zip|pkl|py|sh)', file):
            os.remove(file)
    os.chdir('../')


@hydra.main(version_base=None, config_path="conf_dataset", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg["seed"])
    if cfg["download_csn"]:
        download_codesearchnet_dataset(cfg["dataset_dir"])


if __name__ == '__main__':
    main()
