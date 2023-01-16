import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import RobertaTokenizerFast

from data import read_datasets, tokenize_dataset
from model.builder import build_model, build_model_checkpoint
from model.trainer import train, eval_test
from train_tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN


def check_tokens(tokenizer):
    logger = logging.getLogger()
    logger.info(tokenizer(['for i in x', 'for loop'], padding="max_length", truncation=True,
                          max_length=128))


def load_tokenizer(cfg):
    tokenizer = RobertaTokenizerFast(vocab_file=os.path.join(cfg['tokenizer_checkpoint'],
                                                             'vocab.json'),
                                     merges_file=os.path.join(cfg['tokenizer_checkpoint'],
                                                              'merges.txt'),
                                     bos_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                                     pad_token=PAD_TOKEN, unk_token=UNK_TOKEN)
    check_tokens(tokenizer)
    return tokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger()
    set_seed(cfg["seed"])

    # load datasets
    train_set, valid_set, test_set = read_datasets(cfg)
    logger.info(f'Training samples {len(train_set)}')
    logger.info(f'Valid samples {len(valid_set)}')
    logger.info(f'Test samples {len(test_set)}')

    # tokenization
    tokenizer = load_tokenizer(cfg)
    train_set = tokenize_dataset(train_set, tokenizer, max_len=512)
    test_set = tokenize_dataset(test_set, tokenizer, max_len=512)
    valid_set = tokenize_dataset(valid_set, tokenizer, max_len=512)
    logger.info(f'{train_set}')

    if cfg["mode"] == "training":
        # get model
        model = build_model(vocab_size=len(tokenizer), num_labels=2, d_model=768, max_len=512, pe=cfg["pe_embeddings"])
        # train
        train(train_set, valid_set, model, cfg["checkpoint"])
    elif cfg["mode"] == "testing":
        model = build_model_checkpoint(vocab_size=len(tokenizer), num_labels=2, d_model=768, max_len=512,
                                       pe=cfg["pe_embeddings"], checkpoint=cfg["checkpoint"])
        eval_test(test_set, model)


if __name__ == '__main__':
    main()
