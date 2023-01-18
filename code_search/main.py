import logging
import os
import random

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from transformers import AutoTokenizer

from data import read_datasets, tokenize_dataset
from model.trainer import train, eval_test
from model.builder import build_model, build_model_checkpoint


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


def load_tokenizers():
    nl_tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    code_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    return nl_tokenizer, code_tokenizer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger()
    set_seed(cfg["seed"])

    train_set, valid_set, test_set = read_datasets(cfg)
    logger.info(f'Training samples {len(train_set)}')
    logger.info(f'Valid samples {len(valid_set)}')
    logger.info(f'Test samples {len(test_set)}')

    nl_tokenizer, code_tokenizer = load_tokenizers()
    if cfg["mode"] == "training":
        train_set = tokenize_dataset(train_set, nl_tokenizer, code_tokenizer, 512, 128, 'code', 'nl')
        valid_set = tokenize_dataset(valid_set, nl_tokenizer, code_tokenizer, 512, 128, 'code', 'nl')

        model = build_model(len(code_tokenizer), d_model=768, max_len=512, pe='random',
                            nl_hg_model='distilroberta-base')

        if cfg["wandb"]:
            wandb.init(project=cfg["wandb_params"]["project"],
                       entity=cfg["wandb_params"]["entity"],
                       name="test")
        train(train_set, valid_set, model, cfg["checkpoint"], wandb_enabled=cfg["wandb"])
    elif cfg["mode"] == "testing":
        test_set = tokenize_dataset(test_set, nl_tokenizer, code_tokenizer, 512, 128, 'code', 'nl')

        model = build_model_checkpoint(len(code_tokenizer), cfg["checkpoint"], d_model=768, max_len=512, pe='random',
                                       nl_hg_model='distilroberta-base')
        eval_test(test_set, model, batch_size=64)
    else:
        raise ValueError("Only training and testing model allowed")


if __name__ == '__main__':
    main()
