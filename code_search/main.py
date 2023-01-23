import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

import wandb
from data import read_train_val_datasets, tokenize_dataset, read_test_set
from model.builder import build_model, build_model_checkpoint
from model.trainer import train, eval_test


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
    nl_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    code_tokenizer = nl_tokenizer
    return nl_tokenizer, code_tokenizer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger()
    set_seed(cfg["seed"])

    nl_tokenizer, code_tokenizer = load_tokenizers()
    if cfg["mode"] == "training":
        train_set, valid_set = read_train_val_datasets(cfg)
        logger.info(f'Training samples {len(train_set)}')
        logger.info(f'Valid samples {len(valid_set)}')
        train_set = tokenize_dataset(train_set, nl_tokenizer,
                                     code_tokenizer,
                                     cfg["training_params"]["block_size_code"],
                                     cfg["training_params"]["block_size_nl"],
                                     'code', 'nl')
        valid_set = tokenize_dataset(valid_set, nl_tokenizer,
                                     code_tokenizer,
                                     cfg["training_params"]["block_size_code"],
                                     cfg["training_params"]["block_size_nl"],
                                     'code', 'nl')

        model = build_model(code_tokenizer, nl_tokenizer, cfg)

        if cfg["wandb"]:
            config = {
                "n_layers": cfg["model_params"]["n_layers"],
                "bach_size_train": cfg["training_params"]["batch_size_train"],
                "lr": cfg["training_params"]["lr"]
            }
            wandb.init(project=cfg["wandb_params"]["project"],
                       entity=cfg["wandb_params"]["entity"],
                       name=cfg["pe_model"],
                       config=config)
        train(train_set=train_set,
              valid_set=valid_set,
              model=model,
              checkpoint=cfg["checkpoint"],
              batch_size_train=cfg["training_params"]["batch_size_train"],
              batch_size_eval=cfg["training_params"]["batch_size_eval"],
              lr=cfg["training_params"]["lr"],
              epochs=cfg["training_params"]["epochs"],
              patience=cfg["training_params"]["patience"],
              gradient_accumulation=cfg["training_params"]["gradient_accumulation"],
              max_grad_norm=cfg["training_params"]["max_grad_norm"],
              wandb_enabled=cfg["wandb"],
              log_steps=cfg["training_params"]["log_steps"])
    elif cfg["mode"] == "testing":
        test_set = read_test_set(cfg)
        logger.info(f'Test samples of lang {cfg["lang_eval"]}: {len(test_set)}')
        logger.info(f'Distractors: {cfg["training_params"]["batch_size_eval"] - 1}')
        test_set = tokenize_dataset(test_set, nl_tokenizer,
                                    code_tokenizer,
                                    cfg["training_params"]["block_size_code"],
                                    cfg["training_params"]["block_size_nl"],
                                    'code', 'nl')

        model = build_model_checkpoint(code_tokenizer, nl_tokenizer, cfg)
        eval_test(test_set, model, batch_size=cfg["training_params"]["batch_size_eval"])
    else:
        raise ValueError("Only training and testing model allowed")


if __name__ == '__main__':
    main()
