import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import RobertaTokenizerFast, AutoTokenizer

import wandb
from data import read_datasets, tokenize_dataset
from model.builder import build_model, build_model_checkpoint
from model.trainer import train, eval_test
from train_tokenizers import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN


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


def load_tokenizers(cfg):
    nl_tokenizer = RobertaTokenizerFast(vocab_file=os.path.join(cfg['tokenizer_nl_checkpoint'],
                                                                'vocab.json'),
                                        merges_file=os.path.join(cfg['tokenizer_nl_checkpoint'],
                                                                 'merges.txt'),
                                        bos_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                                        pad_token=PAD_TOKEN, unk_token=UNK_TOKEN)
    code_tokenizer = RobertaTokenizerFast(vocab_file=os.path.join(cfg['tokenizer_code_checkpoint'],
                                                                  'vocab.json'),
                                          merges_file=os.path.join(cfg['tokenizer_code_checkpoint'],
                                                                   'merges.txt'),
                                          bos_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                                          pad_token=PAD_TOKEN, unk_token=UNK_TOKEN)

    nl_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    code_tokenizer = nl_tokenizer
    return nl_tokenizer, code_tokenizer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger()
    set_seed(cfg["seed"])

    train_set, valid_set, test_set = read_datasets(cfg)
    logger.info(f'Training samples {len(train_set)}')
    logger.info(f'Valid samples {len(valid_set)}')
    logger.info(f'Test samples {len(test_set)}')

    nl_tokenizer, code_tokenizer = load_tokenizers(cfg)
    if cfg["mode"] == "training":
        # TODO load parameters of the conifg
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

        model = build_model(len(code_tokenizer), len(nl_tokenizer), cfg)

        if cfg["wandb"]:
            wandb.init(project=cfg["wandb_params"]["project"],
                       entity=cfg["wandb_params"]["entity"],
                       name=cfg["pe_model"])
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
        test_set = tokenize_dataset(test_set, nl_tokenizer,
                                    code_tokenizer,
                                    cfg["training_params"]["block_size_code"],
                                    cfg["training_params"]["block_size_nl"],
                                    'code', 'nl')

        model = build_model_checkpoint(len(code_tokenizer), len(nl_tokenizer), cfg)
        eval_test(test_set, model, batch_size=cfg["training_params"]["batch_size_eval"])
    else:
        raise ValueError("Only training and testing model allowed")


if __name__ == '__main__':
    main()
