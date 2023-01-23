import logging
import os

import hydra
from omegaconf import DictConfig
from tokenizers.implementations import ByteLevelBPETokenizer

from data import read_train_val_datasets


def batch_iterator(dataset, batch_size=100, column="code"):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size][column]


BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'


def train_tokenizers(train_set, cfg):
    # code
    tokenizer_code = ByteLevelBPETokenizer()
    tokenizer_code.train_from_iterator(batch_iterator(train_set, column="code"),
                                       vocab_size=cfg["vocab_size"],
                                       min_frequency=2,
                                       special_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN])

    os.makedirs(cfg['tokenizer_code_checkpoint'], exist_ok=True)
    tokenizer_code.save_model(cfg['tokenizer_code_checkpoint'])

    # nl
    tokenizer_nl = ByteLevelBPETokenizer()
    tokenizer_nl.train_from_iterator(batch_iterator(train_set, column="nl"),
                                     vocab_size=cfg["vocab_size"],
                                     min_frequency=2,
                                     special_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN])

    os.makedirs(cfg['tokenizer_nl_checkpoint'], exist_ok=True)
    tokenizer_nl.save_model(cfg['tokenizer_nl_checkpoint'])


@hydra.main(version_base=None, config_path=".", config_name="config_tokenizer")
def main(cfg: DictConfig):
    logger = logging.getLogger()
    from main import set_seed
    set_seed(cfg["seed"])

    train_set, _, _ = read_train_val_datasets(cfg)
    logger.info(f'Training samples {len(train_set)}')
    train_tokenizers(train_set, cfg)


if __name__ == '__main__':
    main()
