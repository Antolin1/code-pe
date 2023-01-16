import os

from datasets import load_dataset


def read_datasets(cfg):
    data_files = {'train': os.path.join(cfg["dataset_dir"], 'train.jsonl'),
                  'valid': os.path.join(cfg["dataset_dir"], 'valid.jsonl'),
                  'test': os.path.join(cfg["dataset_dir"], 'test.jsonl')}
    train_set = load_dataset('json', data_files=data_files, split='train')
    valid_set = load_dataset('json', data_files=data_files, split='valid')
    test_set = load_dataset('json', data_files=data_files, split='test')
    return train_set, valid_set, test_set


def tokenize_function(examples, tokenizer, max_len):
    return tokenizer(examples["func"], padding="max_length", truncation=True, max_length=max_len)


def tokenize_dataset(dataset, tokenizer, max_len):
    return dataset.map(lambda examples: tokenize_function(examples,
                                                          tokenizer=tokenizer,
                                                          max_len=max_len),
                       batched=True, load_from_cache_file=False).remove_columns(["project", "commit_id", "func", "idx"])
