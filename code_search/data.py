import os

from datasets import load_dataset


def preprocess_tokens(tokens):
    stc = ' '.join(tokens).replace('\n', ' ')
    stc = ' '.join(stc.strip().split())
    return stc


def read_datasets(cfg):
    data_files = {'train': os.path.join(cfg["dataset_dir"], cfg["lang"], 'train.jsonl'),
                  'valid': os.path.join(cfg["dataset_dir"], cfg["lang"], 'valid.jsonl'),
                  'test': os.path.join(cfg["dataset_dir"], cfg["lang"], 'test.jsonl')}
    columns_to_remove = ['repo', 'path', 'func_name', 'original_string', 'language', 'sha', 'url',
                         'partition', 'code', 'docstring']
    train_set = load_dataset('json', data_files=data_files, split='train').remove_columns(columns_to_remove)
    train_set = train_set.map(lambda sample: {'code': preprocess_tokens(sample['code_tokens']),
                                              'nl': preprocess_tokens(sample['docstring_tokens'])}) \
        .remove_columns(['docstring_tokens', 'code_tokens'])
    valid_set = load_dataset('json', data_files=data_files, split='valid').remove_columns(columns_to_remove)
    valid_set = valid_set.map(lambda sample: {'code': preprocess_tokens(sample['code_tokens']),
                                              'nl': preprocess_tokens(sample['docstring_tokens'])}) \
        .remove_columns(['docstring_tokens', 'code_tokens'])
    test_set = load_dataset('json', data_files=data_files, split='test').remove_columns(columns_to_remove)
    test_set = test_set.map(lambda sample: {'code': preprocess_tokens(sample['code_tokens']),
                                            'nl': preprocess_tokens(sample['docstring_tokens'])}) \
        .remove_columns(['docstring_tokens', 'code_tokens'])
    train_set = train_set.shuffle(seed=cfg["seed"]).select(range(0, 50000))
    test_set = test_set.shuffle(seed=cfg["seed"]).select(range(0, 5000))
    valid_set = valid_set.shuffle(seed=cfg["seed"]).select(range(0, 3000))
    return train_set, valid_set, test_set


def tokenize_function(examples, tokenizer, max_len, column):
    dic = tokenizer(examples[column], padding="max_length", truncation=True, max_length=max_len)
    dic_new = {x + "_" + column: y for x, y in dic.items()}
    return dic_new


def tokenize_column(dataset, tokenizer, max_len, column):
    return dataset.map(lambda examples: tokenize_function(examples,
                                                          tokenizer=tokenizer,
                                                          max_len=max_len,
                                                          column=column),
                       batched=True, load_from_cache_file=True).remove_columns([column])


def tokenize_dataset(dataset, tokenizer_nl, tokenizer_code, max_len_code, max_len_nl, column_code, column_nl):
    new_dataset = tokenize_column(dataset, tokenizer_nl, max_len_nl, column_nl)
    new_dataset = tokenize_column(new_dataset, tokenizer_code, max_len_code, column_code)
    return new_dataset
