import os

from datasets import load_dataset, concatenate_datasets


def preprocess_tokens(tokens):
    stc = ' '.join(tokens).replace('\n', ' ')
    stc = ' '.join(stc.strip().split())
    return stc


LANGUAGES = ('python',
             'javascript',
             'go', 'ruby',
             'java',
             'php')

COLUMNS_TO_REMOVE = ['repo', 'path', 'func_name', 'original_string', 'language', 'sha', 'url',
                     'partition', 'code', 'docstring']


def read_train_val_datasets(cfg, samples_per_lang_train=25000, samples_per_lang_eval=1000):
    train_datasets = []
    val_datasets = []
    for language in LANGUAGES:
        data_files = {'train': os.path.join(cfg["dataset_dir"], language, 'train.jsonl'),
                      'valid': os.path.join(cfg["dataset_dir"], language, 'valid.jsonl')}

        train_set = load_dataset('json', data_files=data_files, split='train').remove_columns(COLUMNS_TO_REMOVE)
        train_set = train_set.shuffle(seed=cfg["seed"]).select(range(0, min(len(train_set), samples_per_lang_train)))
        train_datasets.append(train_set)

        valid_set = load_dataset('json', data_files=data_files, split='valid').remove_columns(COLUMNS_TO_REMOVE)
        valid_set = valid_set.shuffle(seed=cfg["seed"]).select(range(0, min(len(valid_set), samples_per_lang_eval)))
        val_datasets.append(valid_set)

    train_set = concatenate_datasets(train_datasets)
    valid_set = concatenate_datasets(val_datasets)
    train_set = train_set.map(lambda sample: {'code': preprocess_tokens(sample['code_tokens']),
                                              'nl': preprocess_tokens(sample['docstring_tokens'])}) \
        .remove_columns(['docstring_tokens', 'code_tokens'])

    valid_set = valid_set.map(lambda sample: {'code': preprocess_tokens(sample['code_tokens']),
                                              'nl': preprocess_tokens(sample['docstring_tokens'])}) \
        .remove_columns(['docstring_tokens', 'code_tokens'])
    # train_set = train_set.shuffle(seed=cfg["seed"]).select(range(0, 50000))
    # test_set = test_set.shuffle(seed=cfg["seed"]).select(range(0, 5000))
    # valid_set = valid_set.shuffle(seed=cfg["seed"]).select(range(0, 3000))
    return train_set, valid_set


def read_test_set(cfg):
    data_files = {'test': os.path.join(cfg["dataset_dir"], cfg["lang_eval"], 'test.jsonl')}
    test_set = load_dataset('json', data_files=data_files, split='test').remove_columns(COLUMNS_TO_REMOVE)
    test_set = test_set.map(lambda sample: {'code': preprocess_tokens(sample['code_tokens']),
                                            'nl': preprocess_tokens(sample['docstring_tokens'])}) \
        .remove_columns(['docstring_tokens', 'code_tokens'])
    return test_set


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
