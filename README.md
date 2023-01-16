# On the code positional encodings

## Requirements

## Datasets

Defect prediction:
```shell
cd defect_prediction/dataset
pip install gdown
gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
python preprocess.py
```

Code search:
```shell
cd code_search
./download_dataset.sh
```

## Training models for code tasks

Defect prediciton
```shell
cd defect_prediction
./run_train.sh
./run_test.sh
```


## Running analysis

Absolute positions
```shell
python main_analysis.py pe_model=microsoft/codebert-base analysis=absolute_position
```

Relative positions
```shell
python main_analysis.py pe_model=microsoft/codebert-base analysis=relative_position
```

Position wise cosine similarity
```shell
python main_analysis.py pe_model=microsoft/codebert-base analysis=position_wise_cosine_similarity
```

SVD decomposition
```shell
python main_analysis.py pe_model=microsoft/codebert-base analysis=eigenvalues_analysis
```

