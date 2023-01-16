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

## Training models for code tasks

Random embeddings
```shell
cd defect_prediction
python main.py mode=training pe_embeddings=random checkpoint=random_pe_embeddings.bin
python main.py mode=testing pe_embeddings=random checkpoint=random_pe_embeddings.bin
```

Sinusoid embeddings
```shell
cd defect_prediction
python main.py mode=training pe_embeddings=sinusoid checkpoint=sinusoid_pe_embeddings.bin
python main.py mode=testing pe_embeddings=sinusoid checkpoint=sinusoid_pe_embeddings.bin
```

gpt-2
```shell
cd defect_prediction
python main.py mode=training pe_embeddings=gpt2 checkpoint=gpt2_pe_embeddings.bin
python main.py mode=testing pe_embeddings=gpt2 checkpoint=gpt2_pe_embeddings.bin
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

