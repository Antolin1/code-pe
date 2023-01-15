# On the code positional encodings

## Requirements



## Running experiments

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

