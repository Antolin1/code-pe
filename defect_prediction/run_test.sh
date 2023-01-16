python main.py mode=testing pe_embeddings=random checkpoint=random_pe_embeddings.bin
python main.py mode=testing pe_embeddings=sinusoid checkpoint=sinusoid_pe_embeddings.bin
python main.py mode=testing pe_embeddings=gpt2 checkpoint=gpt2_pe_embeddings.bin
python main.py mode=testing pe_embeddings=microsoft/codebert-base checkpoint=codebert_pe_embeddings.bin
python main.py mode=testing pe_embeddings=roberta-base checkpoint=roberta_pe_embeddings.bin
python main.py mode=testing pe_embeddings=huggingface/CodeBERTa-small-v1 checkpoint=codeberta_pe_embeddings.bin
python main.py mode=testing pe_embeddings=bow checkpoint=bow_pe_embeddings.bin
