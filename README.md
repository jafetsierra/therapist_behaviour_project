# Therapist_behaviour_project for reflexai interview

## Installation 

```
poetry install
```

## Prerequisites

in order to correctly execute and run the application you need to prepare some packages/documents

1. generate train and test dataset from raw_file
```
poetry run python pipeline/process_data.py --data-path data/full.csv --train-output-path data/train.csv --test-output-path data/test.csv
```
2. you must have a qdrant client to connect or you can simply run one locally with:
```
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

3. Upload train examples to qdrant vector DB
```
poetry run python pipeline/llm/upload_to_qdrant.py --data-path data/train.csv --collection-name therapist_behaviour
```


4. Bert fin-tunned model weigths for Bert type classifier model. If you already have the model trained you only have to specify the directory of the model's weights in the .env file. Otherwise you will have to train you own version using this useful training script:

```
poetry run python pipeline/bert/training_script.py --train-data-path data/train.csv --test-data-path data/test.csv --max-len 256 --train-batch-size 32 --valid-batch-size 12 --epochs 20 --learning-rate 1e-4 --output-path models/distilbert_finetuned.pth
```
5. Xgboost classifier
you would need to specify the path to the model checkpoint in .env variables. if you dont have any, you can train one using this convenient script:
```
poetry run python pipeline/xgboost/train_xgboost_classifier.py --train-data-path data/train.csv --text-column text --label-column encode_cat --output-model-path models/xgboost_classifier.pkl --output-vectorizer-path models/tfidf_vectorizer.pkl
```

## Use
Once you have all the prerequisites you can try the App locally using:
```
poetry run uvicorn app.api:app --reload
```

## Docker use
Create Docker image
```
make build_amd64
```
Or
```
make build_arm64
```
Dependeing on you architecture type.

Finally, update you .env variable QDRANT_HOST=host.docker.internal  And run docker container with:

 ```
 docker compose up
 ```