# Therapist_behaviour_project for reflexai interview

## installation 

```
poetry install
```

## prerequisites

in order to correctly execute and run the application you need to prepare some packages/documents

generate train and test dataset from raw_file
```
poetry run python pipeline/process_data.py --data-path data/full.csv --train-output-path data/train.csv --test-output-path data/test.csv
```

context file generation for LLM classifier:
```
poetry run python pipeline/llm/generate_context.py --data-path data/train.csv --output-path data/context.txt
```
Bert fin-tunned model weigths for Bert type classifier model. If you already have the model trained you only have to specify the directory of the model's weights in the .env file. Otherwise you will have to train you own version using this useful training script:
```
poetry run 
```