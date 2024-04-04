import logging
import torch
import joblib
import wandb
import typer
import pandas as pd
import asyncio
import sys 
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(root_path)

from transformers import DistilBertTokenizer
from datetime import datetime
from typer import Option
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from config import ENV_VARIABLES, MODELS_DIR
from pipeline.bert.utils import DistillBERTClass
from pipeline.llm.classifier import LlmClassifier
from pipeline.llm import load_yaml_dict, load_txt

def evaluate_model(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions, average='weighted')
    precision = precision_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, recall, precision, f1

def main(
    # Add arguments here
    test_data: str = Option(..., help="Path to the train data file"),
):
    try:
        date = datetime.now().strftime("%Y-%m-%d:%H")
        run_label = f"Evaluation_{date}"
        # Load BERT model
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        model_for_inference = DistillBERTClass()
        model_for_inference.load_state_dict(torch.load(MODELS_DIR / 'distilbert_finetuned.pth',map_location=torch.device('cuda'if torch.cuda.is_available() else "cpu")))
        model_for_inference.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_for_inference.to(device)
        logging.info("BERT model loaded successfully")

        # Load LLM model
        llm_classifier = LlmClassifier(
            chain_config=load_yaml_dict(ENV_VARIABLES["LLM_CONFIG_PATH"]),
            classes=load_txt(ENV_VARIABLES["CLASSES_PATH"])
        )
        logging.info("LLM model loaded successfully")

        # Load xgboost model
        xgb_classifier = joblib.load(ENV_VARIABLES["XGBOOST_MODEL_PATH"])
        vectorizer = joblib.load(ENV_VARIABLES["XGBOOST_VECTORIZER_PATH"])
        logging.info("XGBoost model loaded successfully")

        content = pd.read_csv(test_data)
        encode_dict = {'question': 0, 'therapist_input': 1, 'reflection': 2, 'other': 3}

        with wandb.init(
        project="therapist_behaviour_project", 
        config={}, 
        name=run_label) as run:
            loggin_records = []
            for index, row in content.iterrows():
                    line = row['text']
                    inputs = tokenizer(line, return_tensors="pt", truncation=True, padding=True, max_length=256)
                    # Move the tensors to the device of the model
                    inputs = {k: v.to(device) for k, v in inputs.items()}   
                    with torch.no_grad():
                        outputs = model_for_inference(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    bert_output = probabilities.argmax(dim=1).item()
                    llm_output = asyncio.run(llm_classifier.run(line))
                    llm_output = encode_dict[llm_output]
                    xgb_output = xgb_classifier.predict(vectorizer.transform([line]))[0]
                    logging.info(f"BERT: {bert_output}, LLM: {llm_output}, XGBoost: {xgb_output}")
                    loggin_records.append({
                        "line": line,
                        "bert_output": bert_output,
                        "llm_output": llm_output,
                        "xgb_output": xgb_output,
                        "true_label": row['encode_cat']
                    })
            results_df = pd.DataFrame(loggin_records)

            bert_accuracy, bert_recall, bert_precision, bert_f1 = evaluate_model(results_df['true_label'], results_df['bert_output'])
            llm_accuracy, llm_recall, llm_precision, llm_f1 = evaluate_model(results_df['true_label'], results_df['llm_output'])
            xgb_accuracy, xgb_recall, xgb_precision, xgb_f1 = evaluate_model(results_df['true_label'], results_df['xgb_output'])

            logging.info(f"BERT Model - Accuracy: {bert_accuracy}, Recall: {bert_recall}, Precision: {bert_precision}, F1: {bert_f1}")
            logging.info(f"LLM Model - Accuracy: {llm_accuracy}, Recall: {llm_recall}, Precision: {llm_precision}, F1: {llm_f1}")
            logging.info(f"XGBoost Model - Accuracy: {xgb_accuracy}, Recall: {xgb_recall}, Precision: {xgb_precision}, F1: {xgb_f1}")

            run.log({"individual_predicitons": wandb.Table(dataframe=results_df)})

            metrics_data = {
                "Model": ["BERT", "LLM", "XGBoost"],
                "Accuracy": [bert_accuracy, llm_accuracy, xgb_accuracy],
                "Recall": [bert_recall, llm_recall, xgb_recall],
                "Precision": [bert_precision, llm_precision, xgb_precision],
                "F1": [bert_f1, llm_f1, xgb_f1]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_table = wandb.Table(dataframe=metrics_df)
            run.log({"Evaluation Metrics": metrics_table})

            logging.info("Evaluation completed successfully")
    except Exception as e:
        logging.error(f"Error in lifespan: {e}")

if __name__ == "__main__":
    typer.run(main)