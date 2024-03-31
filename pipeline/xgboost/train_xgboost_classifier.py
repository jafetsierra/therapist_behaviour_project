import pandas as pd
import typer
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from typer import Option

# Function to load and preprocess data
def load_and_preprocess_data(filepath, text_column, label_column):
    df = pd.read_csv(filepath)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[text_column])
    y = df[label_column]
    return X, y, vectorizer

# Function to train the model
def train_model(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    return model

# Main training function
def train_and_save_model(
        train_data_path = Option(..., help="Path to the training data"),
        text_column = Option(..., help="Path to the training data"),
        label_column = Option(..., help="Path to the training data"),
        output_model_path = Option(..., help="Path to the training data"), 
        output_vectorizer_path = Option(..., help="Path to the training data")
        ):
    X, y, vectorizer = load_and_preprocess_data(train_data_path, text_column, label_column)
    model = train_model(X, y)

    # Save the model and vectorizer
    joblib.dump(model, output_model_path)
    joblib.dump(vectorizer, output_vectorizer_path)
    print(f"Model and vectorizer saved to {output_model_path} and {output_vectorizer_path} respectively.")

if __name__ == "__main__":
    typer.run(train_and_save_model)