import torch
import typer
import pandas as pd

from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from torch import cuda
from typing import Annotated

from .utils import Therapist, DistillBERTClass, train, valid, load_data


def main(
    train_data_path: Annotated[str, typer.Option(help="Path to the train data file")],
    test_data_path: Annotated[str, typer.Option(help="Path to the test data file")],
    MAX_LEN: Annotated[int, typer.Option(default=512,help="Maximum length of the input")],
    TRAIN_BATCH_SIZE: Annotated[int, typer.Option(default=12,help="Training batch size")],
    VALID_BATCH_SIZE: Annotated[int, typer.Option(default=12,help="Validation batch size")],
    EPOCHS: Annotated[int, typer.Option(default=3,help="Number of epochs")],
    LEARNING_RATE: Annotated[float, typer.Option(default=1e-5,help="Learning rate")],
    OUTPUT_PATH: Annotated[str, typer.Option(help="Path to save the model")]
):

    device = 'cuda' if cuda.is_available() else 'cpu'

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    train_dataset = pd.read_csv(train_data_path)
    test_dataset = pd.read_csv(test_data_path)

    training_set = Therapist(train_dataset, tokenizer, MAX_LEN) 
    testing_set = Therapist(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = DistillBERTClass()
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train(epoch, loss_function, optimizer, model, training_loader, device)

    acc = valid(model, testing_loader, loss_function, device)
    print("Accuracy on test data = %0.2f%%" % acc)

    torch.save(model.state_dict(), OUTPUT_PATH)

if __name__ == "__main__":

    typer.run(main)