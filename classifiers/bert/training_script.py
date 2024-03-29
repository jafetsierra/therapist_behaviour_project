import torch
import typer

from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from torch import cuda
from typing import Tuple

from .utils import Therapist, DistillBERTClass, train, valid, load_data


def main():

    device = 'cuda' if cuda.is_available() else 'cpu'

    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 12
    VALID_BATCH_SIZE = 12
    EPOCHS = 1
    LEARNING_RATE = 1e-05

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    train_dataset, test_dataset = load_data('data/therapist_behaviour.csv')

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

if __name__ == "__main__":

    typer.run(main)