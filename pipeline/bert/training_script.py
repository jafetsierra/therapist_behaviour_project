import torch
import typer
import pandas as pd
import os
import sys

from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from torch import cuda
from typer import Option
from torch.optim.lr_scheduler import StepLR

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(root_path)

from .utils import Therapist, DistillBERTClass, train, valid, plot_results, get_weights
from config import DATA_DIR

def main(
    train_data_path: str = Option(..., help="Path to the train data file"),
    test_data_path: str = Option(..., help="Path to the test data file"),
    max_len: int = Option(512, help="Maximum length of the input"),
    train_batch_size: int = Option(12, help="Training batch size"),
    valid_batch_size: int = Option(12, help="Validation batch size"),
    epochs: int = Option(3, help="Number of epochs"),
    learning_rate: float = Option(1e-3, help="Learning rate"),
    output_path: str = Option(..., help="Path to save the model")
):

    results = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    device = 'cuda' if cuda.is_available() else 'cpu'

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

    train_dataset = pd.read_csv(train_data_path)
    test_dataset = pd.read_csv(test_data_path)

    training_set = Therapist(train_dataset, tokenizer, max_len) 
    testing_set = Therapist(test_dataset, tokenizer, max_len)

    train_params = {'batch_size': train_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': valid_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = DistillBERTClass()
    model.to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    weights = torch.tensor(get_weights(train_dataset), dtype=torch.float).to(device)

    loss_function = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
    # loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)

    min_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_limit = 2  

    for epoch in range(epochs):
        train_loss, train_accu, val_loss, val_accu = train(epoch, loss_function, optimizer, model, training_loader, testing_loader, device, scheduler)
        results['epoch'].append(epoch)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accu)
        results['val_loss'].append(val_loss)
        results['val_accuracy'].append(val_accu)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), output_path)  # Save best model
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter > early_stop_limit:
            print("Early stopping triggered")
            break

    test_loss, test_acc = valid(model, testing_loader, loss_function, device)
    print("loss on test data = %0.2f" % test_loss)
    print("Accuracy on test data = %0.2f%%" % test_acc)

    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_DIR / 'training.csv')
    plot_results(results_df)

    torch.save(model.state_dict(), output_path)

if __name__ == "__main__":

    typer.run(main)



