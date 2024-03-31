import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import DistilBertModel
from typing import Tuple

class Therapist(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        text = str(self.data.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.encode_cat[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len
    
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
        for param in self.l1.transformer.layer.parameters():
            param.requires_grad = False

        # for lin_param in self.l1.transformer.layer[5].ffn.lin2.parameters():
        #     lin_param.requires_grad = True

        self.linear_1 = torch.nn.Linear(768, 64)
        self.norm_1 = torch.nn.LayerNorm(64) 
        self.dropout = torch.nn.Dropout(0.7)
        self.classifier = torch.nn.Linear(64, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.dropout(pooler)
        pooler = self.linear_1(pooler)
        pooler = self.norm_1(pooler) 
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
def load_data(path)-> Tuple[pd.DataFrame, pd.DataFrame]:

    raw_df = pd.read_csv(path)

    df = raw_df[raw_df['interlocutor']=='therapist'][['utterance_text', 'main_therapist_behaviour']]
    df = df.rename(columns={'utterance_text':'text', 'main_therapist_behaviour':'category'})

    encode_dict = {}

    def encode_cat(x):
        if x not in encode_dict.keys():
            encode_dict[x]=len(encode_dict)
        return encode_dict[x]

    df['encode_cat'] = df['category'].apply(lambda x: encode_cat(x))

    df = df.drop_duplicates(subset=['text'])

    train_size = 0.9
    train_dataset, test_dataset = train_test_split(df, train_size=train_size, stratify=df['encode_cat'], random_state=200)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    return train_dataset, test_dataset

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train(epoch, loss_function, optimizer, model, training_loader, validation_loader, device, scheduler):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    n = 71
    model.train()

    for i,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if i%n==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per {n} steps: {loss_step:.3f},  --Training Accuracy per {n} steps: {accu_step:.3f}")
            

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    scheduler.step()

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        print(f"Current learning rate after epoch {epoch}: {current_lr}")

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples

    print(f"Epoch: {epoch} \n--Training Loss: {epoch_loss:.3f},  --Training Accuracy: {epoch_accu:.3f}")

    val_loss,val_accu = valid(model, validation_loader, loss_function, device)

    return epoch_loss, epoch_accu, val_loss, val_accu

def valid(model, testing_loader, loss_function, device):
    model.eval()
    n_correct = 0; tr_loss=0; nb_tr_steps = 0; nb_tr_examples = 0
    with torch.no_grad():
        for i, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
    
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"--Validation Loss: {epoch_loss:.3f},  --Validation Accuracy: {epoch_accu:.3f}")
    
    return epoch_loss, epoch_accu

def plot_results(results_df:pd.DataFrame):
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results_df['epoch'], results_df['train_loss'], label='Train Loss')
    plt.plot(results_df['epoch'], results_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(results_df['epoch'], results_df['train_accuracy'], label='Train Accuracy')
    plt.plot(results_df['epoch'], results_df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('data/training_validation_results.png')